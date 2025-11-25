# Multi-resolution Aggregation

This is prototype of an aggregation library. We have the following axioms:

  1. Data selection is declarative using STAC
  2. Aggregation doesn't duplicate the at rest data

Number 2 above doesn't imply that we can't fetch the files to do the
aggregation- rather, that we don't reprocess and save the entire dataset.

## Sparse point data -- why is this hard?

Fundamentally, we are looking to cast point data (ATL0\*, OPR, etc) to a grid.
We run into two issues immediately:

  1. Too dense-- multiple points per cell
  2. Too sparse-- no points for many cells

The latter of these issues is annoying, but we can work around. It's really the
former issue that's a showstopper-- xarray doesn't have a way to handle
collisions, so we have to have define what to do with these overlapping
observations.

Hence the 'agg' in xagg.

## Prior art

Previously when I worked for the ICESat-2 project science office, we tackled
this problem using hierarchical indexing-- i.e., resharding the atl06 dataset
using healpix based morton indexing to files stored in hive format. This avoided
the hard issue of multiple points per cell because we kept the data a columnar
data type, which could be rasterized on the fly using vaex. However, it failed
axiom #2, since it required duplicating and storing atl06 to a new 'at rest
format'.

This duplication was a requirement because the resharding process occurred on 91
day 'chunks' of orbital repeats, and had non-trivial computational requirements
that weren't realistic to process on the fly for users. Specifically, since we
were building the spatial tree from the root to the leaves, we had to task high
memory nodes to get the data into memory for the initial sharding.

## A new approach: building the tree from the leaves instead of from the root

To meet axiom #2, we switch the order of the tree construction. This also lets
us build the tree in parallel using small, commodity worker nodes that are
cheap. This makes the process easy to replicate, and since we can scale very
wide when doing horizontal scaling, reasonably performant in terms of wall clock
time. Since the process can be done 'on the fly', we can purge any downloaded or
replicated source data granules as temp files. Also, since the aggregation
output is significantly compacted (via the aggregations), it's not unreasonable
to save that output if we would like to.

### Vocabulary

Some basic terminology definitions:

  - **Base Aggregation Cell** : This is likely to be shorted to 'cell' or 'base
    cell', but both of those other terms are confusing. 'Base cell' has a
    specific meaning in healpix which refers to one of the 12 top level cells at
    the root of the tree-- this is contrast to our 'base aggregation cells'
    which are the finest resolution of aggregation (i.e., the leaf nodes).
    Similarly, using 'cell' in isolation ignores the fact that we are defining a
    hierarchy of cells at different resolutions. A 'base aggregation cell' is
    the finest level of aggregation, and is hence defines the resolution of the
    cells in our 'shards'.
  - **Shard** : A 'shard' is the lowest level of chunking, and is thus also a
    'chunk'. We call it a 'shard' indicate that it is not divisible; there are
    no 'chunks' under the 'shards', just the raw original data. This also makes
    the shards (potentially) unique in that they encode explicit links to the
    underlying raw granules.

### Basic functions

Our top level aggregation function needs to know the following things:

  1. What is the input data? (This is defined by the STAC catalog)
  2. What is the aggregation function? (This will likely be defined by a yml
     template)
  3. What is the base aggregation cell size?
  4. What is the shard size (in both coverage, and extent)

Item's 3 and 4 will likely be passed to functions by the template in 2. In
practice, we could also programmatically set the shard size based on the base
aggregation size (or at least pass a sensible default). The 'coverage' of the
shard refers to what order of morton index it covers on the earth-- i.e., it's
*spatial extent*. The 'extent' of a shard refers to how many base cells are
contained within it, i.e., it's *data extent* in terms of number of rows and
columns.

Our aggregation functions do two things:

  1. Chunk the input dataset into file groups that correspond to shard coverage.
  2. Read in the file groups, and aggregate that data into cells within the
     given shards.

We have to do item 1 as a prerequisite to doing number 2. However, this
preprocessing step of 'chunking' input data is a common design pattern, and
there will likely be users that want just that functionality by itself to
integrate in their own processing workflows.

### Choosing shard and base aggregation cell sizes

The base aggregation cell is probably easier to decide and define that the shard
size. For ICESat-2, the input data has beam pairs that are spaced 3.3 kilometers
apart, and we'd probably like to match to that-- i.e., either 1.5, 3, 6, or 12
kilometers on a side. For the shard size, it will depend on base cell-- we'll
need to be able to read all of the data into memory at the shard level, so we
don't want them to be too big. We'd also like shard outputs to have enough cells
to be useful raster arrays.

For our prototype, we'll start with a fairly fine base cell resolution of 1.5
km (order=12). For shard aggregation, it probably makes sense to go to either
50km (i.e., 1024 cells), or 100km (4096);  we'll start with 100km for the
prototype and adjust as needed for memory.

### Determining the shards to iterate through

Determining the shards themselves is easy, and fast:

```python
antart = pd.read_csv('Ant_Grounded_DrainageSystem_Polygons.txt', names=['Lat','Lon','basin'], delim_whitespace=True)
shards = np.unique(geo2mort(antart.Lat.values, antart.Lon.values, order=6))
```
At this shard size, we have 644 shards; we'd have 269 at order 5, or 1464 at
order 7. The main constraints here are memory; i.e., what's a shard actually
need to read in to execute an aggregation? To be more explicit, what are the
files that will match to each individual shard?

Note that the above code will only determine which shards are needed for
covering the drainage basin boundaries-- to determine what the overall coverage
is for the polygons (i.e., the boundaries and internal area) we need to know the
what the minimum spanning tree is that covers the polygon:

```python
morton_greedy, _ = greedy_morton_polygon(
    antart['Lat'].values,
    antart['Lon'].values,
    order=18,
    max_boxes=25,
    ordermax=6,
    verbose=False
)

all_order6_cells = []
for morton in morton_greedy:
    # Generate children at order 6 for this parent morton
    children = generate_morton_children(morton, target_order=6)
    all_order6_cells.extend(children)

# Get unique order-6 cells
shards = np.unique(np.array(all_order6_cells))
```
