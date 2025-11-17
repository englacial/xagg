#!/usr/bin/env python3
"""
Alternative approach: Query NASA CMR directly (not STAC) for ICESat-2 ATL06 data
with support for orbital cycle and region filtering via native CMR parameters.

The CMR API has better native support for ICESat-2 orbital parameters than CMR-STAC.
"""

import requests
from typing import List, Optional, Dict
import geopandas as gpd
from shapely.geometry import box
import pandas as pd


def query_atl06_cmr(
    cycle: int,
    regions: List[int],
    rgts: Optional[List[int]] = None,
    version: str = "006",
    provider: str = "NSIDC_CPRD",
    page_size: int = 2000,
) -> gpd.GeoDataFrame:
    """
    Query NASA CMR directly for ATL06 data with cycle and region filtering.
    
    Parameters
    ----------
    cycle : int
        Orbital cycle number (e.g., 22)
    regions : List[int]
        List of granule region numbers (1-14), e.g., [10, 11, 12]
    rgts : Optional[List[int]], optional
        List of specific Reference Ground Tracks to filter, by default None
    version : str, optional
        ATL06 version, by default "006"
    provider : str, optional
        CMR provider, by default "NSIDC_CPRD" (cloud-hosted)
    page_size : int, optional
        Number of results per page, by default 2000
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with granule metadata and geometries
    """
    
    # CMR granule search endpoint
    cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
    
    # Build query parameters
    params = {
        "provider": provider,
        "short_name": "ATL06",
        "version": version,
        "page_size": page_size,
        "sort_key": "start_date",
    }
    
    # Add cycle parameter
    # CMR supports cycle as a query parameter for ICESat-2 products
    params["cycle"] = str(cycle)
    
    # Note: CMR doesn't have a direct "region" parameter
    # We'll need to filter by filename pattern after retrieval
    
    print(f"Querying CMR for ATL06 v{version}:")
    print(f"  Provider: {provider}")
    print(f"  Cycle: {cycle}")
    print(f"  Regions: {regions}")
    if rgts:
        print(f"  RGTs: {rgts}")
    
    all_granules = []
    headers = {"Accept": "application/vnd.nasa.cmr.umm_json+json"}
    
    # Fetch all pages
    while True:
        response = requests.get(cmr_url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        items = data.get("items", [])
        
        if not items:
            break
        
        all_granules.extend(items)
        
        # Check if there are more pages
        if len(items) < page_size:
            break
        
        # Get next page token
        # CMR uses scroll_id for pagination in UMM-JSON
        # Update params with the CMR-Scroll-Id from headers
        scroll_id = response.headers.get("CMR-Scroll-Id")
        if scroll_id:
            params["scroll_id"] = scroll_id
        else:
            break
    
    print(f"Retrieved {len(all_granules)} granules from CMR")
    
    # Filter by region and optionally by RGT from granule names
    filtered_granules = []
    for granule in all_granules:
        umm = granule.get("umm", {})
        granule_ur = umm.get("GranuleUR", "")
        
        # Parse granule filename
        # Format: ATL06_YYYYMMDDhhmmss_ttttccnn_rrr_vv
        try:
            parts = granule_ur.split("_")
            if len(parts) >= 3:
                rgt_cycle_region = parts[2]
                
                # Extract RGT (first 4 digits)
                granule_rgt = int(rgt_cycle_region[0:4])
                
                # Extract cycle (next 2 digits)
                granule_cycle = int(rgt_cycle_region[4:6])
                
                # Extract region (last 2 digits)
                granule_region = int(rgt_cycle_region[6:8])
                
                # Filter by region
                if granule_region not in regions:
                    continue
                
                # Filter by RGT if specified
                if rgts and granule_rgt not in rgts:
                    continue
                
                filtered_granules.append(granule)
                
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse granule UR {granule_ur}: {e}")
            continue
    
    print(f"Filtered to {len(filtered_granules)} granules matching regions {regions}")
    
    # Convert to GeoDataFrame
    records = []
    for granule in filtered_granules:
        umm = granule.get("umm", {})
        
        # Get granule ID
        granule_id = umm.get("GranuleUR", "")
        
        # Get bounding box from spatial extent
        spatial_extent = umm.get("SpatialExtent", {})
        horiz_spatial = spatial_extent.get("HorizontalSpatialDomain", {})
        geometry_obj = horiz_spatial.get("Geometry", {})
        bounding_rectangles = geometry_obj.get("BoundingRectangles", [])
        
        if not bounding_rectangles:
            continue
        
        # Use first bounding rectangle
        bbox_dict = bounding_rectangles[0]
        west = bbox_dict.get("WestBoundingCoordinate", 0)
        south = bbox_dict.get("SouthBoundingCoordinate", 0)
        east = bbox_dict.get("EastBoundingCoordinate", 0)
        north = bbox_dict.get("NorthBoundingCoordinate", 0)
        
        # Create geometry
        geom = box(west, south, east, north)
        
        # Get temporal info
        temporal = umm.get("TemporalExtent", {})
        range_date_times = temporal.get("RangeDateTime", {})
        begin_date = range_date_times.get("BeginningDateTime", "")
        end_date = range_date_times.get("EndingDateTime", "")
        
        # Get URLs from related URLs
        related_urls = umm.get("RelatedUrls", [])
        data_urls = []
        for url_obj in related_urls:
            url_type = url_obj.get("Type", "")
            if "GET DATA" in url_type:
                data_urls.append(url_obj.get("URL", ""))
        
        # Parse granule components from filename
        parts = granule_id.split("_")
        rgt_cycle_region = parts[2] if len(parts) > 2 else ""
        granule_rgt = int(rgt_cycle_region[0:4]) if len(rgt_cycle_region) >= 4 else None
        granule_region = int(rgt_cycle_region[6:8]) if len(rgt_cycle_region) >= 8 else None
        
        record = {
            "granule_id": granule_id,
            "rgt": granule_rgt,
            "cycle": cycle,
            "region": granule_region,
            "bbox_west": west,
            "bbox_south": south,
            "bbox_east": east,
            "bbox_north": north,
            "geometry": geom,
            "begin_datetime": begin_date,
            "end_datetime": end_date,
            "urls": data_urls,
            "n_urls": len(data_urls),
        }
        records.append(record)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    
    return gdf


def save_to_geoparquet(gdf: gpd.GeoDataFrame, output_path: str):
    """Save GeoDataFrame to GeoParquet format."""
    gdf_copy = gdf.copy()
    gdf_copy["urls"] = gdf_copy["urls"].apply(lambda x: "|".join(x) if x else "")
    gdf_copy.to_parquet(output_path, index=False)
    print(f"\nSaved {len(gdf_copy)} records to {output_path}")


if __name__ == "__main__":
    # Query for cycle 22, regions 10-12
    cycle = 22
    regions = [10, 11, 12]
    
    gdf = query_atl06_cmr(
        cycle=cycle,
        regions=regions,
        version="006",
        provider="NSIDC_CPRD",
    )
    
    # Display results
    print("\nResults summary:")
    print(f"Total granules: {len(gdf)}")
    
    if len(gdf) > 0:
        print(f"\nRGT distribution:")
        print(gdf["rgt"].value_counts().sort_index().head(10))
        
        print(f"\nRegion distribution:")
        print(gdf["region"].value_counts().sort_index())
        
        print(f"\nSample granules:")
        print(gdf[["granule_id", "rgt", "region", "bbox_west", "bbox_north"]].head(10))
        
        # Save to GeoParquet
        output_file = f"/mnt/user-data/outputs/atl06_cycle{cycle}_regions_{'_'.join(map(str, regions))}_cmr.parquet"
        save_to_geoparquet(gdf, output_file)
