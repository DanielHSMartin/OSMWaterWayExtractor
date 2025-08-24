#!/usr/bin/env python3
"""
OSM Waterway Extraction and Graph Minimization Script (Fixed Version)

This script extracts waterway data from OpenStreetMap PBF files and creates
a minimized graph representation suitable for offline mobile applications.

Usage:
    python osm_waterway_extractor_fixed.py <input.osm.pbf> [--min-length METERS] [--precision DECIMALS]

Example:
    python osm_waterway_extractor_fixed.py brazil-latest.osm.pbf --min-length 100 --precision 5

Requirements:
    pip install osmium pyproj shapely geopandas
"""

import argparse
import json
import gzip
import os
import sys
import math
from pathlib import Path
from collections import Counter
import logging
import concurrent.futures
import numpy as np

# Check for required libraries and provide installation instructions
try:
    import osmium
    import osmium.geom
except ImportError:
    print("Error: osmium library not found.")
    print("Install with: pip install osmium")
    sys.exit(1)

try:
    from shapely.geometry import LineString, Point
    from shapely.ops import linemerge
except ImportError:
    print("Error: shapely library not found.")
    print("Install with: pip install shapely")
    sys.exit(1)

try:
    import pyproj
    from pyproj import Geod
except ImportError:
    print("Error: pyproj library not found.")
    print("Install with: pip install pyproj")
    sys.exit(1)

try:
    import geopandas as gpd
    import pandas as pd
except ImportError:
    print("Error: geopandas library not found.")
    print("Install with: pip install geopandas")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WaterwayHandler(osmium.SimpleHandler):
    """OSM handler to extract waterway data (rivers and streams only)."""
    
    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.waterways = []
        # Use osmium's WKB factory to get geometries
        self.wkb_factory = osmium.geom.WKBFactory()
        
    def way(self, w):
        """Extract ways that are rivers or streams."""
        tags = {tag.k: tag.v for tag in w.tags}
        
        # Only process waterways that are rivers or streams
        if tags.get('waterway') in ['river', 'stream']:
            try:
                # Get the geometry using osmium's WKB factory
                wkb = self.wkb_factory.create_linestring(w)
                if wkb:
                    # Convert WKB to shapely geometry
                    from shapely import wkb as shapely_wkb
                    line = shapely_wkb.loads(wkb)
                    
                    # Extract coordinates
                    coords = list(line.coords)
                    
                    if len(coords) >= 2:  # Valid waterway needs at least 2 points
                        self.waterways.append({
                            'id': w.id,
                            'coordinates': coords,
                            'tags': tags
                        })
                        
            except Exception as e:
                # Skip ways that can't be processed (e.g., incomplete geometry)
                logger.debug(f"Skipping way {w.id}: {e}")


def process_single_waterway(waterway, simplify_tolerance, keep_intermediate_distance, coordinate_precision):
    from shapely.geometry import LineString
    def round_coordinates(lat, lon):
        return (round(lat, coordinate_precision), round(lon, coordinate_precision))
    coords = waterway['coordinates']
    if len(coords) < 2:
        return None
    if simplify_tolerance is not None and simplify_tolerance > 0:
        line = LineString(coords)
        simplified = line.simplify(simplify_tolerance, preserve_topology=False)
        coords = list(simplified.coords)
        if len(coords) < 2:
            return None
    rounded_coords = []
    for lon, lat in coords:
        rounded_coord = round_coordinates(lat, lon)
        rounded_coords.append(rounded_coord)
    # Add intermediate nodes at regular intervals if requested
    if keep_intermediate_distance and len(rounded_coords) > 5:
        # Use the same logic as WaterwayGraphBuilder.add_intermediate_nodes
        result = [rounded_coords[0]]
        cumulative_distance = 0
        last_added_distance = 0
        for i in range(1, len(rounded_coords)):
            lat1, lon1 = rounded_coords[i-1][0], rounded_coords[i-1][1]
            lat2, lon2 = rounded_coords[i][0], rounded_coords[i][1]
            lat_diff = lat2 - lat1
            lon_diff = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
            approx_distance = math.sqrt(lat_diff**2 + lon_diff**2) * 111000
            cumulative_distance += approx_distance
            if cumulative_distance - last_added_distance >= keep_intermediate_distance:
                result.append(rounded_coords[i])
                last_added_distance = cumulative_distance
            if i == len(rounded_coords) - 1:
                if rounded_coords[i] != result[-1]:
                    result.append(rounded_coords[i])
        rounded_coords = result
    return rounded_coords


# Top-level numpy segment length calculation
import numpy as np

def calculate_segment_length_numpy(coords):
    """Vectorized segment length calculation using numpy."""
    if len(coords) < 2:
        return 0
    arr = np.array(coords)
    lat = arr[:, 0]
    lon = arr[:, 1]
    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    # Approximate distance between consecutive points
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)
    lat_avg = (lat_rad[:-1] + lat_rad[1:]) / 2
    # 1 degree ≈ 111km, adjust for latitude
    dx = dlat * 111000
    dy = dlon * 111000 * np.cos(lat_avg)
    dist = np.sqrt(dx**2 + dy**2)
    return float(np.sum(dist))


# Top-level numpy segment length calculation (optimized version)
def calculate_segment_length_vectorized(coords_list):
    """Vectorized segment length calculation for multiple coordinate lists."""
    if not coords_list:
        return []
    
    lengths = []
    for coords in coords_list:
        if len(coords) < 2:
            lengths.append(0)
            continue
            
        arr = np.array(coords)
        lat = arr[:, 0]
        lon = arr[:, 1]
        
        # Convert degrees to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Approximate distance between consecutive points
        dlat = np.diff(lat_rad)
        dlon = np.diff(lon_rad)
        lat_avg = (lat_rad[:-1] + lat_rad[1:]) / 2
        
        # 1 degree ≈ 111km, adjust for latitude
        dx = dlat * 111000
        dy = dlon * 111000 * np.cos(lat_avg)
        lengths.append(float(np.sum(np.sqrt(dx**2 + dy**2))))
    
    return lengths


def process_edges_for_waterway(args):
    waterway_coords, important_nodes, min_length_meters, critical_nodes = args
    edges = []
    if len(waterway_coords) < 2:
        return edges
    
    # Pre-compute important node positions for faster lookups
    important_positions = set()
    for i, coord in enumerate(waterway_coords):
        if coord in important_nodes:
            important_positions.add(i)
    
    current_start_idx = 0
    
    for i in range(1, len(waterway_coords)):
        coord = waterway_coords[i]
        is_last = i == len(waterway_coords) - 1
        
        if i in important_positions or is_last:
            # Extract segment coordinates efficiently
            segment_coords = waterway_coords[current_start_idx:i+1]
            
            if len(segment_coords) >= 2:
                segment_length = calculate_segment_length_numpy(segment_coords)
                
                # Check if we should keep this edge (optimized critical node check)
                if segment_length >= min_length_meters or coord in critical_nodes:
                    edges.append({
                        'start': segment_coords[0],
                        'end': segment_coords[-1],
                        'length': segment_length,
                        'coordinates': segment_coords  # No copy needed, slice is already new
                    })
            
            current_start_idx = i
    
    return edges


def process_edges_batch(batch_args):
    """Process a batch of waterways for edge creation - optimized version."""
    edges = []
    for args in batch_args:
        waterway_coords, important_nodes, min_length_meters, critical_nodes = args
        if len(waterway_coords) < 2:
            continue
        
        # Pre-compute important node positions for faster lookups
        important_positions = set()
        for i, coord in enumerate(waterway_coords):
            if coord in important_nodes:
                important_positions.add(i)
        
        current_start_idx = 0
        
        for i in range(1, len(waterway_coords)):
            coord = waterway_coords[i]
            is_last = i == len(waterway_coords) - 1
            
            if i in important_positions or is_last:
                # Extract segment coordinates efficiently
                segment_coords = waterway_coords[current_start_idx:i+1]
                
                if len(segment_coords) >= 2:
                    segment_length = calculate_segment_length_numpy(segment_coords)
                    
                    # Check if we should keep this edge
                    if segment_length >= min_length_meters or coord in critical_nodes:
                        edges.append({
                            'start': segment_coords[0],
                            'end': segment_coords[-1],
                            'length': segment_length,
                            'coordinates': segment_coords
                        })
                
                current_start_idx = i
    
    return edges


def process_edges_for_waterway_vectorized(args):
    """Optimized edge processing using faster important node lookups."""
    waterway_coords, important_nodes, min_length_meters, critical_nodes = args
    if len(waterway_coords) < 2:
        return []
    
    # Find important positions using faster approach - avoid numpy array creation for set lookups
    important_positions = []
    for i, coord in enumerate(waterway_coords):
        if coord in important_nodes:
            important_positions.append(i)
    
    # Always include the last index
    if not important_positions or important_positions[-1] != len(waterway_coords) - 1:
        important_positions.append(len(waterway_coords) - 1)
    
    edges = []
    start_idx = 0
    
    for end_idx in important_positions:
        if end_idx > start_idx:
            segment_coords = waterway_coords[start_idx:end_idx + 1]
            
            if len(segment_coords) >= 2:
                # Use fast vectorized distance calculation
                segment_length = calculate_segment_length_numpy(segment_coords)
                end_coord = waterway_coords[end_idx]
                
                # Check if we should keep this edge
                if segment_length >= min_length_meters or end_coord in critical_nodes:
                    edges.append({
                        'start': segment_coords[0],
                        'end': segment_coords[-1],
                        'length': segment_length,
                        'coordinates': segment_coords
                    })
            
            start_idx = end_idx
    
    return edges


def process_edges_for_waterway_fast(args):
    """Ultra-fast edge processing using pre-computed coordinate indices."""
    waterway_coords, important_indices, min_length_meters, critical_nodes = args
    if len(waterway_coords) < 2:
        return []
    
    edges = []
    current_start_idx = 0
    
    # Process only the pre-computed important indices
    for end_idx in important_indices:
        if end_idx > current_start_idx:
            segment_coords = waterway_coords[current_start_idx:end_idx + 1]
            
            if len(segment_coords) >= 2:
                segment_length = calculate_segment_length_numpy(segment_coords)
                end_coord = waterway_coords[end_idx]
                
                # Check if we should keep this edge
                if segment_length >= min_length_meters or end_coord in critical_nodes:
                    edges.append({
                        'start': segment_coords[0],
                        'end': segment_coords[-1],
                        'length': segment_length,
                        'coordinates': segment_coords
                    })
            
            current_start_idx = end_idx
    
    return edges


def preprocess_waterways_for_fast_edge_creation(waterway_endpoints, important_nodes):
    """Pre-process waterways to find important node indices for ultra-fast processing."""
    print("Preprocessing waterways for fast edge creation...")
    
    # Convert important_nodes to a more efficient lookup structure
    important_nodes_set = set(important_nodes)
    
    processed_args = []
    processed_count = 0
    
    # Process in smaller batches to show progress less frequently
    batch_size = 1000
    total_waterways = len(waterway_endpoints)
    
    # Only show progress every 20% (5 times maximum)
    progress_interval = max(total_waterways // 5, 50000)
    
    for i in range(0, total_waterways, batch_size):
        batch_end = min(i + batch_size, total_waterways)
        batch = waterway_endpoints[i:batch_end]
        
        for waterway_coords in batch:
            # Find indices of important nodes in this waterway using the optimized set
            important_indices = []
            for j, coord in enumerate(waterway_coords):
                if coord in important_nodes_set:
                    important_indices.append(j)
            
            # Always include the last index if not already included
            if not important_indices or important_indices[-1] != len(waterway_coords) - 1:
                important_indices.append(len(waterway_coords) - 1)
            
            processed_args.append((waterway_coords, important_indices))
            processed_count += 1
        
        # Show progress less frequently
        if processed_count % progress_interval == 0:
            percentage = (processed_count / total_waterways) * 100
            print(f"Preprocessed {processed_count:,}/{total_waterways:,} waterways ({percentage:.0f}%)")
    
    print(f"Preprocessing complete! Processed {len(waterway_endpoints):,} waterways")
    return processed_args


def create_edges_super_fast(waterway_endpoints, junctions, endpoints, min_length_meters):
    """Super-fast edge creation using only start/end points and known junctions."""
    print("Creating edges using super-fast approach...")
    
    # Convert junctions and endpoints to sets for O(1) lookup
    junction_set = set(junctions)
    endpoint_set = set(endpoints)
    critical_nodes = junction_set | endpoint_set
    
    edges = []
    processed_count = 0
    total_waterways = len(waterway_endpoints)
    
    # Show progress only 3 times: 33%, 66%, 100%
    progress_interval = max(total_waterways // 3, 200000)
    
    for waterway_coords in waterway_endpoints:
        processed_count += 1
        if processed_count % progress_interval == 0:
            percentage = (processed_count / total_waterways) * 100
            print(f"Edge creation progress: {percentage:.0f}% ({processed_count:,}/{total_waterways:,} waterways)")
        
        if len(waterway_coords) < 2:
            continue
        
        # For each waterway, create a single edge from start to end
        # unless there are junctions in between
        start_coord = waterway_coords[0]
        end_coord = waterway_coords[-1]
        
        # Find any junctions along this waterway (excluding start/end)
        junction_indices = []
        for i, coord in enumerate(waterway_coords[1:-1], 1):  # Skip first and last
            if coord in junction_set:
                junction_indices.append(i)
        
        # Create edges between important points
        segment_start_idx = 0
        important_indices = junction_indices + [len(waterway_coords) - 1]  # Add end index
        
        for end_idx in important_indices:
            if end_idx > segment_start_idx:
                segment_coords = waterway_coords[segment_start_idx:end_idx + 1]
                
                if len(segment_coords) >= 2:
                    segment_length = calculate_segment_length_numpy(segment_coords)
                    segment_end_coord = waterway_coords[end_idx]
                    
                    # Keep this edge if it meets length requirement or ends at critical node
                    if segment_length >= min_length_meters or segment_end_coord in critical_nodes:
                        edges.append({
                            'start': segment_coords[0],
                            'end': segment_coords[-1],
                            'length': segment_length,
                            'coordinates': segment_coords
                        })
                
                segment_start_idx = end_idx
    
    print(f"Edge creation complete! Created {len(edges):,} edges")
    return edges


class WaterwayGraphBuilder:
    """Build and minimize waterway network graph, with geometry simplification."""
    
    def __init__(self, min_length_meters=100, coordinate_precision=5, simplify_tolerance=None, keep_intermediate_distance=None):
        self.min_length_meters = min_length_meters
        self.coordinate_precision = coordinate_precision
        self.geod = Geod(ellps='WGS84')  # For accurate distance calculations
        self.simplify_tolerance = simplify_tolerance
        self.keep_intermediate_distance = keep_intermediate_distance  # Distance in meters to add intermediate nodes
    
    def round_coordinates(self, lat, lon):
        """Round coordinates to specified precision."""
        return (
            round(lat, self.coordinate_precision),
            round(lon, self.coordinate_precision)
        )
    
    def calculate_distance(self, coord1, coord2):
        """Calculate distance between two coordinates in meters."""
        # coord format is (lon, lat) from shapely, but geod expects (lon, lat, lon, lat)
        _, _, distance = self.geod.inv(coord1[0], coord1[1], coord2[0], coord2[1])
        return distance
    
    def calculate_segment_length_fast(self, coords):
        """Calculate segment length using fast approximation."""
        if len(coords) < 2:
            return 0
        
        total_length = 0
        for i in range(len(coords) - 1):
            # Use approximate distance calculation for speed
            lat1, lon1 = coords[i][0], coords[i][1]
            lat2, lon2 = coords[i + 1][0], coords[i + 1][1]
            
            # Rough approximation: 1 degree ≈ 111km, adjust for latitude
            lat_diff = lat2 - lat1
            lon_diff = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
            approx_distance = math.sqrt(lat_diff**2 + lon_diff**2) * 111000  # Convert to meters
            
            total_length += approx_distance
        
        return total_length
    
    def add_intermediate_nodes(self, coords, distance_meters):
        """Add nodes at regular distance intervals along a waterway (optimized)."""
        if not coords or len(coords) < 2 or not distance_meters:
            return coords
        
        # For very short waterways, don't add intermediate nodes
        if len(coords) < 10:  # Skip for short waterways
            return coords
            
        result = [coords[0]]  # Always keep the first point
        cumulative_distance = 0
        last_added_distance = 0
        
        # Use simpler distance calculation for better performance
        for i in range(1, len(coords)):
            # Approximate distance using simple lat/lon differences (faster than geodesic)
            lat1, lon1 = coords[i-1][0], coords[i-1][1]
            lat2, lon2 = coords[i][0], coords[i][1]
            
            # Rough approximation: 1 degree ≈ 111km, adjust for latitude
            lat_diff = lat2 - lat1
            lon_diff = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
            approx_distance = math.sqrt(lat_diff**2 + lon_diff**2) * 111000  # Convert to meters
            
            cumulative_distance += approx_distance
            
            # Check if we should add an intermediate node
            if cumulative_distance - last_added_distance >= distance_meters:
                result.append(coords[i])
                last_added_distance = cumulative_distance
            
            # Always add the last point
            if i == len(coords) - 1:
                if coords[i] != result[-1]:  # Avoid duplicates
                    result.append(coords[i])
        
        return result
    
    def build_graph(self, waterways):
        """Build graph from waterway coordinates, with optional geometry simplification. Parallelized with batching."""
        logger.info("Building waterway graph...")
        node_frequency = Counter()
        waterway_endpoints = []
        processed_count = 0
        total_waterways = len(waterways)
        
        # Show progress only at 25%, 50%, 75%, 100%
        progress_interval = max(total_waterways // 4, 100000)
        
        # Use map for efficient batching
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(
                process_single_waterway,
                waterways,
                [self.simplify_tolerance]*total_waterways,
                [self.keep_intermediate_distance]*total_waterways,
                [self.coordinate_precision]*total_waterways,
                chunksize=1000
            )
            for rounded_coords in results:
                processed_count += 1
                if processed_count % progress_interval == 0:
                    percentage = (processed_count / total_waterways) * 100
                    print(f"Waterway processing: {percentage:.0f}% ({processed_count:,}/{total_waterways:,})")
                if not rounded_coords or len(rounded_coords) < 2:
                    continue
                waterway_endpoints.append(rounded_coords)
                for coord in rounded_coords:
                    node_frequency[coord] += 1
        # Step 2: Identify junctions (nodes that appear in multiple ways) and endpoints
        junctions = set()
        endpoints = set()
        intermediate_nodes = set()  # New: nodes added at regular intervals
        
        for waterway_coords in waterway_endpoints:
            start_coord = waterway_coords[0]
            end_coord = waterway_coords[-1]
            
            # Endpoints of waterways
            if node_frequency[start_coord] == 1:
                endpoints.add(start_coord)
            else:
                junctions.add(start_coord)
                
            if node_frequency[end_coord] == 1:
                endpoints.add(end_coord)
            else:
                junctions.add(end_coord)
                
            # Internal nodes that connect to other waterways
            for coord in waterway_coords[1:-1]:
                if node_frequency[coord] > 1:
                    junctions.add(coord)
                elif self.keep_intermediate_distance:
                    # If we're keeping intermediate nodes, mark them as important
                    intermediate_nodes.add(coord)
        
        # Step 3: Build simplified edges between important nodes
        important_nodes = junctions | endpoints | intermediate_nodes
        critical_nodes = junctions | endpoints  # Pre-compute this once
        edges = []
        logger.info(f"Building {len(important_nodes):,} graph nodes...")
        
        # Use super-fast edge creation that avoids expensive preprocessing
        edges = create_edges_super_fast(waterway_endpoints, junctions, endpoints, self.min_length_meters)
        
        logger.info(f"Graph: {len(edges):,} edges, {len(junctions):,} junctions, {len(endpoints):,} endpoints")
        
        # Step 4: Create final node list and edge index mapping
        all_nodes = set()
        for edge in edges:
            all_nodes.add(edge['start'])
            all_nodes.add(edge['end'])
        
        # Convert to list and create index mapping
        node_list = list(all_nodes)
        node_to_index = {node: i for i, node in enumerate(node_list)}
        
        # Create final edge list with integer indices and full geometry
        final_edges = []
        for edge in edges:
            start_idx = node_to_index[edge['start']]
            end_idx = node_to_index[edge['end']]
            # Store the full geometry as [lat, lon] pairs
            final_edges.append({
                'start': start_idx,
                'end': end_idx,
                'length': edge['length'],
                'coordinates': edge['coordinates']
            })
        
        # Convert nodes to [lat, lon] format
        final_nodes = [[lat, lon] for lat, lon in node_list]
        
        logger.info(f"Final graph: {len(final_nodes)} nodes, {len(final_edges)} edges")
        
        return final_nodes, final_edges


def extract_waterways(pbf_file, cache_file=None):
    """Extract waterway data from OSM PBF file, with optional caching."""
    import json
    if cache_file and os.path.exists(cache_file):
        logger.info(f"Loading cached waterways from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            waterways = json.load(f)
        logger.info(f"Loaded {len(waterways)} waterways from cache")
        return waterways
    logger.info(f"Extracting waterways from {pbf_file}")
    if not os.path.exists(pbf_file):
        raise FileNotFoundError(f"Input file not found: {pbf_file}")
    handler = WaterwayHandler()
    try:
        handler.apply_file(pbf_file, locations=True)
    except Exception as e:
        logger.error(f"Error processing OSM file: {e}")
        raise
    logger.info(f"Extracted {len(handler.waterways)} waterways")
    if cache_file:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(handler.waterways, f, separators=(',', ':'))
        logger.info(f"Saved extracted waterways to {cache_file}")
    return handler.waterways


def save_compressed_json(data, filepath):
    """Save data as compressed JSON."""
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'))  # Compact JSON
    
    size = os.path.getsize(filepath)
    logger.info(f"Saved {filepath}")
    return size


def get_output_filenames(input_file):
    """Generate output filenames based on input file."""
    base_name = Path(input_file).stem
    if base_name.endswith('.osm'):
        base_name = base_name[:-4]
    
    nodes_file = f"{base_name}.nodes.json.gz"
    edges_file = f"{base_name}.edges.json.gz"
    
    return nodes_file, edges_file


def get_waterway_cache_filename(input_file):
    """Generate a filename for caching extracted waterways."""
    base_name = Path(input_file).stem
    if base_name.endswith('.osm'):
        base_name = base_name[:-4]
    return f"{base_name}.waterways.json"


def main():
    parser = argparse.ArgumentParser(
        description="Extract and minimize waterway networks from OSM PBF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python osm_waterway_extractor_fixed.py brazil-latest.osm.pbf
  python osm_waterway_extractor_fixed.py data.osm.pbf --min-length 50 --precision 4 --simplify-tolerance 0.0001

Output files will be named based on input file:
  input: brazil-latest.osm.pbf
  output: brazil-latest.nodes.json.gz, brazil-latest.edges.json.gz
        """
    )
    
    parser.add_argument('input_file', help='Path to input OSM PBF file')
    parser.add_argument('--min-length', type=float, default=100,
                        help='Minimum edge length in meters (default: 100)')
    parser.add_argument('--precision', type=int, default=5,
                        help='Coordinate precision in decimal places (default: 5)')
    parser.add_argument('--simplify-tolerance', type=float, default=None,
                        help='Simplification tolerance for geometry (in degrees, e.g., 0.0001). Higher values mean more simplification. Default: no simplification')
    parser.add_argument('--reuse-waterways', action='store_true',
                        help='Reuse previously extracted waterways if available (cache file)')
    parser.add_argument('--export-geojson', action='store_true',
                        help='Export the resulting graph as a GeoJSON file for GIS evaluation')
    parser.add_argument('--keep-intermediate-nodes', type=float, metavar='METERS',
                        help='Keep intermediate nodes at regular distance intervals (in meters, e.g., 500)')
    
    args = parser.parse_args()
    try:
        # Step 1 & 2: Extract waterways from PBF or cache
        cache_file = get_waterway_cache_filename(args.input_file) if args.reuse_waterways else None
        waterways = extract_waterways(args.input_file, cache_file=cache_file)
        
        if not waterways:
            logger.warning("No waterways found in the input file")
            return
        
        # Step 3 & 4: Build and minimize graph
        graph_builder = WaterwayGraphBuilder(
            min_length_meters=args.min_length,
            coordinate_precision=args.precision,
            simplify_tolerance=args.simplify_tolerance,
            keep_intermediate_distance=args.keep_intermediate_nodes
        )
        nodes, edges = graph_builder.build_graph(waterways)
        
        # Step 5: Save compressed output
        nodes_file, edges_file = get_output_filenames(args.input_file)
        nodes_size = save_compressed_json(nodes, nodes_file)
        edges_size = save_compressed_json(edges, edges_file)

        # Export GeoJSON if requested
        if args.export_geojson:
            try:
                import geojson
            except ImportError:
                print("Error: geojson library not found. Install with: pip install geojson")
                sys.exit(1)
            features = []
            # Load full edge data to get geometry
            import gzip, json as _json
            with gzip.open(edges_file, 'rt', encoding='utf-8') as f:
                edge_data = _json.load(f)
            for edge in edge_data:
                # edge['coordinates'] is a list of [lat, lon]; GeoJSON expects [lon, lat]
                coords = [(lon, lat) for lat, lon in edge['coordinates']]
                line = geojson.LineString(coords)
                features.append(geojson.Feature(geometry=line))
            fc = geojson.FeatureCollection(features)
            geojson_file = Path(nodes_file).with_suffix('.geojson')
            with open(geojson_file, 'w') as f:
                geojson.dump(fc, f)
            logger.info(f"Exported GeoJSON to {geojson_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("EXTRACTION SUMMARY")
        print("="*50)
        print(f"Input file: {args.input_file}")
        print(f"Original waterways: {len(waterways):,}")
        print(f"Final nodes: {len(nodes):,}")
        print(f"Final edges: {len(edges):,}")
        print(f"Min edge length: {args.min_length}m")
        print(f"Coordinate precision: {args.precision} decimal places")
        if args.simplify_tolerance is not None:
            print(f"Simplification tolerance: {args.simplify_tolerance}")
        if args.keep_intermediate_nodes is not None:
            print(f"Intermediate nodes every: {args.keep_intermediate_nodes}m")
        print(f"\nOutput files:")
        print(f"  {nodes_file} ({nodes_size:,} bytes)")
        print(f"  {edges_file} ({edges_size:,} bytes)")
        print(f"  Total size: {nodes_size + edges_size:,} bytes")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()