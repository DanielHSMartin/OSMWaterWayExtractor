#!/usr/bin/env python3
"""
OSM Waterway Extraction and Cleaning Script v2.1

This script extracts waterway data from OpenStreetMap PBF files and creates
network graphs with comprehensive cleaning, snapping, and optimization features.

Supports accurate geodesic distance calculations, spatial indexing, union-find
clustering, deterministic ID assignment, and multiple output formats.

Usage:
    python osm_waterway_extractor.py <input.osm.pbf> [--config CONFIG_FILE] [options]

Example:
    python osm_waterway_extractor.py brazil-latest.osm.pbf --config config.yaml

Requirements:
    pip install osmium pyproj shapely geopandas numpy pyyaml xxhash rtree fastparquet
"""

import argparse
import json
import gzip
import os
import sys
import math
import time
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any, Set
import logging
import concurrent.futures
import multiprocessing
import functools
import numpy as np
import yaml

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

try:
    import xxhash
except ImportError:
    print("Error: xxhash library not found.")
    print("Install with: pip install xxhash")
    sys.exit(1)

try:
    from rtree import index
except ImportError:
    print("Error: rtree library not found.")
    print("Install with: pip install rtree")
    sys.exit(1)

try:
    import fastparquet
except ImportError:
    print("Error: fastparquet library not found.")
    print("Install with: pip install fastparquet")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class for waterway extraction parameters."""
    # Processing parameters
    snap_tolerance_m: float = 2.0
    min_fragment_length_m: float = 50.0
    coordinate_precision: int = 6
    parallel_workers: int = 8
    distance_calculation_method: str = "geodesic"
    waterway_types: List[str] = field(default_factory=lambda: ["river", "canal"])
    
    # Geometry simplification parameters
    enable_geometry_simplification: bool = True
    simplification_tolerance_m: float = 1.0
    
    # Clustering parameters
    max_displacement_multiplier: float = 1.5
    warning_displacement_multiplier: float = 1.2
    max_cluster_size_warning: int = 10
    enable_union_find: bool = True
    
    # ID assignment parameters
    server_strategy: str = "deterministic"
    mobile_strategy: str = "sequential"
    hash_function: str = "xxhash"
    hash_length: int = 8
    hash_encoding: str = "base62"
    
    # Output parameters
    server_formats: List[str] = field(default_factory=lambda: ["parquet", "csv", "geojson", "jsongz"])
    compression: bool = True
    include_geodesic_distances: bool = True
    generate_mobile_csv: bool = False
    generate_id_mapping: bool = False
    generate_manifest: bool = False
    
    # QA parameters
    enable_comprehensive_metrics: bool = True
    distance_validation_samples: int = 1000
    generate_debug_outputs: bool = False
    qa_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "max_unsnapped_near_miss_pct": 0.1,
        "min_width_parse_success_rate": 0.7,
        "max_crossing_edges_pct": 1.0
    })
    
    # Caching parameters
    enable_parameter_based_caching: bool = True
    cache_directory: str = "./intermediate"
    reuse_extraction: bool = True
    
    def __post_init__(self):
        if self.waterway_types is None:
            self.waterway_types = ["river", "canal"]
        if self.server_formats is None:
            self.server_formats = ["parquet", "csv", "geojson"]
        if self.qa_thresholds is None:
            self.qa_thresholds = {
                "max_unsnapped_near_miss_pct": 0.1,
                "min_width_parse_success_rate": 0.7,
                "max_crossing_edges_pct": 1.0
            }
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract only configuration sections that map to Config fields
        config_dict = {}
        
        # Map YAML sections to Config attributes
        if 'processing' in data:
            processing = data['processing']
            config_dict.update({
                'snap_tolerance_m': processing.get('snap_tolerance_m', 2.0),
                'min_fragment_length_m': processing.get('min_fragment_length_m', 50.0),
                'coordinate_precision': processing.get('coordinate_precision', 6),
                'parallel_workers': processing.get('parallel_workers', 8),
                'distance_calculation_method': processing.get('distance_calculation_method', 'geodesic'),
                'waterway_types': processing.get('waterway_types', ['river', 'canal']),
                'enable_geometry_simplification': processing.get('enable_geometry_simplification', True),
                'simplification_tolerance_m': processing.get('simplification_tolerance_m', 1.0)
            })
        
        if 'clustering' in data:
            clustering = data['clustering']
            config_dict.update({
                'max_displacement_multiplier': clustering.get('max_displacement_multiplier', 1.5),
                'warning_displacement_multiplier': clustering.get('warning_displacement_multiplier', 1.2),
                'max_cluster_size_warning': clustering.get('max_cluster_size_warning', 10),
                'enable_union_find': clustering.get('enable_union_find', True)
            })
        
        if 'ids' in data:
            ids = data['ids']
            config_dict.update({
                'server_strategy': ids.get('server_strategy', 'deterministic'),
                'mobile_strategy': ids.get('mobile_strategy', 'sequential'),
                'hash_function': ids.get('hash_function', 'xxhash'),
                'hash_length': ids.get('hash_length', 8),
                'hash_encoding': ids.get('hash_encoding', 'base62')
            })
        
        if 'output' in data:
            output = data['output']
            config_dict.update({
                'server_formats': output.get('server_formats', ['parquet', 'csv', 'geojson']),
                'compression': output.get('compression', True),
                'include_geodesic_distances': output.get('include_geodesic_distances', True),
                'generate_mobile_csv': output.get('generate_mobile_csv', False),
                'generate_id_mapping': output.get('generate_id_mapping', False),
                'generate_manifest': output.get('generate_manifest', False),
            })
        
        if 'qa' in data:
            qa = data['qa']
            config_dict.update({
                'enable_comprehensive_metrics': qa.get('enable_comprehensive_metrics', True),
                'distance_validation_samples': qa.get('distance_validation_samples', 1000),
                'generate_debug_outputs': qa.get('generate_debug_outputs', False),
                'qa_thresholds': qa.get('qa_thresholds', {
                    'max_unsnapped_near_miss_pct': 0.1,
                    'min_width_parse_success_rate': 0.7,
                    'max_crossing_edges_pct': 1.0
                })
            })
        
        if 'caching' in data:
            caching = data['caching']
            config_dict.update({
                'enable_parameter_based_caching': caching.get('enable_parameter_based_caching', True),
                'cache_directory': caching.get('cache_directory', './intermediate'),
                'reuse_extraction': caching.get('reuse_extraction', True)
            })
        
        return cls(**config_dict)
    
    def get_parameter_hash(self) -> str:
        """Generate a hash of configuration parameters for caching."""
        # Create a deterministic string representation of config
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def get_step_parameter_hash(self, step_name: str) -> str:
        """Generate a hash of configuration parameters relevant to a specific processing step."""
        step_params = {}
        
        if step_name == "extraction":
            # Only parameters that affect waterway extraction
            step_params = {
                'waterway_types': self.waterway_types,
            }
        elif step_name == "processed":
            # Parameters that affect waterway processing
            step_params = {
                'waterway_types': self.waterway_types,
                'coordinate_precision': self.coordinate_precision,
                'min_fragment_length_m': self.min_fragment_length_m,
            }
        elif step_name == "simplified":
            # Parameters that affect geometry simplification (depends on processed)
            step_params = {
                'waterway_types': self.waterway_types,
                'coordinate_precision': self.coordinate_precision,
                'min_fragment_length_m': self.min_fragment_length_m,
                'enable_geometry_simplification': self.enable_geometry_simplification,
                'simplification_tolerance_m': self.simplification_tolerance_m,
            }
        elif step_name == "endpoints":
            # Parameters that affect endpoint extraction (depends on simplified)
            step_params = {
                'waterway_types': self.waterway_types,
                'coordinate_precision': self.coordinate_precision,
                'min_fragment_length_m': self.min_fragment_length_m,
                'enable_geometry_simplification': self.enable_geometry_simplification,
                'simplification_tolerance_m': self.simplification_tolerance_m,
            }
        elif step_name == "clustering":
            # Parameters that affect clustering (depends on endpoints)
            step_params = {
                'waterway_types': self.waterway_types,
                'coordinate_precision': self.coordinate_precision,
                'min_fragment_length_m': self.min_fragment_length_m,
                'enable_geometry_simplification': self.enable_geometry_simplification,
                'simplification_tolerance_m': self.simplification_tolerance_m,
                'snap_tolerance_m': self.snap_tolerance_m,
                'max_displacement_multiplier': self.max_displacement_multiplier,
                'warning_displacement_multiplier': self.warning_displacement_multiplier,
                'enable_union_find': self.enable_union_find,
                'distance_calculation_method': self.distance_calculation_method,
            }
        elif step_name == "edges":
            # Parameters that affect edge creation (depends on clustering)
            step_params = {
                'waterway_types': self.waterway_types,
                'coordinate_precision': self.coordinate_precision,
                'min_fragment_length_m': self.min_fragment_length_m,
                'enable_geometry_simplification': self.enable_geometry_simplification,
                'simplification_tolerance_m': self.simplification_tolerance_m,
                'snap_tolerance_m': self.snap_tolerance_m,
                'max_displacement_multiplier': self.max_displacement_multiplier,
                'warning_displacement_multiplier': self.warning_displacement_multiplier,
                'enable_union_find': self.enable_union_find,
                'distance_calculation_method': self.distance_calculation_method,
                'include_geodesic_distances': self.include_geodesic_distances,
            }
        elif step_name == "nodes":
            # Parameters that affect node creation (depends on edges)
            step_params = {
                'waterway_types': self.waterway_types,
                'coordinate_precision': self.coordinate_precision,
                'min_fragment_length_m': self.min_fragment_length_m,
                'enable_geometry_simplification': self.enable_geometry_simplification,
                'simplification_tolerance_m': self.simplification_tolerance_m,
                'snap_tolerance_m': self.snap_tolerance_m,
                'max_displacement_multiplier': self.max_displacement_multiplier,
                'warning_displacement_multiplier': self.warning_displacement_multiplier,
                'enable_union_find': self.enable_union_find,
                'distance_calculation_method': self.distance_calculation_method,
                'include_geodesic_distances': self.include_geodesic_distances,
                'server_strategy': self.server_strategy,
                'mobile_strategy': self.mobile_strategy,
                'hash_function': self.hash_function,
                'hash_length': self.hash_length,
                'hash_encoding': self.hash_encoding,
            }
        else:
            # Fallback to all parameters
            step_params = asdict(self)
        
        # Create a deterministic string representation of step-specific config
        config_str = json.dumps(step_params, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class UnionFind:
    """Union-Find data structure for clustering endpoints with transitive relationships."""
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.cluster_data = defaultdict(list)  # Store actual coordinates in clusters
    
    def add(self, item: Tuple[float, float]):
        """Add a coordinate to the union-find structure."""
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            self.cluster_data[item] = [item]
    
    def find(self, item: Tuple[float, float]) -> Tuple[float, float]:
        """Find the root representative of the cluster containing item."""
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  # Path compression
        return self.parent[item]
    
    def union(self, item1: Tuple[float, float], item2: Tuple[float, float]):
        """Union two clusters containing item1 and item2."""
        root1 = self.find(item1)
        root2 = self.find(item2)
        
        if root1 == root2:
            return
        
        # Union by rank
        if self.rank[root1] < self.rank[root2]:
            root1, root2 = root2, root1
        
        self.parent[root2] = root1
        
        # Merge cluster data
        self.cluster_data[root1].extend(self.cluster_data[root2])
        del self.cluster_data[root2]
        
        if self.rank[root1] == self.rank[root2]:
            self.rank[root1] += 1
    
    def get_clusters(self) -> Dict[Tuple[float, float], List[Tuple[float, float]]]:
        """Get all clusters as a dictionary mapping root to list of coordinates."""
        clusters = {}
        for item in self.parent:
            root = self.find(item)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(item)
        return clusters


# Global geod calculator for multiprocessing (initialized in worker processes)
_GLOBAL_GEOD = None

def _init_worker_geod(method="geodesic"):
    """Initialize geodesic calculator in worker process"""
    global _GLOBAL_GEOD
    _GLOBAL_GEOD = Geod(ellps='WGS84')

def _calculate_geodesic_distance(coord1, coord2):
    """Calculate geodesic distance between two coordinates (worker function)"""
    global _GLOBAL_GEOD
    if _GLOBAL_GEOD is None:
        _GLOBAL_GEOD = Geod(ellps='WGS84')
    
    # coord format is (lat, lon), but geod expects (lon, lat, lon, lat)
    _, _, distance = _GLOBAL_GEOD.inv(coord1[1], coord1[0], coord2[1], coord2[0])
    return distance

def _calculate_segment_length_worker(coords):
    """Calculate segment length in worker process"""
    if len(coords) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(coords)):
        total_length += _calculate_geodesic_distance(coords[i-1], coords[i])
    
    return total_length

def _calculate_segment_length_vectorized(coords_list):
    """Calculate segment lengths for multiple coordinate lists using vectorized operations"""
    try:
        from pyproj import Geod
        import numpy as np
        
        geod = Geod(ellps='WGS84')
        results = []
        
        for coords in coords_list:
            if len(coords) < 2:
                results.append(0.0)
                continue
                
            # Convert to numpy arrays for vectorized operations
            coords_array = np.array(coords)
            lats = coords_array[:, 0]
            lons = coords_array[:, 1]
            
            if len(coords) >= 2:
                # Get consecutive pairs
                lat1 = lats[:-1]
                lon1 = lons[:-1]
                lat2 = lats[1:]
                lon2 = lons[1:]
                
                # Vectorized geodesic calculation
                try:
                    distances = geod.inv(lon1, lat1, lon2, lat2)[2]
                    total_length = np.sum(distances)
                    results.append(float(total_length))
                except:
                    # Fall back to single calculations
                    total_length = 0.0
                    for i in range(1, len(coords)):
                        total_length += geod.inv(coords[i-1][1], coords[i-1][0], 
                                              coords[i][1], coords[i][0])[2]
                    results.append(total_length)
            else:
                results.append(0.0)
                
        return results
        
    except ImportError:
        # Fall back to single calculations if numpy not available
        from pyproj import Geod
        geod = Geod(ellps='WGS84')
        results = []
        
        for coords in coords_list:
            if len(coords) < 2:
                results.append(0.0)
                continue
                
            total_length = 0.0
            for i in range(1, len(coords)):
                total_length += geod.inv(coords[i-1][1], coords[i-1][0], 
                                      coords[i][1], coords[i][0])[2]
            results.append(total_length)
            
        return results

def _process_waterway_batch_for_edges(args):
    """Process a batch of waterways to create edges with vectorized distance calculations"""
    waterway_batch, coord_mapping, config_dict = args
    
    # Recreate necessary objects in worker process
    from pyproj import Geod
    import xxhash
    
    geod = Geod(ellps='WGS84')
    
    all_edges = []
    
    # Batch process segments for distance calculation
    segment_coords_list = []
    segment_metadata = []
    
    for waterway in waterway_batch:
        coords = waterway['coordinates']
        way_id = waterway['id']
        tags = waterway['tags']
        
        # Apply coordinate mapping from clustering
        mapped_coords = [coord_mapping.get(coord, coord) for coord in coords]
        
        # Split waterway at junction points and collect segments
        waterway_edges = _split_waterway_collect_segments(
            mapped_coords, way_id, tags, coord_mapping, config_dict
        )
        
        for edge_data in waterway_edges:
            segment_coords_list.append(edge_data['coordinates'])
            segment_metadata.append(edge_data)
    
    # Calculate all segment lengths in batch using vectorized operations
    if segment_coords_list:
        segment_lengths = _calculate_segment_length_vectorized(segment_coords_list)
        
        # Assign lengths back to edges
        for i, edge_data in enumerate(segment_metadata):
            edge_data['length_m'] = segment_lengths[i]
            all_edges.append(edge_data)
    
    return all_edges

def _optimize_coord_mapping_for_multiprocessing(coord_mapping, config_dict):
    """Convert coordinate mapping to a more efficient format for multiprocessing"""
    # Get precision from config to ensure consistency
    precision = config_dict.get('coordinate_precision', 6)
    
    # Convert to a format that serializes more efficiently
    optimized_mapping = {}
    
    for coord, mapped_coord in coord_mapping.items():
        # Include ALL mappings using the correct precision
        key = f"{coord[0]:.{precision}f},{coord[1]:.{precision}f}"
        value = f"{mapped_coord[0]:.{precision}f},{mapped_coord[1]:.{precision}f}"
        optimized_mapping[key] = value
    
    return optimized_mapping

def _restore_coord_mapping(optimized_mapping):
    """Restore coordinate mapping from optimized format"""
    coord_mapping = {}
    
    for key, value in optimized_mapping.items():
        lat1, lon1 = map(float, key.split(','))
        lat2, lon2 = map(float, value.split(','))
        coord_mapping[(lat1, lon1)] = (lat2, lon2)
    
    return coord_mapping

def _process_waterway_batch_optimized(args):
    """Process a batch of waterways with optimized coordinate mapping"""
    waterway_batch, optimized_coord_mapping, config_dict = args
    
    # Restore coordinate mapping
    coord_mapping = _restore_coord_mapping(optimized_coord_mapping)
    
    # No need to add identity mappings since they're already included
    # The original code was adding duplicate identity mappings which corrupted the junction detection
    
    # Process using the batch function
    batch_args = (waterway_batch, coord_mapping, config_dict)
    return _process_waterway_batch_for_edges(batch_args)

def _split_waterway_collect_segments(coords, way_id, tags, coord_mapping, config_dict):
    """Split a waterway into segments and collect metadata (without calculating lengths)"""
    if len(coords) < 2:
        return []
    
    # Find all junction points in this waterway
    junction_indices = []
    junction_coords = set(coord for coord, mapped in coord_mapping.items() 
                        if coord != mapped or _is_junction_coord_worker(coord, coord_mapping))
    
    for i, coord in enumerate(coords):
        if coord in junction_coords or i == 0 or i == len(coords) - 1:
            junction_indices.append(i)
    
    # Create edge metadata between junction points
    edges = []
    for i in range(len(junction_indices) - 1):
        start_idx = junction_indices[i]
        end_idx = junction_indices[i + 1]
        
        if end_idx > start_idx:
            segment_coords = coords[start_idx:end_idx + 1]
            
            if len(segment_coords) >= 2:
                # Generate deterministic IDs
                start_coord = segment_coords[0]
                end_coord = segment_coords[-1]
                
                start_node_id = _generate_node_id_worker(start_coord[0], start_coord[1], config_dict)
                end_node_id = _generate_node_id_worker(end_coord[0], end_coord[1], config_dict)
                edge_id = _generate_edge_id_worker(start_node_id, end_node_id, way_id, i, config_dict)
                
                # Extract and normalize width information
                width_info = _parse_width_tags_worker(tags)
                
                edges.append({
                    'id': edge_id,
                    'from_node_id': start_node_id,
                    'to_node_id': end_node_id,
                    'length_m': 0.0,  # Will be calculated in batch
                    'coordinates': segment_coords,
                    'name': tags.get('name', ''),
                    'type': tags.get('waterway', ''),
                    'width_raw': width_info['raw'],
                    'width_m': width_info['meters'],
                    'width_source': width_info['source'],
                    'original_way_id': way_id
                })
    
    return edges


def _process_waterway_for_edges(args):
    """Process a single waterway to create edges (multiprocessing worker function)"""
    waterway, coord_mapping, config_dict = args
    
    # Recreate necessary objects in worker process
    from pyproj import Geod
    geod = Geod(ellps='WGS84')
    
    coords = waterway['coordinates']
    way_id = waterway['id']
    tags = waterway['tags']
    
    # Apply coordinate mapping from clustering
    mapped_coords = [coord_mapping.get(coord, coord) for coord in coords]
    
    # Split waterway at junction points
    edges = _split_waterway_at_junctions_worker(
        mapped_coords, way_id, tags, coord_mapping, config_dict, geod
    )
    
    return edges

def _split_waterway_at_junctions_worker(coords, way_id, tags, coord_mapping, config_dict, geod):
    """Split a waterway into edges at junction points (worker function)"""
    if len(coords) < 2:
        return []
    
    # Find all junction points in this waterway
    junction_indices = []
    junction_coords = set(coord for coord, mapped in coord_mapping.items() 
                        if coord != mapped or _is_junction_coord_worker(coord, coord_mapping))
    
    for i, coord in enumerate(coords):
        if coord in junction_coords or i == 0 or i == len(coords) - 1:
            junction_indices.append(i)
    
    # Create edges between junction points
    edges = []
    for i in range(len(junction_indices) - 1):
        start_idx = junction_indices[i]
        end_idx = junction_indices[i + 1]
        
        if end_idx > start_idx:
            segment_coords = coords[start_idx:end_idx + 1]
            
            if len(segment_coords) >= 2:
                # Calculate accurate geodesic length
                length_m = _calculate_segment_length_geodesic_worker(segment_coords, geod)
                
                # Generate deterministic IDs
                start_coord = segment_coords[0]
                end_coord = segment_coords[-1]
                
                start_node_id = _generate_node_id_worker(start_coord[0], start_coord[1], config_dict)
                end_node_id = _generate_node_id_worker(end_coord[0], end_coord[1], config_dict)
                edge_id = _generate_edge_id_worker(start_node_id, end_node_id, way_id, i, config_dict)
                
                # Extract and normalize width information
                width_info = _parse_width_tags_worker(tags)
                
                edges.append({
                    'id': edge_id,
                    'from_node_id': start_node_id,
                    'to_node_id': end_node_id,
                    'length_m': length_m,
                    'coordinates': segment_coords,
                    'name': tags.get('name', ''),
                    'type': tags.get('waterway', ''),
                    'width_raw': width_info['raw'],
                    'width_m': width_info['meters'],
                    'width_source': width_info['source'],
                    'original_way_id': way_id
                })
    
    return edges

def _calculate_segment_length_geodesic_worker(coords, geod):
    """Calculate segment length using geodesic calculations in worker"""
    if len(coords) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i-1]
        lat2, lon2 = coords[i]
        # Geod.inv expects (lon1, lat1, lon2, lat2) and returns (az12, az21, dist)
        total_length += geod.inv(lon1, lat1, lon2, lat2)[2]
    
    return total_length

def _is_junction_coord_worker(coord, coord_mapping):
    """Check if a coordinate is a junction point (worker function)"""
    return coord in coord_mapping

def _generate_node_id_worker(lat, lon, config_dict):
    """Generate deterministic node ID for a coordinate (worker function)"""
    import xxhash
    
    # Round to specified precision for consistent hashing
    coordinate_precision = config_dict.get('coordinate_precision', 6)
    hash_length = config_dict.get('hash_length', 8)
    
    rounded_lat = round(lat, coordinate_precision)
    rounded_lon = round(lon, coordinate_precision)
    
    hasher = xxhash.xxh64()
    hasher.update(f"{rounded_lat},{rounded_lon}".encode())
    hash_int = hasher.intdigest()
    
    # Convert to base62 for compact representation
    result = _int_to_base62_worker(hash_int)[:hash_length]
    return f"n{result}"

def _generate_edge_id_worker(from_node_id, to_node_id, original_way_id, segment_index, config_dict):
    """Generate deterministic edge ID (worker function)"""
    import xxhash
    
    hash_length = config_dict.get('hash_length', 8)
    
    hasher = xxhash.xxh64()
    hasher.update(f"{from_node_id}-{to_node_id}-{original_way_id}-{segment_index}".encode())
    hash_int = hasher.intdigest()
    
    result = _int_to_base62_worker(hash_int)[:hash_length]
    return f"e{result}"

def _int_to_base62_worker(num):
    """Convert integer to base62 string (worker function)"""
    if num == 0:
        return '0'
    
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    result = ""
    num = abs(num)  # Ensure positive
    
    while num > 0:
        result = chars[num % 62] + result
        num //= 62
    
    return result

def _parse_width_tags_worker(tags):
    """Parse width information from OSM tags (worker function)"""
    width_raw = tags.get('width', '')
    width_m = None
    width_source = 'none'
    
    if width_raw:
        width_source = 'tag'
        try:
            # Handle common width formats
            width_str = width_raw.lower().strip()
            
            if 'm' in width_str:
                # "5 m", "5m", "5.5 m"
                width_m = float(width_str.replace('m', '').strip())
            elif 'ft' in width_str or 'feet' in width_str:
                # "15 ft", "15 feet"
                feet = float(width_str.replace('ft', '').replace('feet', '').strip())
                width_m = feet * 0.3048  # Convert feet to meters
            elif width_str.replace('.', '').isdigit():
                # Assume meters if just a number
                width_m = float(width_str)
            
            # Validate reasonable width values
            if width_m is not None and (width_m <= 0 or width_m > 1000):
                width_m = None  # Invalid width
                
        except (ValueError, AttributeError):
            pass  # Keep width_m as None for unparseable values
    
    return {
        'raw': width_raw,
        'meters': width_m,
        'source': width_source
    }


def _process_waterway_coordinates_chunk(args):
    """Process waterway coordinates chunk for multiprocessing"""
    waterway_chunk, coordinate_precision = args
    
    processed = []
    
    for waterway in waterway_chunk:
        coords = waterway['coordinates']
        if len(coords) < 2:
            continue
            
        # Round coordinates to specified precision
        rounded_coords = [
            (round(lat, coordinate_precision), 
             round(lon, coordinate_precision))
            for lat, lon in coords
        ]
        
        # Remove consecutive duplicate coordinates
        deduplicated_coords = [rounded_coords[0]]
        for coord in rounded_coords[1:]:
            if coord != deduplicated_coords[-1]:
                deduplicated_coords.append(coord)
        
        if len(deduplicated_coords) >= 2:
            processed.append({
                'id': waterway['id'],
                'coordinates': deduplicated_coords,
                'tags': waterway['tags']
            })
    
    return processed

def _simplify_geometry_chunk(args):
    """Simplify geometry chunk for multiprocessing"""
    waterway_chunk, simplification_tolerance_m = args
    
    simplified = []
    tolerance_degrees = simplification_tolerance_m * 0.00001
    
    # Import shapely in worker process
    try:
        from shapely.geometry import LineString
    except ImportError:
        # Fall back to returning original if shapely not available
        return waterway_chunk
    
    for waterway in waterway_chunk:
        coords = waterway['coordinates']
        if len(coords) < 2:
            continue
            
        try:
            # Convert to LineString
            line = LineString([(lon, lat) for lat, lon in coords])
            
            # Simplify the geometry
            simplified_line = line.simplify(tolerance_degrees, preserve_topology=True)
            
            # Convert back to coordinates
            if simplified_line.geom_type == 'LineString':
                simplified_coords = [(lat, lon) for lon, lat in simplified_line.coords]
                
                # Ensure we still have at least 2 points
                if len(simplified_coords) >= 2:
                    simplified.append({
                        'id': waterway['id'],
                        'coordinates': simplified_coords,
                        'tags': waterway['tags']
                    })
                    
        except Exception as e:
            # Fall back to original if simplification fails
            simplified.append(waterway)
    
    return simplified

def _extract_endpoints_chunk(args):
    """Extract endpoints from waterway chunk for multiprocessing"""
    waterway_chunk, = args
    
    endpoint_count = Counter()
    all_coordinates_count = Counter()
    all_endpoints = []
    
    # Count how many times each coordinate appears as an endpoint
    for waterway in waterway_chunk:
        coords = waterway['coordinates']
        start_coord = coords[0]
        end_coord = coords[-1]
        
        endpoint_count[start_coord] += 1
        endpoint_count[end_coord] += 1
        all_endpoints.extend([start_coord, end_coord])
        
        # Also count ALL coordinates to detect interior intersections
        for coord in coords:
            all_coordinates_count[coord] += 1
    
    return endpoint_count, all_coordinates_count, all_endpoints


class GeodCalculator:
    """Handles accurate geodesic distance calculations."""
    
    def __init__(self, method: str = "geodesic"):
        self.method = method
        self.geod = Geod(ellps='WGS84')  # WGS84 ellipsoid for accuracy
    
    def distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate geodesic distance between two coordinates in meters.
        
        Args:
            coord1: (lat, lon) tuple
            coord2: (lat, lon) tuple
        
        Returns:
            Distance in meters
        """
        if self.method == "geodesic":
            # coord format is (lat, lon), but geod expects (lon, lat, lon, lat)
            _, _, distance = self.geod.inv(coord1[1], coord1[0], coord2[1], coord2[0])
            return distance
        else:
            # Fallback to approximate calculation for now
            return self._approximate_distance(coord1, coord2)
    
    def _approximate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Approximate distance calculation (fallback method)."""
        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        lat_avg = (lat1 + lat2) / 2
        
        # Convert to meters (approximate)
        dx = dlat * 111000  # 1 degree latitude ≈ 111km
        dy = dlon * 111000 * math.cos(lat_avg)  # Adjust longitude for latitude
        
        return math.sqrt(dx**2 + dy**2)
    
    def calculate_segment_length(self, coords: List[Tuple[float, float]]) -> float:
        """Calculate total length of a coordinate sequence using geodesic distances."""
        if len(coords) < 2:
            return 0.0
        
        # For long segments, use vectorized calculation for better performance
        if len(coords) > 20:
            return self._calculate_segment_length_batch(coords)
        
        total_length = 0.0
        for i in range(1, len(coords)):
            total_length += self.distance(coords[i-1], coords[i])
        
        return total_length
    
    def _calculate_segment_length_batch(self, coords: List[Tuple[float, float]]) -> float:
        """Calculate segment length using batch processing for long segments."""
        if len(coords) < 2:
            return 0.0
        
        total_length = 0.0
        
        # Process coordinates in batches for better cache locality
        if self.method == "geodesic":
            # For geodesic, use the same method but with better memory access patterns
            for i in range(1, len(coords)):
                lat1, lon1 = coords[i-1]
                lat2, lon2 = coords[i]
                # Geod.inv expects (lon1, lat1, lon2, lat2) and returns (az12, az21, dist)
                total_length += self.geod.inv(lon1, lat1, lon2, lat2)[2]
        else:
            # For other methods, use the regular distance function
            for i in range(1, len(coords)):
                total_length += self.distance(coords[i-1], coords[i])
        
        return total_length
    
    def calculate_multiple_segment_lengths(self, segments: List[List[Tuple[float, float]]]) -> List[float]:
        """Calculate lengths for multiple segments efficiently."""
        return [self.calculate_segment_length(segment) for segment in segments]


class IDGenerator:
    """Generates deterministic and sequential IDs for nodes and edges."""
    
    def __init__(self, config: Config):
        self.config = config
        self.node_id_map = {}  # coord -> deterministic_id
        self.edge_id_map = {}  # (from_id, to_id, way_id, segment) -> deterministic_id
        self.mobile_node_counter = 0
        self.mobile_edge_counter = 0
        self.mobile_id_mapping = {'nodes': {}, 'edges': {}}
    
    def _hash_coordinate(self, lat: float, lon: float) -> str:
        """Generate deterministic hash for a coordinate."""
        # Round to specified precision for consistent hashing
        rounded_lat = round(lat, self.config.coordinate_precision)
        rounded_lon = round(lon, self.config.coordinate_precision)
        
        if self.config.hash_function == "xxhash":
            hasher = xxhash.xxh64()
            hasher.update(f"{rounded_lat},{rounded_lon}".encode())
            hash_int = hasher.intdigest()
        else:
            # Fallback to built-in hash
            hash_int = hash(f"{rounded_lat},{rounded_lon}")
        
        # Convert to base62 for compact representation
        return self._int_to_base62(hash_int)[:self.config.hash_length]
    
    def _int_to_base62(self, num: int) -> str:
        """Convert integer to base62 string."""
        if num == 0:
            return '0'
        
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        result = ""
        num = abs(num)  # Ensure positive
        
        while num > 0:
            result = chars[num % 62] + result
            num //= 62
        
        return result
    
    def generate_node_id(self, lat: float, lon: float) -> str:
        """Generate deterministic node ID for a coordinate."""
        coord_key = (round(lat, self.config.coordinate_precision), 
                    round(lon, self.config.coordinate_precision))
        
        if coord_key not in self.node_id_map:
            det_id = f"n{self._hash_coordinate(lat, lon)}"
            self.node_id_map[coord_key] = det_id
        
        return self.node_id_map[coord_key]
    
    def generate_edge_id(self, from_node_id: str, to_node_id: str, 
                        original_way_id: int, segment_index: int) -> str:
        """Generate deterministic edge ID."""
        edge_key = (from_node_id, to_node_id, original_way_id, segment_index)
        
        if edge_key not in self.edge_id_map:
            # Create hash from all components
            if self.config.hash_function == "xxhash":
                hasher = xxhash.xxh64()
                hasher.update(f"{from_node_id}-{to_node_id}-{original_way_id}-{segment_index}".encode())
                hash_int = hasher.intdigest()
            else:
                hash_int = hash(f"{from_node_id}-{to_node_id}-{original_way_id}-{segment_index}")
            
            det_id = f"e{self._int_to_base62(hash_int)[:self.config.hash_length]}"
            self.edge_id_map[edge_key] = det_id
        
        return self.edge_id_map[edge_key]
    
    def get_mobile_node_id(self, det_id: str) -> int:
        """Get sequential mobile ID for a deterministic node ID."""
        if det_id not in self.mobile_id_mapping['nodes']:
            self.mobile_node_counter += 1
            self.mobile_id_mapping['nodes'][det_id] = self.mobile_node_counter
        return self.mobile_id_mapping['nodes'][det_id]
    
    def get_mobile_edge_id(self, det_id: str) -> int:
        """Get sequential mobile ID for a deterministic edge ID.""" 
        if det_id not in self.mobile_id_mapping['edges']:
            self.mobile_edge_counter += 1
            self.mobile_id_mapping['edges'][det_id] = self.mobile_edge_counter
        return self.mobile_id_mapping['edges'][det_id]


class SpatialIndex:
    """R-tree spatial index for efficient proximity search of endpoints."""
    
    def __init__(self):
        self.idx = index.Index()
        self.id_to_coord = {}  # Maps index ID to coordinate
        self.coord_to_id = {}  # Maps coordinate to index ID  
        self.next_id = 0
    
    def add_point(self, coord: Tuple[float, float]):
        """Add a coordinate to the spatial index."""
        if coord not in self.coord_to_id:
            point_id = self.next_id
            self.next_id += 1
            
            lat, lon = coord
            # R-tree expects (minx, miny, maxx, maxy) - we use point coordinates
            self.idx.insert(point_id, (lon, lat, lon, lat))
            self.id_to_coord[point_id] = coord
            self.coord_to_id[coord] = point_id
    
    def find_within_distance(self, coord: Tuple[float, float], distance_m: float, 
                            geod_calc: GeodCalculator) -> List[Tuple[float, float]]:
        """Find all points within geodesic distance of the given coordinate."""
        lat, lon = coord
        
        # Convert distance to approximate degree bounds for initial R-tree query
        # This is just for bounding box pre-filtering; actual distance is calculated geodesically
        lat_deg_per_m = 1.0 / 111000  # Approximate: 1 degree ≈ 111km
        lon_deg_per_m = lat_deg_per_m / max(math.cos(math.radians(lat)), 0.01)  # Adjust for latitude
        
        degree_buffer = distance_m * max(lat_deg_per_m, lon_deg_per_m)
        
        # Query R-tree with bounding box
        bbox = (lon - degree_buffer, lat - degree_buffer, 
                lon + degree_buffer, lat + degree_buffer)
        
        candidate_ids = list(self.idx.intersection(bbox))
        
        # For large numbers of candidates, use batch distance calculation for better performance
        if len(candidate_ids) > 100:
            return self._filter_candidates_batch(coord, candidate_ids, distance_m, geod_calc)
        else:
            # Filter candidates using accurate geodesic distance
            nearby_coords = []
            for candidate_id in candidate_ids:
                candidate_coord = self.id_to_coord[candidate_id]
                if geod_calc.distance(coord, candidate_coord) <= distance_m:
                    nearby_coords.append(candidate_coord)
            
            return nearby_coords
    
    def _filter_candidates_batch(self, coord: Tuple[float, float], candidate_ids: List[int], 
                                distance_m: float, geod_calc: GeodCalculator) -> List[Tuple[float, float]]:
        """Filter candidate coordinates using batch processing for better performance."""
        nearby_coords = []
        
        # Process candidates in smaller batches for better cache locality
        batch_size = 50
        for i in range(0, len(candidate_ids), batch_size):
            batch_ids = candidate_ids[i:i + batch_size]
            
            for candidate_id in batch_ids:
                candidate_coord = self.id_to_coord[candidate_id]
                if geod_calc.distance(coord, candidate_coord) <= distance_m:
                    nearby_coords.append(candidate_coord)
        
        return nearby_coords


class SnappingClusterer:
    """Handles advanced snapping and node merging with union-find clustering."""
    
    def __init__(self, config: Config, geod_calc: GeodCalculator):
        self.config = config
        self.geod_calc = geod_calc
        self.cluster_metrics = {
            'total_clusters': 0,
            'singleton_clusters': 0,
            'displacement_p50_m': 0.0,
            'displacement_p95_m': 0.0,
            'displacement_p99_m': 0.0,
            'largest_cluster_size': 0,
            'clusters_above_threshold': 0
        }
    
    def cluster_endpoints(self, endpoints: List[Tuple[float, float]]) -> Dict[Tuple[float, float], Tuple[float, float]]:
        """Cluster endpoints using union-find algorithm and return mapping from original to centroid."""
        logger.info(f"Clustering {len(endpoints)} endpoints with {self.config.snap_tolerance_m}m tolerance...")
        
        # Build spatial index
        spatial_idx = SpatialIndex()
        for endpoint in endpoints:
            spatial_idx.add_point(endpoint)
        
        # Initialize union-find structure
        uf = UnionFind()
        for endpoint in endpoints:
            uf.add(endpoint)
        
        # Find all pairs within snapping tolerance and union them
        # Use batch processing for better performance with large datasets
        edges_added = 0
        if len(endpoints) > 10000:
            # For very large datasets, process in chunks to improve memory locality
            chunk_size = 1000
            for i in range(0, len(endpoints), chunk_size):
                chunk_endpoints = endpoints[i:i + chunk_size]
                for endpoint in chunk_endpoints:
                    nearby = spatial_idx.find_within_distance(endpoint, self.config.snap_tolerance_m, self.geod_calc)
                    for nearby_point in nearby:
                        if nearby_point != endpoint:
                            uf.union(endpoint, nearby_point)
                            edges_added += 1
        else:
            # Regular processing for smaller datasets
            for endpoint in endpoints:
                nearby = spatial_idx.find_within_distance(endpoint, self.config.snap_tolerance_m, self.geod_calc)
                for nearby_point in nearby:
                    if nearby_point != endpoint:
                        uf.union(endpoint, nearby_point)
                        edges_added += 1
        
        logger.info(f"Added {edges_added} union edges for clustering")
        
        # Get final clusters and calculate centroids
        clusters = uf.get_clusters()
        coord_mapping = {}
        displacements = []
        
        for root, cluster_coords in clusters.items():
            if len(cluster_coords) == 1:
                # Singleton cluster - no change needed
                coord_mapping[cluster_coords[0]] = cluster_coords[0]
            else:
                # Calculate geodesic-aware centroid
                centroid = self._calculate_cluster_centroid(cluster_coords)
                
                # Validate displacement constraints
                max_displacement = 0.0
                for coord in cluster_coords:
                    displacement = self.geod_calc.distance(coord, centroid)
                    max_displacement = max(max_displacement, displacement)
                    displacements.append(displacement)
                    coord_mapping[coord] = centroid
                
                # Quality control checks
                max_allowed = self.config.snap_tolerance_m * self.config.max_displacement_multiplier
                if max_displacement > max_allowed:
                    logger.error(f"Cluster displacement {max_displacement:.2f}m exceeds limit {max_allowed:.2f}m")
                    raise ValueError("Cluster displacement exceeds safety threshold")
                
                warning_threshold = self.config.snap_tolerance_m * self.config.warning_displacement_multiplier
                if max_displacement > warning_threshold:
                    logger.warning(f"Cluster displacement {max_displacement:.2f}m exceeds warning threshold {warning_threshold:.2f}m")
                
                if len(cluster_coords) > self.config.max_cluster_size_warning:
                    logger.warning(f"Large cluster with {len(cluster_coords)} endpoints detected")
        
        # Update metrics
        self._update_cluster_metrics(clusters, displacements)
        
        logger.info(f"Clustered into {len(clusters)} groups with {self.cluster_metrics['singleton_clusters']} singletons")
        
        return coord_mapping
    
    def _calculate_cluster_centroid(self, coords: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate geodesic-aware centroid of a cluster."""
        if len(coords) == 1:
            return coords[0]
        
        # Simple arithmetic mean for small clusters (reasonable for short distances)
        # For larger distances, this should use proper geodesic averaging
        lat_sum = sum(coord[0] for coord in coords)
        lon_sum = sum(coord[1] for coord in coords)
        
        return (lat_sum / len(coords), lon_sum / len(coords))
    
    def _update_cluster_metrics(self, clusters: Dict, displacements: List[float]):
        """Update clustering metrics for QA reporting."""
        self.cluster_metrics['total_clusters'] = len(clusters)
        self.cluster_metrics['singleton_clusters'] = sum(1 for coords in clusters.values() if len(coords) == 1)
        
        if displacements:
            displacements.sort()
            n = len(displacements)
            self.cluster_metrics['displacement_p50_m'] = displacements[int(n * 0.5)]
            self.cluster_metrics['displacement_p95_m'] = displacements[int(n * 0.95)]
            self.cluster_metrics['displacement_p99_m'] = displacements[int(n * 0.99)]
        
        self.cluster_metrics['largest_cluster_size'] = max(len(coords) for coords in clusters.values())
        warning_threshold = self.config.snap_tolerance_m * self.config.warning_displacement_multiplier
        self.cluster_metrics['clusters_above_threshold'] = sum(
            1 for coords in clusters.values() 
            if len(coords) > 1 and any(
                self.geod_calc.distance(coord, self._calculate_cluster_centroid(list(coords))) > warning_threshold
                for coord in coords
            )
        )


class WaterwayHandler(osmium.SimpleHandler):
    """OSM handler to extract waterway data based on configuration."""
    
    def __init__(self, config: Config):
        osmium.SimpleHandler.__init__(self)
        self.config = config
        self.waterways = []
        self.waterway_type_counts = Counter()  # Track what waterway types we see
        self.matching_waterways = 0  # Track how many match our filter
        self.processed_waterways = 0  # Track how many we successfully process
        # Use osmium's WKB factory to get geometries - need location handler for coordinates
        self.wkb_factory = osmium.geom.WKBFactory()
        
    def way(self, w):
        """Extract ways that match configured waterway types."""
        tags = {tag.k: tag.v for tag in w.tags}
        
        # Track all waterway types found (for debugging)
        if 'waterway' in tags:
            self.waterway_type_counts[tags['waterway']] += 1
        
        # Only process waterways that match configuration
        if tags.get('waterway') in self.config.waterway_types:
            self.matching_waterways += 1
            
            try:
                # Get the geometry using osmium's WKB factory
                wkb = self.wkb_factory.create_linestring(w)
                if wkb:
                    # Convert WKB to shapely geometry
                    from shapely import wkb as shapely_wkb
                    line = shapely_wkb.loads(wkb)
                    
                    # Extract coordinates as (lat, lon) tuples for consistency
                    coords = [(lat, lon) for lon, lat in line.coords]
                    
                    if len(coords) >= 2:  # Valid waterway needs at least 2 points
                        self.waterways.append({
                            'id': w.id,
                            'coordinates': coords,
                            'tags': tags
                        })
                        self.processed_waterways += 1
                    else:
                        logger.debug(f"Skipping way {w.id}: insufficient coordinates ({len(coords)})")
                else:
                    logger.debug(f"Skipping way {w.id}: no geometry created by WKB factory")
                        
            except Exception as e:
                # Skip ways that can't be processed (e.g., incomplete geometry)
                logger.debug(f"Error processing way {w.id}: {e}")




class ModernWaterwayGraphBuilder:
    """Modern waterway graph builder implementing Specification v2.1 features."""
    
    def __init__(self, config: Config):
        self.config = config
        self.geod_calc = GeodCalculator(config.distance_calculation_method)
        self.id_generator = IDGenerator(config)
        self.clusterer = SnappingClusterer(config, self.geod_calc)
        self.qa_metrics = {}
        
    def build_graph(self, waterways: List[Dict], input_file: str = "unknown") -> Tuple[List[Dict], List[Dict]]:
        """Build a waterway graph from extracted waterway data."""
        logger.info(f"Building graph from {len(waterways)} waterways using v2.1 specification...")
        
        start_time = time.time()
        
        # Step 1: Process and clean waterway coordinates
        logger.info("Step 1: Processing and cleaning waterway coordinates...")
        processed_cache_file = get_intermediate_cache_filename(input_file, self.config, "processed") if self.config.enable_parameter_based_caching else None
        
        if processed_cache_file and os.path.exists(processed_cache_file):
            processed_waterways = load_intermediate_cache(processed_cache_file)
        else:
            processed_waterways = self._process_waterways(waterways)
            if processed_cache_file:
                save_intermediate_cache(processed_waterways, processed_cache_file)
        
        # Step 1.5: Simplify geometries (new step)
        logger.info("Step 1.5: Simplifying waterway geometries...")
        simplified_cache_file = get_intermediate_cache_filename(input_file, self.config, "simplified") if self.config.enable_parameter_based_caching else None
        
        if simplified_cache_file and os.path.exists(simplified_cache_file):
            simplified_waterways = load_intermediate_cache(simplified_cache_file)
        else:
            simplified_waterways = self._simplify_geometries(processed_waterways)
            if simplified_cache_file:
                save_intermediate_cache(simplified_waterways, simplified_cache_file)
        
        # Step 1.75: Detect line intersections and split waterways (new step per spec 3.5)
        logger.info("Step 1.75: Detecting line intersections and splitting waterways...")
        intersections_cache_file = get_intermediate_cache_filename(input_file, self.config, "intersections") if self.config.enable_parameter_based_caching else None
        
        if intersections_cache_file and os.path.exists(intersections_cache_file):
            waterways_with_intersections = load_intermediate_cache(intersections_cache_file)
        else:
            waterways_with_intersections = self._detect_and_split_intersections(simplified_waterways)
            if intersections_cache_file:
                save_intermediate_cache(waterways_with_intersections, intersections_cache_file)
        
        # Step 2: Extract all unique endpoints and junctions
        logger.info("Step 2: Extracting endpoints and identifying junctions...")
        endpoints_cache_file = get_intermediate_cache_filename(input_file, self.config, "endpoints") if self.config.enable_parameter_based_caching else None
        
        if endpoints_cache_file and os.path.exists(endpoints_cache_file):
            cached_data = load_intermediate_cache(endpoints_cache_file)
            endpoints, junctions = cached_data['endpoints'], cached_data['junctions']
        else:
            endpoints, junctions = self._extract_endpoints_and_junctions(waterways_with_intersections)
            if endpoints_cache_file:
                save_intermediate_cache({'endpoints': endpoints, 'junctions': junctions}, endpoints_cache_file)
        
        # Step 3: Snap and cluster endpoints using union-find
        logger.info("Step 3: Snapping and clustering endpoints...")
        clustering_cache_file = get_intermediate_cache_filename(input_file, self.config, "clustering") if self.config.enable_parameter_based_caching else None
        
        if clustering_cache_file and os.path.exists(clustering_cache_file):
            coord_mapping = load_intermediate_cache(clustering_cache_file)
        else:
            coord_mapping = self.clusterer.cluster_endpoints(endpoints + junctions)
            if clustering_cache_file:
                save_intermediate_cache(coord_mapping, clustering_cache_file)
        
        # Step 4: Create edges with accurate distance calculations
        logger.info("Step 4: Creating edges with geodesic distances...")
        edges_cache_file = get_intermediate_cache_filename(input_file, self.config, "edges") if self.config.enable_parameter_based_caching else None
        
        if edges_cache_file and os.path.exists(edges_cache_file):
            edges = load_intermediate_cache(edges_cache_file)
        else:
            edges = self._create_edges(waterways_with_intersections, coord_mapping)
            if edges_cache_file:
                save_intermediate_cache(edges, edges_cache_file)
        
        # Step 5: Build final node list with deterministic IDs
        logger.info("Step 5: Building node list with deterministic IDs...")
        nodes_cache_file = get_intermediate_cache_filename(input_file, self.config, "nodes") if self.config.enable_parameter_based_caching else None
        
        if nodes_cache_file and os.path.exists(nodes_cache_file):
            nodes = load_intermediate_cache(nodes_cache_file)
        else:
            nodes = self._build_nodes(coord_mapping)
            if nodes_cache_file:
                save_intermediate_cache(nodes, nodes_cache_file)
        
        # Step 6: Generate QA metrics
        logger.info("Step 6: Generating QA metrics...")
        self._generate_qa_metrics(waterways, simplified_waterways, nodes, edges, time.time() - start_time)
        
        logger.info(f"Graph built: {len(nodes)} nodes, {len(edges)} edges in {time.time() - start_time:.2f}s")
        
        return nodes, edges
    
    def _process_waterways(self, waterways: List[Dict]) -> List[Dict]:
        """Process waterway coordinates and apply coordinate precision."""
        # Use parallel processing for better performance with large datasets
        # Based on performance analysis: break-even point is around 500 waterways
        if self.config.parallel_workers > 1 and len(waterways) > 500:
            logger.info(f"Processing {len(waterways)} waterways using {self.config.parallel_workers} parallel workers")
            # Try multiprocessing first for better CPU utilization
            try:
                return self._process_waterways_multiprocessing(waterways)
            except Exception as e:
                logger.warning(f"Multiprocessing failed ({e}), falling back to threading")
                return self._process_waterways_parallel(waterways)
        else:
            logger.info(f"Processing {len(waterways)} waterways sequentially")
            return self._process_waterways_sequential(waterways)
    
    def _process_waterways_multiprocessing(self, waterways: List[Dict]) -> List[Dict]:
        """Process waterways using multiprocessing for improved performance."""
        # Split waterways into chunks for parallel processing
        chunk_size = max(1, len(waterways) // self.config.parallel_workers)
        waterway_chunks = [waterways[i:i + chunk_size] for i in range(0, len(waterways), chunk_size)]
        
        # Prepare work items for multiprocessing
        work_items = [(chunk, self.config.coordinate_precision) for chunk in waterway_chunks]
        
        all_processed = []
        
        with multiprocessing.Pool(processes=self.config.parallel_workers) as pool:
            try:
                chunk_results = pool.map(_process_waterway_coordinates_chunk, work_items)
                
                # Flatten results
                for chunk_processed in chunk_results:
                    all_processed.extend(chunk_processed)
                    
            except Exception as e:
                logger.error(f"Error in multiprocessing coordinate processing: {e}")
                pool.terminate()
                pool.join()
                raise
        
        logger.info(f"Processed {len(all_processed)}/{len(waterways)} waterways after coordinate cleaning")
        return all_processed
    
    def _process_waterways_sequential(self, waterways: List[Dict]) -> List[Dict]:
        """Process waterways sequentially (original implementation)."""
        processed = []
        
        for waterway in waterways:
            coords = waterway['coordinates']
            if len(coords) < 2:
                continue
                
            # Round coordinates to specified precision
            rounded_coords = [
                (round(lat, self.config.coordinate_precision), 
                 round(lon, self.config.coordinate_precision))
                for lat, lon in coords
            ]
            
            # Remove consecutive duplicate coordinates
            deduplicated_coords = [rounded_coords[0]]
            for coord in rounded_coords[1:]:
                if coord != deduplicated_coords[-1]:
                    deduplicated_coords.append(coord)
            
            if len(deduplicated_coords) >= 2:
                processed.append({
                    'id': waterway['id'],
                    'coordinates': deduplicated_coords,
                    'tags': waterway['tags']
                })
                
        logger.info(f"Processed {len(processed)}/{len(waterways)} waterways after coordinate cleaning")
        return processed
    
    def _process_waterways_parallel(self, waterways: List[Dict]) -> List[Dict]:
        """Process waterways using parallel processing for improved performance."""
        # Split waterways into chunks for parallel processing
        chunk_size = max(1, len(waterways) // self.config.parallel_workers)
        waterway_chunks = [waterways[i:i + chunk_size] for i in range(0, len(waterways), chunk_size)]
        
        all_processed = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit tasks for each chunk
            future_to_chunk = {
                executor.submit(self._process_waterway_chunk_coordinates, chunk): chunk 
                for chunk in waterway_chunks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_processed = future.result()
                    all_processed.extend(chunk_processed)
                except Exception as e:
                    logger.error(f"Error processing coordinate chunk of size {len(chunk)}: {e}")
                    # Fall back to sequential processing for this chunk
                    all_processed.extend(self._process_waterway_chunk_coordinates(chunk))
        
        logger.info(f"Processed {len(all_processed)}/{len(waterways)} waterways after coordinate cleaning")
        return all_processed
    
    def _process_waterway_chunk_coordinates(self, waterway_chunk: List[Dict]) -> List[Dict]:
        """Process a chunk of waterways for coordinate precision and deduplication."""
        processed = []
        
        for waterway in waterway_chunk:
            coords = waterway['coordinates']
            if len(coords) < 2:
                continue
                
            # Round coordinates to specified precision
            rounded_coords = [
                (round(lat, self.config.coordinate_precision), 
                 round(lon, self.config.coordinate_precision))
                for lat, lon in coords
            ]
            
            # Remove consecutive duplicate coordinates
            deduplicated_coords = [rounded_coords[0]]
            for coord in rounded_coords[1:]:
                if coord != deduplicated_coords[-1]:
                    deduplicated_coords.append(coord)
            
            if len(deduplicated_coords) >= 2:
                processed.append({
                    'id': waterway['id'],
                    'coordinates': deduplicated_coords,
                    'tags': waterway['tags']
                })
        
        return processed
    
    def _simplify_geometries(self, waterways: List[Dict]) -> List[Dict]:
        """Simplify waterway geometries to reduce size while preserving topology."""
        if not self.config.enable_geometry_simplification:
            logger.info("Geometry simplification disabled, skipping...")
            return waterways
        
        # Use parallel processing for better performance with large datasets
        if self.config.parallel_workers > 1 and len(waterways) > 500:
            logger.info(f"Simplifying {len(waterways)} geometries using {self.config.parallel_workers} parallel workers")
            # Try multiprocessing first for better CPU utilization
            try:
                return self._simplify_geometries_multiprocessing(waterways)
            except Exception as e:
                logger.warning(f"Multiprocessing failed ({e}), falling back to threading")
                return self._simplify_geometries_parallel(waterways)
        else:
            logger.info(f"Simplifying {len(waterways)} geometries sequentially")
            return self._simplify_geometries_sequential(waterways)
    
    def _simplify_geometries_multiprocessing(self, waterways: List[Dict]) -> List[Dict]:
        """Simplify geometries using multiprocessing for improved performance."""
        logger.info(f"Simplifying geometries with tolerance {self.config.simplification_tolerance_m}m")
        
        # Split waterways into chunks for parallel processing
        chunk_size = max(1, len(waterways) // self.config.parallel_workers)
        waterway_chunks = [waterways[i:i + chunk_size] for i in range(0, len(waterways), chunk_size)]
        
        # Prepare work items for multiprocessing
        work_items = [(chunk, self.config.simplification_tolerance_m) for chunk in waterway_chunks]
        
        all_simplified = []
        
        with multiprocessing.Pool(processes=self.config.parallel_workers) as pool:
            try:
                chunk_results = pool.map(_simplify_geometry_chunk, work_items)
                
                # Flatten results
                for chunk_simplified in chunk_results:
                    all_simplified.extend(chunk_simplified)
                    
            except Exception as e:
                logger.error(f"Error in multiprocessing geometry simplification: {e}")
                pool.terminate()
                pool.join()
                raise
        
        logger.info(f"Simplified {len(all_simplified)}/{len(waterways)} waterways")
        return all_simplified
    
    def _simplify_geometries_sequential(self, waterways: List[Dict]) -> List[Dict]:
        """Simplify geometries sequentially (original implementation)."""
        logger.info(f"Simplifying geometries with tolerance {self.config.simplification_tolerance_m}m")
        simplified = []
        
        for waterway in waterways:
            coords = waterway['coordinates']
            if len(coords) < 2:
                continue
                
            try:
                # Convert to LineString
                line = LineString([(lon, lat) for lat, lon in coords])
                
                # Calculate simplification tolerance in degrees (rough approximation)
                # 1 meter ≈ 0.00001 degrees at equator
                tolerance_degrees = self.config.simplification_tolerance_m * 0.00001
                
                # Simplify the geometry
                simplified_line = line.simplify(tolerance_degrees, preserve_topology=True)
                
                # Convert back to coordinates
                if simplified_line.geom_type == 'LineString':
                    simplified_coords = [(lat, lon) for lon, lat in simplified_line.coords]
                    
                    # Ensure we still have at least 2 points
                    if len(simplified_coords) >= 2:
                        simplified.append({
                            'id': waterway['id'],
                            'coordinates': simplified_coords,
                            'tags': waterway['tags']
                        })
                        
            except Exception as e:
                logger.debug(f"Error simplifying waterway {waterway['id']}: {e}")
                # Fall back to original if simplification fails
                simplified.append(waterway)
        
        logger.info(f"Simplified {len(simplified)}/{len(waterways)} waterways")
        return simplified
    
    def _simplify_geometries_parallel(self, waterways: List[Dict]) -> List[Dict]:
        """Simplify geometries using parallel processing for improved performance."""
        logger.info(f"Simplifying geometries with tolerance {self.config.simplification_tolerance_m}m")
        
        # Split waterways into chunks for parallel processing
        chunk_size = max(1, len(waterways) // self.config.parallel_workers)
        waterway_chunks = [waterways[i:i + chunk_size] for i in range(0, len(waterways), chunk_size)]
        
        all_simplified = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit tasks for each chunk
            future_to_chunk = {
                executor.submit(self._simplify_geometries_chunk, chunk): chunk 
                for chunk in waterway_chunks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_simplified = future.result()
                    all_simplified.extend(chunk_simplified)
                except Exception as e:
                    logger.error(f"Error simplifying geometry chunk of size {len(chunk)}: {e}")
                    # Fall back to sequential processing for this chunk
                    all_simplified.extend(self._simplify_geometries_chunk(chunk))
        
        logger.info(f"Simplified {len(all_simplified)}/{len(waterways)} waterways")
        return all_simplified
    
    def _simplify_geometries_chunk(self, waterway_chunk: List[Dict]) -> List[Dict]:
        """Simplify a chunk of waterway geometries."""
        simplified = []
        tolerance_degrees = self.config.simplification_tolerance_m * 0.00001
        
        for waterway in waterway_chunk:
            coords = waterway['coordinates']
            if len(coords) < 2:
                continue
                
            try:
                # Convert to LineString
                line = LineString([(lon, lat) for lat, lon in coords])
                
                # Simplify the geometry
                simplified_line = line.simplify(tolerance_degrees, preserve_topology=True)
                
                # Convert back to coordinates
                if simplified_line.geom_type == 'LineString':
                    simplified_coords = [(lat, lon) for lon, lat in simplified_line.coords]
                    
                    # Ensure we still have at least 2 points
                    if len(simplified_coords) >= 2:
                        simplified.append({
                            'id': waterway['id'],
                            'coordinates': simplified_coords,
                            'tags': waterway['tags']
                        })
                        
            except Exception as e:
                logger.debug(f"Error simplifying waterway {waterway['id']}: {e}")
                # Fall back to original if simplification fails
                simplified.append(waterway)
        
        return simplified
    
    def _detect_and_split_intersections(self, waterways: List[Dict]) -> List[Dict]:
        """Detect line intersections and split waterways at intersection points (Specification 3.5)."""
        if len(waterways) < 2:
            logger.info("Less than 2 waterways, skipping intersection detection")
            return waterways
            
        logger.info(f"Detecting intersections between {len(waterways)} waterways...")
        
        # Create spatial index for efficient intersection detection
        try:
            from rtree import index
            spatial_index = index.Index()
            
            # Index all waterway line segments  
            waterway_lines = {}
            for i, waterway in enumerate(waterways):
                coords = waterway['coordinates']
                if len(coords) >= 2:
                    line = LineString([(lon, lat) for lat, lon in coords])
                    waterway_lines[i] = line
                    # Add to spatial index using bounds
                    spatial_index.insert(i, line.bounds)
            
            logger.info(f"Created spatial index with {len(waterway_lines)} waterway lines")
            
            # Find all intersection points
            intersection_points = []
            intersected_waterways = set()
            
            for i, line_i in waterway_lines.items():
                # Query spatial index for potential intersections
                potential_intersections = list(spatial_index.intersection(line_i.bounds))
                
                for j in potential_intersections:
                    if j <= i:  # Avoid duplicate checks and self-intersection
                        continue
                        
                    line_j = waterway_lines[j]
                    
                    try:
                        # Check for actual geometric intersection
                        intersection = line_i.intersection(line_j)
                        
                        if not intersection.is_empty:
                            if intersection.geom_type == 'Point' and isinstance(intersection, Point):
                                # Single intersection point
                                point = intersection
                                lat, lon = point.y, point.x
                                intersection_points.append((lat, lon))
                                intersected_waterways.add(i)
                                intersected_waterways.add(j)
                                logger.debug(f"Found intersection between waterway {waterways[i]['id']} and {waterways[j]['id']} at ({lat:.6f}, {lon:.6f})")
                                
                            elif intersection.geom_type == 'MultiPoint':
                                # Multiple intersection points
                                for point in intersection.geoms:
                                    if isinstance(point, Point):
                                        lat, lon = point.y, point.x
                                        intersection_points.append((lat, lon))
                                        intersected_waterways.add(i)
                                        intersected_waterways.add(j)
                                        logger.debug(f"Found intersection between waterway {waterways[i]['id']} and {waterways[j]['id']} at ({lat:.6f}, {lon:.6f})")
                                    
                            # Note: LineString intersections (overlapping segments) are more complex and less common
                            # For now, we focus on point intersections which solve the reported issue
                                    
                    except Exception as e:
                        logger.debug(f"Error checking intersection between waterways {waterways[i]['id']} and {waterways[j]['id']}: {e}")
                        continue
                        
            logger.info(f"Found {len(intersection_points)} intersection points affecting {len(intersected_waterways)} waterways")
            
            if not intersection_points:
                logger.info("No intersections found, returning original waterways")
                return waterways
                
            # Split waterways at intersection points
            return self._split_waterways_at_intersections(waterways, intersection_points)
            
        except ImportError:
            logger.warning("rtree not available, falling back to basic intersection detection")
            return self._detect_intersections_basic(waterways)
        except Exception as e:
            logger.error(f"Error in intersection detection: {e}")
            logger.warning("Falling back to original waterways without intersection detection")
            return waterways
    
    def _detect_intersections_basic(self, waterways: List[Dict]) -> List[Dict]:
        """Basic intersection detection without spatial indexing (fallback)."""
        logger.info("Using basic O(n²) intersection detection...")
        
        intersection_points = []
        
        for i in range(len(waterways)):
            for j in range(i + 1, len(waterways)):
                try:
                    coords_i = waterways[i]['coordinates']
                    coords_j = waterways[j]['coordinates']
                    
                    if len(coords_i) < 2 or len(coords_j) < 2:
                        continue
                        
                    line_i = LineString([(lon, lat) for lat, lon in coords_i])
                    line_j = LineString([(lon, lat) for lat, lon in coords_j])
                    
                    intersection = line_i.intersection(line_j)
                    
                    if not intersection.is_empty and intersection.geom_type == 'Point' and isinstance(intersection, Point):
                        lat, lon = intersection.y, intersection.x
                        intersection_points.append((lat, lon))
                        logger.debug(f"Found intersection between waterway {waterways[i]['id']} and {waterways[j]['id']} at ({lat:.6f}, {lon:.6f})")
                        
                except Exception as e:
                    logger.debug(f"Error checking intersection between waterways {waterways[i]['id']} and {waterways[j]['id']}: {e}")
                    continue
                    
        logger.info(f"Found {len(intersection_points)} intersection points")
        
        if not intersection_points:
            return waterways
            
        return self._split_waterways_at_intersections(waterways, intersection_points)
    
    def _split_waterways_at_intersections(self, waterways: List[Dict], intersection_points: List[Tuple[float, float]]) -> List[Dict]:
        """Split waterways by inserting intersection points into their coordinate sequences."""
        if not intersection_points:
            return waterways
            
        logger.info(f"Splitting waterways at {len(intersection_points)} intersection points...")
        
        modified_waterways = []
        split_count = 0
        
        for waterway in waterways:
            coords = waterway['coordinates']
            if len(coords) < 2:
                modified_waterways.append(waterway)
                continue
                
            # Create LineString for this waterway
            line = LineString([(lon, lat) for lat, lon in coords])
            
            # Find intersection points that lie on this line
            points_on_line = []
            for int_lat, int_lon in intersection_points:
                int_point = Point(int_lon, int_lat)
                
                # Check if intersection point lies on this line (with small tolerance for floating point precision)
                distance_to_line = line.distance(int_point)
                tolerance_degrees = self.config.snap_tolerance_m * 0.00001  # rough conversion to degrees
                
                if distance_to_line < tolerance_degrees:
                    # Find the position along the line where this intersection occurs
                    position = line.project(int_point)
                    points_on_line.append((position, int_lat, int_lon))
                    
            if not points_on_line:
                # No intersections on this waterway
                modified_waterways.append(waterway)
                continue
                
            # Sort intersection points by their position along the line
            points_on_line.sort(key=lambda x: x[0])
            
            # Insert intersection points into the coordinate sequence
            new_coords = []
            
            for i, coord in enumerate(coords):
                new_coords.append(coord)
                
                # If not the last coordinate, check for intersections in the next segment
                if i < len(coords) - 1:
                    next_coord = coords[i + 1]
                    current_segment_start = line.project(Point(coord[1], coord[0]))
                    current_segment_end = line.project(Point(next_coord[1], next_coord[0]))
                    
                    # Find intersection points that fall within this segment
                    for position, int_lat, int_lon in points_on_line:
                        point_distance = position
                        
                        # Check if intersection point is in this segment
                        if current_segment_start < point_distance < current_segment_end:
                            # Round to the configured precision
                            rounded_lat = round(int_lat, self.config.coordinate_precision)
                            rounded_lon = round(int_lon, self.config.coordinate_precision)
                            
                            # Only add if it's not already present (avoid duplicates)
                            if (rounded_lat, rounded_lon) not in new_coords:
                                new_coords.append((rounded_lat, rounded_lon))
                                split_count += 1
                                logger.debug(f"Inserted intersection point ({rounded_lat:.6f}, {rounded_lon:.6f}) into waterway {waterway['id']}")
            
            # Remove consecutive duplicate coordinates
            deduplicated_coords = [new_coords[0]] if new_coords else []
            for coord in new_coords[1:]:
                if coord != deduplicated_coords[-1]:
                    deduplicated_coords.append(coord)
            
            # Create updated waterway with intersection points inserted
            modified_waterways.append({
                'id': waterway['id'],
                'coordinates': deduplicated_coords,
                'tags': waterway['tags']
            })
            
        logger.info(f"Inserted {split_count} intersection points into waterway coordinate sequences")
        return modified_waterways
    
    def _extract_endpoints_and_junctions(self, waterways: List[Dict]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Extract unique endpoints and identify junction points."""
        # Use parallel processing for better performance with large datasets
        # Based on performance analysis: break-even point is around 500 waterways
        if self.config.parallel_workers > 1 and len(waterways) > 500:
            logger.info(f"Extracting endpoints from {len(waterways)} waterways using {self.config.parallel_workers} parallel workers")
            # Try multiprocessing first for better CPU utilization
            try:
                return self._extract_endpoints_multiprocessing(waterways)
            except Exception as e:
                logger.warning(f"Multiprocessing failed ({e}), falling back to threading")
                return self._extract_endpoints_parallel(waterways)
        else:
            logger.info(f"Extracting endpoints from {len(waterways)} waterways sequentially")
            return self._extract_endpoints_sequential(waterways)
    
    def _extract_endpoints_multiprocessing(self, waterways: List[Dict]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Extract endpoints using multiprocessing for improved performance."""
        # Split waterways into chunks for parallel processing
        chunk_size = max(1, len(waterways) // self.config.parallel_workers)
        waterway_chunks = [waterways[i:i + chunk_size] for i in range(0, len(waterways), chunk_size)]
        
        # Prepare work items for multiprocessing
        work_items = [(chunk,) for chunk in waterway_chunks]
        
        all_endpoint_counts = []
        all_coordinates_counts = []
        all_endpoints = []
        
        with multiprocessing.Pool(processes=self.config.parallel_workers) as pool:
            try:
                chunk_results = pool.map(_extract_endpoints_chunk, work_items)
                
                # Collect results
                for chunk_endpoint_count, chunk_all_coords_count, chunk_endpoints in chunk_results:
                    all_endpoint_counts.append(chunk_endpoint_count)
                    all_coordinates_counts.append(chunk_all_coords_count)
                    all_endpoints.extend(chunk_endpoints)
                    
            except Exception as e:
                logger.error(f"Error in multiprocessing endpoint extraction: {e}")
                pool.terminate()
                pool.join()
                raise
        
        # Merge all endpoint counts
        endpoint_count = Counter()
        for count_dict in all_endpoint_counts:
            endpoint_count.update(count_dict)
        
        # Merge all coordinate counts
        all_coordinates_count = Counter()
        for count_dict in all_coordinates_counts:
            all_coordinates_count.update(count_dict)
        
        # Coordinates that appear more than once are junctions (includes both endpoint and interior junctions)
        endpoint_junctions = [coord for coord, count in endpoint_count.items() if count > 1]
        interior_junctions = [coord for coord, count in all_coordinates_count.items() 
                             if count > 1 and coord not in endpoint_junctions]
        
        # Combine both types of junctions
        all_junctions = endpoint_junctions + interior_junctions
        endpoints = list(set(all_endpoints))
        
        logger.info(f"Found {len(endpoints)} unique endpoints, {len(endpoint_junctions)} endpoint junctions, {len(interior_junctions)} interior junctions")
        return endpoints, all_junctions
    
    def _extract_endpoints_sequential(self, waterways: List[Dict]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Extract endpoints sequentially and detect intersection junctions."""
        endpoint_count = Counter()
        all_coordinates_count = Counter()  # Count ALL coordinates, not just endpoints
        all_endpoints = []
        
        # Count how many times each coordinate appears as an endpoint
        for waterway in waterways:
            coords = waterway['coordinates']
            start_coord = coords[0]
            end_coord = coords[-1]
            
            endpoint_count[start_coord] += 1
            endpoint_count[end_coord] += 1
            all_endpoints.extend([start_coord, end_coord])
            
            # Also count ALL coordinates to detect interior intersections
            for coord in coords:
                all_coordinates_count[coord] += 1
        
        # Coordinates that appear more than once are junctions (includes both endpoint and interior junctions)
        endpoint_junctions = [coord for coord, count in endpoint_count.items() if count > 1]
        interior_junctions = [coord for coord, count in all_coordinates_count.items() 
                             if count > 1 and coord not in endpoint_junctions]
        
        # Combine both types of junctions
        all_junctions = endpoint_junctions + interior_junctions
        endpoints = list(set(all_endpoints))
        
        logger.info(f"Found {len(endpoints)} unique endpoints, {len(endpoint_junctions)} endpoint junctions, {len(interior_junctions)} interior junctions")
        return endpoints, all_junctions
    
    def _extract_endpoints_parallel(self, waterways: List[Dict]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Extract endpoints using parallel processing for improved performance."""
        # Split waterways into chunks for parallel processing
        chunk_size = max(1, len(waterways) // self.config.parallel_workers)
        waterway_chunks = [waterways[i:i + chunk_size] for i in range(0, len(waterways), chunk_size)]
        
        all_endpoint_counts = []
        all_coordinates_counts = []
        all_endpoints = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit tasks for each chunk
            future_to_chunk = {
                executor.submit(self._extract_endpoints_chunk, chunk): chunk 
                for chunk in waterway_chunks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_endpoint_count, chunk_all_coords_count, chunk_endpoints = future.result()
                    all_endpoint_counts.append(chunk_endpoint_count)
                    all_coordinates_counts.append(chunk_all_coords_count)
                    all_endpoints.extend(chunk_endpoints)
                except Exception as e:
                    logger.error(f"Error extracting endpoints from chunk of size {len(chunk)}: {e}")
                    # Fall back to sequential processing for this chunk
                    chunk_endpoint_count, chunk_all_coords_count, chunk_endpoints = self._extract_endpoints_chunk(chunk)
                    all_endpoint_counts.append(chunk_endpoint_count)
                    all_coordinates_counts.append(chunk_all_coords_count)
                    all_endpoints.extend(chunk_endpoints)
        
        # Merge all endpoint counts
        endpoint_count = Counter()
        for count_dict in all_endpoint_counts:
            endpoint_count.update(count_dict)
        
        # Merge all coordinate counts
        all_coordinates_count = Counter()
        for count_dict in all_coordinates_counts:
            all_coordinates_count.update(count_dict)
        
        # Coordinates that appear more than once are junctions (includes both endpoint and interior junctions)
        endpoint_junctions = [coord for coord, count in endpoint_count.items() if count > 1]
        interior_junctions = [coord for coord, count in all_coordinates_count.items() 
                             if count > 1 and coord not in endpoint_junctions]
        
        # Combine both types of junctions
        all_junctions = endpoint_junctions + interior_junctions
        endpoints = list(set(all_endpoints))
        
        logger.info(f"Found {len(endpoints)} unique endpoints, {len(endpoint_junctions)} endpoint junctions, {len(interior_junctions)} interior junctions")
        return endpoints, all_junctions
    
    def _extract_endpoints_chunk(self, waterway_chunk: List[Dict]) -> Tuple[Counter, Counter, List[Tuple[float, float]]]:
        """Extract endpoints from a chunk of waterways."""
        endpoint_count = Counter()
        all_coordinates_count = Counter()
        all_endpoints = []
        
        # Count how many times each coordinate appears as an endpoint
        for waterway in waterway_chunk:
            coords = waterway['coordinates']
            start_coord = coords[0]
            end_coord = coords[-1]
            
            endpoint_count[start_coord] += 1
            endpoint_count[end_coord] += 1
            all_endpoints.extend([start_coord, end_coord])
            
            # Also count ALL coordinates to detect interior intersections
            for coord in coords:
                all_coordinates_count[coord] += 1
        
        return endpoint_count, all_coordinates_count, all_endpoints
    
    def _create_edges(self, waterways: List[Dict], coord_mapping: Dict) -> List[Dict]:
        """Create edges with accurate geodesic distances and deterministic IDs."""
        # Use parallel processing for better performance
        # Based on performance analysis: multiprocessing overhead makes it slower for small datasets
        # Break-even point is around 500 waterways, so use 500 as threshold
        if self.config.parallel_workers > 1 and len(waterways) > 500:
            logger.info(f"Processing {len(waterways)} waterways using {self.config.parallel_workers} parallel workers")
            # Try ProcessPoolExecutor for CPU-intensive tasks, fall back to ThreadPoolExecutor
            try:
                return self._create_edges_multiprocessing(waterways, coord_mapping)
            except Exception as e:
                logger.warning(f"Multiprocessing failed ({e}), falling back to threading")
                return self._create_edges_parallel(waterways, coord_mapping)
        else:
            logger.info(f"Processing {len(waterways)} waterways sequentially")
            return self._create_edges_sequential(waterways, coord_mapping)
    
    def _create_edges_multiprocessing(self, waterways: List[Dict], coord_mapping: Dict) -> List[Dict]:
        """Create edges using multiprocessing with vectorized distance calculations."""
        # Prepare configuration for worker processes
        config_dict = {
            'coordinate_precision': self.config.coordinate_precision,
            'hash_length': self.config.hash_length,
            'min_fragment_length_m': self.config.min_fragment_length_m
        }
        
        # Optimize coordinate mapping for more efficient serialization
        optimized_coord_mapping = _optimize_coord_mapping_for_multiprocessing(coord_mapping, config_dict)
        
        # Use smaller batches for better parallelization and less memory usage per process
        batch_size = max(1, len(waterways) // (self.config.parallel_workers * 2))
        waterway_batches = [
            waterways[i:i + batch_size] 
            for i in range(0, len(waterways), batch_size)
        ]
        
        # Prepare work items for multiprocessing
        work_items = [(batch, optimized_coord_mapping, config_dict) for batch in waterway_batches]
        
        all_edges = []
        
        # Use spawn method for better isolation (important for complex objects)
        with multiprocessing.Pool(
            processes=self.config.parallel_workers,
            initializer=_init_worker_geod,
            initargs=("geodesic",)
        ) as pool:
            try:
                # Process waterway batches in parallel
                batch_results = pool.map(_process_waterway_batch_optimized, work_items)
                
                # Flatten results
                for batch_edges in batch_results:
                    all_edges.extend(batch_edges)
                    
            except Exception as e:
                logger.error(f"Error in multiprocessing: {e}")
                pool.terminate()
                pool.join()
                raise
        
        # Filter by minimum length
        filtered_edges = []
        for edge in all_edges:
            if edge['length_m'] >= self.config.min_fragment_length_m:
                filtered_edges.append(edge)
        
        logger.info(f"Created {len(filtered_edges)}/{len(all_edges)} edges after length filtering")
        return filtered_edges
    
    def _create_edges_sequential(self, waterways: List[Dict], coord_mapping: Dict) -> List[Dict]:
        """Create edges sequentially (original implementation)."""
        edges = []
        
        for waterway in waterways:
            coords = waterway['coordinates']
            way_id = waterway['id']
            tags = waterway['tags']
            
            # Apply coordinate mapping from clustering
            mapped_coords = [coord_mapping.get(coord, coord) for coord in coords]
            
            # Split waterway at junction points
            edges.extend(self._split_waterway_at_junctions(mapped_coords, way_id, tags, coord_mapping))
        
        # Filter by minimum length
        filtered_edges = []
        for edge in edges:
            if edge['length_m'] >= self.config.min_fragment_length_m:
                filtered_edges.append(edge)
        
        logger.info(f"Created {len(filtered_edges)}/{len(edges)} edges after length filtering")
        return filtered_edges
    
    def _create_edges_parallel(self, waterways: List[Dict], coord_mapping: Dict) -> List[Dict]:
        """Create edges using parallel processing for improved performance."""
        # Split waterways into chunks for parallel processing
        chunk_size = max(1, len(waterways) // self.config.parallel_workers)
        waterway_chunks = [waterways[i:i + chunk_size] for i in range(0, len(waterways), chunk_size)]
        
        all_edges = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit tasks for each chunk
            future_to_chunk = {
                executor.submit(self._process_waterway_chunk, chunk, coord_mapping): chunk 
                for chunk in waterway_chunks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_edges = future.result()
                    all_edges.extend(chunk_edges)
                except Exception as e:
                    logger.error(f"Error processing waterway chunk of size {len(chunk)}: {e}")
                    # Fall back to sequential processing for this chunk
                    for waterway in chunk:
                        coords = waterway['coordinates']
                        way_id = waterway['id']
                        tags = waterway['tags']
                        mapped_coords = [coord_mapping.get(coord, coord) for coord in coords]
                        all_edges.extend(self._split_waterway_at_junctions(mapped_coords, way_id, tags, coord_mapping))
        
        # Filter by minimum length
        filtered_edges = []
        for edge in all_edges:
            if edge['length_m'] >= self.config.min_fragment_length_m:
                filtered_edges.append(edge)
        
        logger.info(f"Created {len(filtered_edges)}/{len(all_edges)} edges after length filtering")
        return filtered_edges
    
    def _process_waterway_chunk(self, waterway_chunk: List[Dict], coord_mapping: Dict) -> List[Dict]:
        """Process a chunk of waterways in parallel."""
        edges = []
        
        for waterway in waterway_chunk:
            coords = waterway['coordinates']
            way_id = waterway['id']
            tags = waterway['tags']
            
            # Apply coordinate mapping from clustering
            mapped_coords = [coord_mapping.get(coord, coord) for coord in coords]
            
            # Split waterway at junction points
            edges.extend(self._split_waterway_at_junctions(mapped_coords, way_id, tags, coord_mapping))
        
        return edges
    
    def _split_waterway_at_junctions(self, coords: List[Tuple[float, float]], way_id: int, 
                                   tags: Dict, coord_mapping: Dict) -> List[Dict]:
        """Split a waterway into edges at junction points."""
        if len(coords) < 2:
            return []
        
        # Find all junction points in this waterway
        junction_indices = []
        junction_coords = set(coord for coord, mapped in coord_mapping.items() 
                            if coord != mapped or self._is_junction_coord(coord, coord_mapping))
        
        for i, coord in enumerate(coords):
            if coord in junction_coords or i == 0 or i == len(coords) - 1:
                junction_indices.append(i)
        
        # Create edges between junction points
        edges = []
        for i in range(len(junction_indices) - 1):
            start_idx = junction_indices[i]
            end_idx = junction_indices[i + 1]
            
            if end_idx > start_idx:
                segment_coords = coords[start_idx:end_idx + 1]
                
                if len(segment_coords) >= 2:
                    # Calculate accurate geodesic length
                    length_m = self.geod_calc.calculate_segment_length(segment_coords)
                    
                    # Generate deterministic IDs
                    start_coord = segment_coords[0]
                    end_coord = segment_coords[-1]
                    
                    start_node_id = self.id_generator.generate_node_id(start_coord[0], start_coord[1])
                    end_node_id = self.id_generator.generate_node_id(end_coord[0], end_coord[1])
                    edge_id = self.id_generator.generate_edge_id(start_node_id, end_node_id, way_id, i)
                    
                    # Extract and normalize width information
                    width_info = self._parse_width_tags(tags)
                    
                    edges.append({
                        'id': edge_id,
                        'from_node_id': start_node_id,
                        'to_node_id': end_node_id,
                        'length_m': length_m,
                        'coordinates': segment_coords,
                        'name': tags.get('name', ''),
                        'type': tags.get('waterway', ''),
                        'width_raw': width_info['raw'],
                        'width_m': width_info['meters'],
                        'width_source': width_info['source'],
                        'original_way_id': way_id
                    })
        
        return edges
    
    def _is_junction_coord(self, coord: Tuple[float, float], coord_mapping: Dict) -> bool:
        """Check if a coordinate is a junction point."""
        # This is a simplified check - in a full implementation, 
        # we'd track junction coordinates more carefully
        return coord in coord_mapping
    
    def _parse_width_tags(self, tags: Dict) -> Dict[str, Any]:
        """Parse width information from OSM tags."""
        width_raw = tags.get('width', '')
        width_m = None
        width_source = 'none'
        
        if width_raw:
            width_source = 'tag'
            try:
                # Handle common width formats
                width_str = width_raw.lower().strip()
                
                if 'm' in width_str:
                    # "5 m", "5m", "5.5 m"
                    width_m = float(width_str.replace('m', '').strip())
                elif 'ft' in width_str or 'feet' in width_str:
                    # "15 ft", "15 feet"
                    feet = float(width_str.replace('ft', '').replace('feet', '').strip())
                    width_m = feet * 0.3048  # Convert feet to meters
                elif width_str.replace('.', '').isdigit():
                    # Assume meters if just a number
                    width_m = float(width_str)
                
                # Validate reasonable width values
                if width_m is not None and (width_m <= 0 or width_m > 1000):
                    width_m = None  # Invalid width
                    
            except (ValueError, AttributeError):
                pass  # Keep width_m as None for unparseable values
        
        return {
            'raw': width_raw,
            'meters': width_m,
            'source': width_source
        }
    
    def _build_nodes(self, coord_mapping: Dict) -> List[Dict]:
        """Build final node list with deterministic IDs."""
        # Get all unique final coordinates after clustering
        unique_coords = set(coord_mapping.values())
        
        nodes = []
        for coord in unique_coords:
            lat, lon = coord
            node_id = self.id_generator.generate_node_id(lat, lon)
            
            nodes.append({
                'id': node_id,
                'lat': lat,
                'lon': lon,
                'type': 'junction'  # Could be refined to distinguish endpoints vs junctions
            })
        
        return nodes
    
    def _generate_qa_metrics(self, original_waterways: List[Dict], processed_waterways: List[Dict], 
                           nodes: List[Dict], edges: List[Dict], processing_time: float):
        """Generate comprehensive QA metrics per specification."""
        self.qa_metrics = {
            'processing_time_seconds': processing_time,
            'original_waterways': len(original_waterways),
            'processed_waterways': len(processed_waterways),
            'final_nodes': len(nodes),
            'final_edges': len(edges),
            'distance_calculation_method': self.config.distance_calculation_method,
            'coordinate_precision': self.config.coordinate_precision,
            'snap_tolerance_m': self.config.snap_tolerance_m,
            'min_fragment_length_m': self.config.min_fragment_length_m
        }
        
        # Add clustering metrics
        self.qa_metrics.update(self.clusterer.cluster_metrics)
        
        # Calculate edge statistics
        if edges:
            edge_lengths = [edge['length_m'] for edge in edges]
            edge_lengths.sort()
            n = len(edge_lengths)
            
            self.qa_metrics.update({
                'edge_length_p5_m': edge_lengths[int(n * 0.05)],
                'edge_length_p50_m': edge_lengths[int(n * 0.5)],
                'edge_length_p95_m': edge_lengths[int(n * 0.95)],
                'mean_edge_length_m': sum(edge_lengths) / n
            })
        
        # Calculate width parsing statistics
        edges_with_width_raw = sum(1 for edge in edges if edge.get('width_raw'))
        edges_with_width_m = sum(1 for edge in edges if edge.get('width_m') is not None)
        
        self.qa_metrics.update({
            'pct_edges_with_width_raw': (edges_with_width_raw / len(edges) * 100) if edges else 0,
            'pct_edges_with_width_m': (edges_with_width_m / len(edges) * 100) if edges else 0,
            'width_parse_success_rate': (edges_with_width_m / edges_with_width_raw * 100) if edges_with_width_raw else 0
        })


def extract_waterways(pbf_file: str, config: Config, cache_file: Optional[str] = None) -> List[Dict]:
    """Extract waterways from PBF file with optional caching."""
    if cache_file and os.path.exists(cache_file) and config.reuse_extraction:
        logger.info(f"Loading waterways from cache: {cache_file}")
        with gzip.open(cache_file, 'rt', encoding='utf-8') as f:
            return json.load(f)
    
    logger.info(f"Extracting waterways from PBF: {pbf_file}")
    
    handler = WaterwayHandler(config)
    # Apply with locations to get node coordinates for ways
    handler.apply_file(pbf_file, locations=True)
    
    logger.info(f"Extracted {len(handler.waterways)} waterways")
    
    # Log debugging info about processing
    if handler.matching_waterways > 0:
        logger.info(f"Found {handler.matching_waterways} waterways matching filter, successfully processed {handler.processed_waterways}")
    
    # Log what waterway types were found for debugging
    if handler.waterway_type_counts:
        logger.info("Waterway types found in file:")
        for waterway_type, count in handler.waterway_type_counts.most_common():
            logger.info(f"  {waterway_type}: {count}")
    else:
        logger.warning("No waterway tags found in the file at all")
    
    # Save to cache if requested
    if cache_file:
        logger.info(f"Saving waterways to cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with gzip.open(cache_file, 'wt', encoding='utf-8') as f:
            json.dump(handler.waterways, f)
    
    return handler.waterways


class OutputManager:
    """Handles multiple output formats per specification."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def save_outputs(self, nodes: List[Dict], edges: List[Dict], base_filename: str, 
                    qa_metrics: Dict, id_generator: IDGenerator) -> Dict[str, int]:
        """Save outputs in all configured formats and return file sizes."""
        file_sizes = {}
        
        # Server outputs with deterministic IDs
        if "parquet" in self.config.server_formats:
            file_sizes.update(self._save_parquet(nodes, edges, base_filename))
        
        if "csv" in self.config.server_formats:
            file_sizes.update(self._save_csv(nodes, edges, base_filename))
        
        if "geojson" in self.config.server_formats:
            file_sizes.update(self._save_geojson(edges, base_filename))
        
        # JSON GZ format if enabled (compatible with original format)
        if "jsongz" in self.config.server_formats:
            file_sizes.update(self._save_json_gz_format(nodes, edges, base_filename))
        
        # Mobile outputs with sequential IDs (only if enabled)
        if self.config.generate_mobile_csv:
            mobile_nodes, mobile_edges = self._convert_to_mobile_format(nodes, edges, id_generator)
            file_sizes.update(self._save_mobile_csv(mobile_nodes, mobile_edges, base_filename))
        
        # Save QA metrics (always generated as it's essential for validation)
        qa_file = f"{base_filename}.qa_summary.json"
        self._save_json(qa_metrics, qa_file)
        file_sizes[qa_file] = os.path.getsize(qa_file)
        
        # Save ID mapping for reference (only if enabled)
        if self.config.generate_id_mapping:
            mapping_file = f"{base_filename}.id_mapping.json"
            self._save_json(id_generator.mobile_id_mapping, mapping_file)
            file_sizes[mapping_file] = os.path.getsize(mapping_file)
        
        return file_sizes
    
    def _save_json_gz_format(self, nodes: List[Dict], edges: List[Dict], base_filename: str) -> Dict[str, int]:
        """Save outputs in JSON GZ format (compatible with original format)."""
        file_sizes = {}
        
        # Convert to JSON GZ format
        json_gz_nodes, json_gz_edges = self._convert_to_json_gz_format(nodes, edges)
        
        # Save JSON GZ files
        nodes_file = f"{base_filename}.nodes.json.gz"
        edges_file = f"{base_filename}.edges.json.gz"
        
        # Save gzipped JSON files exactly like the original script
        with gzip.open(nodes_file, 'wt', encoding='utf-8') as f:
            json.dump(json_gz_nodes, f, separators=(',', ':'))  # Compact JSON like original
        
        with gzip.open(edges_file, 'wt', encoding='utf-8') as f:
            json.dump(json_gz_edges, f, separators=(',', ':'))  # Compact JSON like original
        
        file_sizes[nodes_file] = os.path.getsize(nodes_file)
        file_sizes[edges_file] = os.path.getsize(edges_file)
        
        logger.info(f"Saved JSON GZ format: {nodes_file} ({file_sizes[nodes_file]:,} bytes)")
        logger.info(f"Saved JSON GZ format: {edges_file} ({file_sizes[edges_file]:,} bytes)")
        
        return file_sizes
    
    def _convert_to_json_gz_format(self, nodes: List[Dict], edges: List[Dict]) -> Tuple[List[List[float]], List[Dict]]:
        """Convert modern format to JSON GZ format exactly matching the original script."""
        
        # Create node ID to index mapping
        node_id_to_index = {}
        json_gz_nodes = []
        
        for i, node in enumerate(nodes):
            node_id_to_index[node['id']] = i
            # JSON GZ format: simple [lat, lon] arrays
            json_gz_nodes.append([node['lat'], node['lon']])
        
        # Convert edges to JSON GZ format
        json_gz_edges = []
        for edge in edges:
            # Map modern node IDs to integer indices
            start_index = node_id_to_index[edge['from_node_id']]
            end_index = node_id_to_index[edge['to_node_id']]
            
            # JSON GZ edge format matches original script exactly
            json_gz_edge = {
                'start': start_index,
                'end': end_index,
                'length': edge['length_m'],  # Use length_m from modern format
                'coordinates': edge['coordinates']  # Coordinates should already be [lat, lon] pairs
            }
            
            json_gz_edges.append(json_gz_edge)
        
        return json_gz_nodes, json_gz_edges
    
    def _save_parquet(self, nodes: List[Dict], edges: List[Dict], base_filename: str) -> Dict[str, int]:
        """Save nodes and edges as Parquet files."""
        nodes_file = f"{base_filename}.nodes.parquet"
        edges_file = f"{base_filename}.edges.parquet"
        
        # Convert to pandas DataFrames and save
        nodes_df = pd.DataFrame(nodes)
        edges_df = pd.DataFrame(edges)
        
        nodes_df.to_parquet(nodes_file, compression='snappy' if self.config.compression else None)
        edges_df.to_parquet(edges_file, compression='snappy' if self.config.compression else None)
        
        return {
            nodes_file: os.path.getsize(nodes_file),
            edges_file: os.path.getsize(edges_file)
        }
    
    def _save_csv(self, nodes: List[Dict], edges: List[Dict], base_filename: str) -> Dict[str, int]:
        """Save nodes and edges as CSV files."""
        nodes_file = f"{base_filename}.nodes.csv.gz" if self.config.compression else f"{base_filename}.nodes.csv"
        edges_file = f"{base_filename}.edges.csv.gz" if self.config.compression else f"{base_filename}.edges.csv"
        
        nodes_df = pd.DataFrame(nodes)
        edges_df = pd.DataFrame(edges)
        
        if self.config.compression:
            nodes_df.to_csv(nodes_file, index=False, compression='gzip')
            edges_df.to_csv(edges_file, index=False, compression='gzip')
        else:
            nodes_df.to_csv(nodes_file, index=False)
            edges_df.to_csv(edges_file, index=False)
        
        return {
            nodes_file: os.path.getsize(nodes_file),
            edges_file: os.path.getsize(edges_file)
        }
    
    def _save_geojson(self, edges: List[Dict], base_filename: str) -> Dict[str, int]:
        """Save edges as GeoJSON for visualization."""
        geojson_file = f"{base_filename}.geojson"
        
        try:
            import geojson
            
            features = []
            for edge in edges:
                # Convert coordinates from (lat, lon) to (lon, lat) for GeoJSON
                coords = [(lon, lat) for lat, lon in edge['coordinates']]
                line = geojson.LineString(coords)
                
                properties = {
                    'id': edge['id'],
                    'length_m': edge['length_m'],
                    'name': edge.get('name', ''),
                    'type': edge.get('type', ''),
                    'width_m': edge.get('width_m')
                }
                
                features.append(geojson.Feature(geometry=line, properties=properties))
            
            fc = geojson.FeatureCollection(features)
            
            with open(geojson_file, 'w') as f:
                geojson.dump(fc, f)
            
            return {geojson_file: os.path.getsize(geojson_file)}
            
        except ImportError:
            logger.warning("geojson library not available, skipping GeoJSON export")
            return {}
    
    def _convert_to_mobile_format(self, nodes: List[Dict], edges: List[Dict], 
                                 id_generator: IDGenerator) -> Tuple[List[Dict], List[Dict]]:
        """Convert to mobile-optimized format with sequential IDs."""
        mobile_nodes = []
        mobile_edges = []
        
        for node in nodes:
            mobile_id = id_generator.get_mobile_node_id(node['id'])
            mobile_nodes.append({
                'id': mobile_id,
                'lat': node['lat'],
                'lon': node['lon']
            })
        
        for edge in edges:
            mobile_edge_id = id_generator.get_mobile_edge_id(edge['id'])
            mobile_from_id = id_generator.get_mobile_node_id(edge['from_node_id'])
            mobile_to_id = id_generator.get_mobile_node_id(edge['to_node_id'])
            
            mobile_edges.append({
                'id': mobile_edge_id,
                'from_node_id': mobile_from_id,
                'to_node_id': mobile_to_id,
                'length_m': edge['length_m'],
                'name': edge.get('name', ''),
                'type': edge.get('type', ''),
                'width_m': edge.get('width_m')
            })
        
        return mobile_nodes, mobile_edges
    
    def _save_mobile_csv(self, mobile_nodes: List[Dict], mobile_edges: List[Dict], 
                        base_filename: str) -> Dict[str, int]:
        """Save mobile-optimized CSV files."""
        nodes_file = f"{base_filename}.mobile.nodes.csv.gz"
        edges_file = f"{base_filename}.mobile.edges.csv.gz"
        
        nodes_df = pd.DataFrame(mobile_nodes)
        edges_df = pd.DataFrame(mobile_edges)
        
        nodes_df.to_csv(nodes_file, index=False, compression='gzip')
        edges_df.to_csv(edges_file, index=False, compression='gzip')
        
        return {
            nodes_file: os.path.getsize(nodes_file),
            edges_file: os.path.getsize(edges_file)
        }
    
    def _save_json(self, data: Any, filename: str):
        """Save data as JSON file."""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


class ManifestGenerator:
    """Generates run manifests for reproducibility."""
    
    @staticmethod
    def generate_manifest(input_file: str, config: Config, qa_metrics: Dict, 
                         file_sizes: Dict[str, int]) -> Dict:
        """Generate a comprehensive run manifest."""
        
        # Calculate input file hash
        input_hash = ManifestGenerator._calculate_file_hash(input_file)
        
        manifest = {
            'version': '2.1',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'input': {
                'file_path': input_file,
                'file_size_bytes': os.path.getsize(input_file) if os.path.exists(input_file) else 0,
                'sha256_hash': input_hash
            },
            'configuration': {
                'parameter_hash': config.get_parameter_hash(),
                'parameters': asdict(config)
            },
            'processing': {
                'duration_seconds': qa_metrics.get('processing_time_seconds', 0),
                'distance_calculation_method': qa_metrics.get('distance_calculation_method', 'unknown'),
                'software_version': '2.1'
            },
            'results': {
                'original_waterways': qa_metrics.get('original_waterways', 0),
                'final_nodes': qa_metrics.get('final_nodes', 0),
                'final_edges': qa_metrics.get('final_edges', 0),
                'output_files': file_sizes
            },
            'qa_summary': {
                'clustering': {
                    'total_clusters': qa_metrics.get('total_clusters', 0),
                    'displacement_p95_m': qa_metrics.get('displacement_p95_m', 0),
                    'largest_cluster_size': qa_metrics.get('largest_cluster_size', 0)
                },
                'quality': {
                    'width_parse_success_rate': qa_metrics.get('width_parse_success_rate', 0),
                    'mean_edge_length_m': qa_metrics.get('mean_edge_length_m', 0)
                }
            }
        }
        
        return manifest
    
    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        if not os.path.exists(file_path) or os.path.isdir(file_path):
            return ""
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()


def get_cache_filename(input_file: str, config: Config) -> str:
    """Generate cache filename based on input file and configuration."""
    base_name = Path(input_file).stem
    # Use extraction-specific parameter hash for selective cache reuse
    extraction_param_hash = config.get_step_parameter_hash("extraction")
    cache_dir = Path(config.cache_directory) / "extraction" / extraction_param_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"{base_name}.waterways.json.gz")


def get_intermediate_cache_filename(input_file: str, config: Config, step_name: str) -> str:
    """Generate intermediate cache filename for a specific processing step."""
    base_name = Path(input_file).stem
    # Use step-specific parameter hash for selective cache reuse
    step_param_hash = config.get_step_parameter_hash(step_name)
    cache_dir = Path(config.cache_directory) / "intermediate" / step_param_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"{base_name}.{step_name}.json.gz")


def save_intermediate_cache(data: Any, cache_file: str):
    """Save intermediate data to cache file."""
    logger.info(f"Saving intermediate data to cache: {cache_file}")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    # Convert coordinate tuples to strings for JSON serialization
    if isinstance(data, dict):
        serializable_data = {}
        for key, value in data.items():
            if isinstance(key, tuple):
                # Convert tuple key to string
                str_key = f"{key[0]},{key[1]}"
                serializable_data[str_key] = value
            else:
                serializable_data[key] = value
        data = serializable_data
    
    with gzip.open(cache_file, 'wt', encoding='utf-8') as f:
        json.dump(data, f)


def load_intermediate_cache(cache_file: str) -> Any:
    """Load intermediate data from cache file."""
    logger.info(f"Loading intermediate data from cache: {cache_file}")
    with gzip.open(cache_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert string keys back to coordinate tuples if needed
    if isinstance(data, dict):
        converted_data = {}
        for key, value in data.items():
            if isinstance(key, str) and ',' in key:
                try:
                    # Try to convert string back to tuple
                    parts = key.split(',')
                    if len(parts) == 2:
                        coord_tuple = (float(parts[0]), float(parts[1]))
                        converted_data[coord_tuple] = value
                    else:
                        converted_data[key] = value
                except ValueError:
                    converted_data[key] = value
            else:
                converted_data[key] = value
        data = converted_data
    
    # Convert coordinate lists back to tuples in lists/values
    def convert_coords_to_tuples(obj):
        if isinstance(obj, list):
            # Check if this is a coordinate pair [lat, lon]
            if len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
                return tuple(obj)
            else:
                # Process each item in the list
                return [convert_coords_to_tuples(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_coords_to_tuples(v) for k, v in obj.items()}
        else:
            return obj
    
    data = convert_coords_to_tuples(data)
    return data


def get_output_base_filename(input_file: str) -> str:
    """Generate base filename for outputs, including directory structure."""
    input_path = Path(input_file)
    
    # Create clean directory name, removing .osm.pbf or similar extensions
    if input_path.name.endswith('.osm.pbf'):
        directory_name = input_path.name.replace('.osm.pbf', '')
    elif input_path.name.endswith('.osm'):
        directory_name = input_path.stem
    else:
        directory_name = input_path.stem
    
    # Create the directory if it doesn't exist
    os.makedirs(directory_name, exist_ok=True)
    
    # Return the base path inside the directory
    return str(Path(directory_name) / directory_name)


def create_test_waterways() -> List[Dict]:
    """Create synthetic test waterways for validation."""
    # Create a test network: main river with a tributary and crossing waterways (to test intersection detection)
    return [
        {
            'id': 1,
            'coordinates': [(52.5, 13.4), (52.51, 13.41), (52.52, 13.42)],  # Main river
            'tags': {'waterway': 'river', 'name': 'Test River', 'width': '10 m'}
        },
        {
            'id': 2,
            'coordinates': [(52.51, 13.41), (52.515, 13.405), (52.52, 13.40)],  # Tributary 
            'tags': {'waterway': 'stream', 'name': 'Test Stream', 'width': '3 m'}
        },
        {
            'id': 3,
            'coordinates': [(52.52, 13.42), (52.53, 13.43), (52.54, 13.44)],  # Continuation
            'tags': {'waterway': 'river', 'name': 'Test River'}
        },
        {
            'id': 4,
            'coordinates': [(52.505, 13.405), (52.515, 13.415), (52.525, 13.425)],  # Crossing waterway (should intersect main river)
            'tags': {'waterway': 'stream', 'name': 'Test Crossing Stream', 'width': '2 m'}
        }
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Extract and clean waterway networks from OSM PBF files (v2.1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python osm_waterway_extractor.py brazil-latest.osm.pbf
  python osm_waterway_extractor.py data.osm.pbf --config custom_config.yaml
  python osm_waterway_extractor.py data.osm.pbf --snap-tolerance 5.0 --precision 6
  python osm_waterway_extractor.py data.osm.pbf --enable-json-gz-format

Configuration:
  Uses config.yaml by default. Command line options override config file settings.
  JSON GZ format can be enabled in config.yaml or with --enable-json-gz-format.

Attribution:
  © OpenStreetMap contributors. Data licensed under ODbL.
        """
    )
    
    parser.add_argument('input_file', help='Path to input OSM PBF file (use "test" for synthetic test data)')
    parser.add_argument('--config', default='config.yaml',
                        help='Path to YAML configuration file (default: config.yaml)')
    
    # Override options for key parameters
    parser.add_argument('--snap-tolerance', type=float,
                        help='Snapping tolerance in meters (overrides config)')
    parser.add_argument('--min-length', type=float,
                        help='Minimum edge length in meters (overrides config)')
    parser.add_argument('--precision', type=int,
                        help='Coordinate precision in decimal places (overrides config)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching and force re-extraction')
    parser.add_argument('--enable-json-gz-format', action='store_true',
                        help='Enable JSON GZ format output (two gzipped JSON files) for compatibility (overrides config)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if os.path.exists(args.config):
            config = Config.from_yaml(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = Config()
            logger.info("Using default configuration (config.yaml not found)")
        
        # Apply command line overrides
        if args.snap_tolerance is not None:
            config.snap_tolerance_m = args.snap_tolerance
        if args.min_length is not None:
            config.min_fragment_length_m = args.min_length
        if args.precision is not None:
            config.coordinate_precision = args.precision
        if args.no_cache:
            config.enable_parameter_based_caching = False
            config.reuse_extraction = False
        
        logger.info(f"Configuration: snap_tolerance={config.snap_tolerance_m}m, "
                   f"min_length={config.min_fragment_length_m}m, "
                   f"precision={config.coordinate_precision}")
        
        # Step 1: Extract waterways from PBF or create test data
        if args.input_file.lower() == "test":
            logger.info("Using synthetic test data")
            waterways = create_test_waterways()
        else:
            cache_file = get_cache_filename(args.input_file, config) if config.enable_parameter_based_caching else None
            waterways = extract_waterways(args.input_file, config, cache_file)
        
        if not waterways:
            logger.warning("No waterways found in the input file")
            return
        
        # Step 2: Build graph using modern architecture
        graph_builder = ModernWaterwayGraphBuilder(config)
        nodes, edges = graph_builder.build_graph(waterways, args.input_file)
        
        # Step 3: Save outputs in multiple formats
        base_filename = get_output_base_filename(args.input_file)
        output_manager = OutputManager(config)
        
        file_sizes = output_manager.save_outputs(nodes, edges, base_filename, 
                                                graph_builder.qa_metrics, graph_builder.id_generator)
        
        # Step 4: Generate manifest (only if enabled)
        if config.generate_manifest:
            manifest = ManifestGenerator.generate_manifest(args.input_file, config, 
                                                          graph_builder.qa_metrics, file_sizes)
            manifest_file = f"{base_filename}.manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            file_sizes[manifest_file] = os.path.getsize(manifest_file)
        
        # Print summary
        print("\n" + "="*60)
        print("WATERWAY EXTRACTION SUMMARY v2.1")
        print("="*60)
        print(f"Input file: {args.input_file}")
        print(f"Original waterways: {len(waterways):,}")
        print(f"Final nodes: {len(nodes):,}")
        print(f"Final edges: {len(edges):,}")
        print(f"Processing time: {graph_builder.qa_metrics.get('processing_time_seconds', 0):.2f}s")
        print(f"\nConfiguration:")
        print(f"  Snap tolerance: {config.snap_tolerance_m}m")
        print(f"  Min edge length: {config.min_fragment_length_m}m")
        print(f"  Coordinate precision: {config.coordinate_precision} decimal places")
        print(f"  Distance calculation: {config.distance_calculation_method}")
        
        print(f"\nQuality Metrics:")
        print(f"  Clusters formed: {graph_builder.qa_metrics.get('total_clusters', 0)}")
        print(f"  Width parse success: {graph_builder.qa_metrics.get('width_parse_success_rate', 0):.1f}%")
        print(f"  Mean edge length: {graph_builder.qa_metrics.get('mean_edge_length_m', 0):.1f}m")
        
        print(f"\nOutput files:")
        for filename, size in file_sizes.items():
            print(f"  {filename} ({size:,} bytes)")
        
        total_size = sum(file_sizes.values())
        print(f"  Total size: {total_size:,} bytes")
        print("="*60)
        print("© OpenStreetMap contributors. Data licensed under ODbL.")
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set
        pass
    
    main()
