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
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any, Set
import logging
import concurrent.futures
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
    waterway_types: List[str] = None
    
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
    server_formats: List[str] = None
    mobile_formats: List[str] = None
    mobile_max_chunk_size_mb: int = 10
    compression: bool = True
    include_geodesic_distances: bool = True
    
    # QA parameters
    enable_comprehensive_metrics: bool = True
    distance_validation_samples: int = 1000
    generate_debug_outputs: bool = False
    qa_thresholds: Dict[str, float] = None
    
    # Caching parameters
    enable_parameter_based_caching: bool = True
    cache_directory: str = "./intermediate"
    reuse_extraction: bool = True
    
    def __post_init__(self):
        if self.waterway_types is None:
            self.waterway_types = ["river", "canal"]
        if self.server_formats is None:
            self.server_formats = ["parquet", "csv", "geojson"]
        if self.mobile_formats is None:
            self.mobile_formats = ["csv"]
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
        
        # Flatten nested dictionary structure
        config_dict = {}
        for section, values in data.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    config_dict[key] = value
            else:
                config_dict[section] = values
        
        return cls(**config_dict)
    
    def get_parameter_hash(self) -> str:
        """Generate a hash of configuration parameters for caching."""
        # Create a deterministic string representation of config
        config_str = json.dumps(asdict(self), sort_keys=True)
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
        """Approximate distance calculation (legacy fallback)."""
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
        
        total_length = 0.0
        for i in range(1, len(coords)):
            total_length += self.distance(coords[i-1], coords[i])
        
        return total_length


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
        
        # Filter candidates using accurate geodesic distance
        nearby_coords = []
        for candidate_id in candidate_ids:
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
        edges_added = 0
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
        # Use osmium's WKB factory to get geometries
        self.wkb_factory = osmium.geom.WKBFactory()
        
    def way(self, w):
        """Extract ways that match configured waterway types."""
        tags = {tag.k: tag.v for tag in w.tags}
        
        # Only process waterways that match configuration
        if tags.get('waterway') in self.config.waterway_types:
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
                        
            except Exception as e:
                # Skip ways that can't be processed (e.g., incomplete geometry)
                logger.debug(f"Skipping way {w.id}: {e}")




class ModernWaterwayGraphBuilder:
    """Modern waterway graph builder implementing Specification v2.1 features."""
    
    def __init__(self, config: Config):
        self.config = config
        self.geod_calc = GeodCalculator(config.distance_calculation_method)
        self.id_generator = IDGenerator(config)
        self.clusterer = SnappingClusterer(config, self.geod_calc)
        self.qa_metrics = {}
        
    def build_graph(self, waterways: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Build a waterway graph from extracted waterway data."""
        logger.info(f"Building graph from {len(waterways)} waterways using v2.1 specification...")
        
        start_time = time.time()
        
        # Step 1: Process and clean waterway coordinates
        logger.info("Step 1: Processing and cleaning waterway coordinates...")
        processed_waterways = self._process_waterways(waterways)
        
        # Step 2: Extract all unique endpoints and junctions
        logger.info("Step 2: Extracting endpoints and identifying junctions...")
        endpoints, junctions = self._extract_endpoints_and_junctions(processed_waterways)
        
        # Step 3: Snap and cluster endpoints using union-find
        logger.info("Step 3: Snapping and clustering endpoints...")
        coord_mapping = self.clusterer.cluster_endpoints(endpoints + junctions)
        
        # Step 4: Create edges with accurate distance calculations
        logger.info("Step 4: Creating edges with geodesic distances...")
        edges = self._create_edges(processed_waterways, coord_mapping)
        
        # Step 5: Build final node list with deterministic IDs
        logger.info("Step 5: Building node list with deterministic IDs...")
        nodes = self._build_nodes(coord_mapping)
        
        # Step 6: Generate QA metrics
        logger.info("Step 6: Generating QA metrics...")
        self._generate_qa_metrics(waterways, processed_waterways, nodes, edges, time.time() - start_time)
        
        logger.info(f"Graph built: {len(nodes)} nodes, {len(edges)} edges in {time.time() - start_time:.2f}s")
        
        return nodes, edges
    
    def _process_waterways(self, waterways: List[Dict]) -> List[Dict]:
        """Process waterway coordinates and apply coordinate precision."""
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
    
    def _extract_endpoints_and_junctions(self, waterways: List[Dict]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Extract unique endpoints and identify junction points."""
        endpoint_count = Counter()
        all_endpoints = []
        
        # Count how many times each coordinate appears as an endpoint
        for waterway in waterways:
            coords = waterway['coordinates']
            start_coord = coords[0]
            end_coord = coords[-1]
            
            endpoint_count[start_coord] += 1
            endpoint_count[end_coord] += 1
            all_endpoints.extend([start_coord, end_coord])
        
        # Coordinates that appear more than once are junctions
        junctions = [coord for coord, count in endpoint_count.items() if count > 1]
        endpoints = list(set(all_endpoints))
        
        logger.info(f"Found {len(endpoints)} unique endpoints, {len(junctions)} junctions")
        return endpoints, junctions
    
    def _create_edges(self, waterways: List[Dict], coord_mapping: Dict) -> List[Dict]:
        """Create edges with accurate geodesic distances and deterministic IDs."""
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
    handler.apply_file(pbf_file)
    
    logger.info(f"Extracted {len(handler.waterways)} waterways")
    
    # Save to cache if requested
    if cache_file:
        logger.info(f"Saving waterways to cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with gzip.open(cache_file, 'wt', encoding='utf-8') as f:
            json.dump(handler.waterways, f)
    
    return handler.waterways
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