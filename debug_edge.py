#!/usr/bin/env python3
"""
Debug script to examine the specific edge that's causing issues
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from waterway_route_calculator import WaterwayRouteCalculator, Point

def debug_edge_geometry():
    """Debug the specific edge geometry"""
    
    # Create the calculator
    calculator = WaterwayRouteCalculator('brazil-latest.nodes.json.gz', 'brazil-latest.edges.json.gz')
    
    # Get the end point and edge
    end_point = Point(-14.42317971535585, -49.56648632971223)
    end_edge, end_point_on_edge, end_distance = calculator.graph.find_closest_edge(end_point)
    
    print("Debug Analysis of Edge Geometry")
    print("=" * 40)
    print(f"End edge: {end_edge.id}")
    print(f"Start node: {end_edge.start_node}")
    print(f"End node: {end_edge.end_node}")
    print(f"Total edge length: {end_edge.length:.2f}m")
    print(f"Number of coordinates: {len(end_edge.coordinates)}")
    print()
    
    # Get node coordinates
    start_node_coord = calculator.graph.nodes[end_edge.start_node]
    end_node_coord = calculator.graph.nodes[end_edge.end_node]
    
    print(f"Start node {end_edge.start_node}: ({start_node_coord.lat:.6f}, {start_node_coord.lon:.6f})")
    print(f"End node {end_edge.end_node}: ({end_node_coord.lat:.6f}, {end_node_coord.lon:.6f})")
    print(f"End point: ({end_point.lat:.6f}, {end_point.lon:.6f})")
    print()
    
    # Check if the edge coordinates match the node coordinates
    first_coord = end_edge.coordinates[0]
    last_coord = end_edge.coordinates[-1]
    
    print(f"First edge coordinate: ({first_coord[0]:.6f}, {first_coord[1]:.6f})")
    print(f"Last edge coordinate: ({last_coord[0]:.6f}, {last_coord[1]:.6f})")
    print()
    
    # Check which node corresponds to which end of the coordinates
    dist_start_to_first = start_node_coord.distance_to(Point(first_coord[0], first_coord[1]))
    dist_start_to_last = start_node_coord.distance_to(Point(last_coord[0], last_coord[1]))
    dist_end_to_first = end_node_coord.distance_to(Point(first_coord[0], first_coord[1]))
    dist_end_to_last = end_node_coord.distance_to(Point(last_coord[0], last_coord[1]))
    
    print(f"Distance from start node to first coordinate: {dist_start_to_first:.2f}m")
    print(f"Distance from start node to last coordinate: {dist_start_to_last:.2f}m")
    print(f"Distance from end node to first coordinate: {dist_end_to_first:.2f}m")
    print(f"Distance from end node to last coordinate: {dist_end_to_last:.2f}m")
    print()
    
    # Determine the correct mapping
    if dist_start_to_first < dist_start_to_last:
        print("Start node corresponds to first coordinate")
        print("End node corresponds to last coordinate")
        coordinates_direction = "start->end"
    else:
        print("Start node corresponds to last coordinate")
        print("End node corresponds to first coordinate")
        coordinates_direction = "end->start"
    
    print(f"Coordinates direction: {coordinates_direction}")
    print()
    
    # Get closest point information
    closest_point, _, seg_idx, ratio = end_edge.get_closest_point_with_position(end_point)
    print(f"End point closest to segment {seg_idx} at ratio {ratio:.3f}")
    print(f"Closest point: ({closest_point.lat:.6f}, {closest_point.lon:.6f})")
    print()
    
    # Calculate distances to both nodes via edge more carefully
    print("Calculating precise distances via edge...")
    
    # Distance to start node via edge coordinates
    start_distance_via_coords = 0.0
    for i in range(seg_idx, 0, -1):
        curr = Point(end_edge.coordinates[i][0], end_edge.coordinates[i][1])
        prev = Point(end_edge.coordinates[i-1][0], end_edge.coordinates[i-1][1])
        start_distance_via_coords += curr.distance_to(prev)
    
    # Add distance from closest point to current segment
    if seg_idx > 0:
        segment_point = Point(end_edge.coordinates[seg_idx][0], end_edge.coordinates[seg_idx][1])
        start_distance_via_coords += closest_point.distance_to(segment_point)
    
    # Distance to end node via edge coordinates  
    end_distance_via_coords = 0.0
    for i in range(seg_idx, len(end_edge.coordinates) - 1):
        curr = Point(end_edge.coordinates[i][0], end_edge.coordinates[i][1])
        next_pt = Point(end_edge.coordinates[i+1][0], end_edge.coordinates[i+1][1])
        end_distance_via_coords += curr.distance_to(next_pt)
    
    # Add distance from closest point to next segment
    if seg_idx + 1 < len(end_edge.coordinates):
        segment_point = Point(end_edge.coordinates[seg_idx+1][0], end_edge.coordinates[seg_idx+1][1])
        end_distance_via_coords += closest_point.distance_to(segment_point)
    
    print(f"Distance to start node via coordinates: {start_distance_via_coords:.2f}m")
    print(f"Distance to end node via coordinates: {end_distance_via_coords:.2f}m")
    print()
    
    # Test both directions and see which makes more sense
    straight_to_start = end_point.distance_to(start_node_coord)
    straight_to_end = end_point.distance_to(end_node_coord)
    
    print(f"Straight-line distance to start node: {straight_to_start:.2f}m")
    print(f"Straight-line distance to end node: {straight_to_end:.2f}m")
    print()
    
    ratio_start = start_distance_via_coords / straight_to_start if straight_to_start > 0 else float('inf')
    ratio_end = end_distance_via_coords / straight_to_end if straight_to_end > 0 else float('inf')
    
    print(f"Via-edge to straight ratio for start node: {ratio_start:.2f}")
    print(f"Via-edge to straight ratio for end node: {ratio_end:.2f}")
    print()
    
    if ratio_end < ratio_start:
        print("End node is more efficient (lower ratio)")
    else:
        print("Start node is more efficient (lower ratio)")

if __name__ == "__main__":
    debug_edge_geometry()