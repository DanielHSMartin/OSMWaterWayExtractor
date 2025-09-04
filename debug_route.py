#!/usr/bin/env python3
"""
Debug script to analyze the route calculation issue in detail
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from waterway_route_calculator import WaterwayRouteCalculator, Point

def debug_route_issue():
    """Debug the specific route calculation issue"""
    
    # Create the calculator
    calculator = WaterwayRouteCalculator('brazil-latest.nodes.json.gz', 'brazil-latest.edges.json.gz')
    
    # Create the waypoints from the problem statement
    start_point = Point(-14.52107623303884, -49.5431544874199)
    end_point = Point(-14.42317971535585, -49.56648632971223)
    
    print("Debug Analysis of Route Calculation Issue")
    print("=" * 50)
    print(f"Start point: ({start_point.lat:.6f}, {start_point.lon:.6f})")
    print(f"End point: ({end_point.lat:.6f}, {end_point.lon:.6f})")
    print()
    
    # Find the closest edges
    print("Finding closest edges...")
    start_edge, start_point_on_edge, start_distance = calculator.graph.find_closest_edge(start_point)
    end_edge, end_point_on_edge, end_distance = calculator.graph.find_closest_edge(end_point)
    
    print(f"Start edge: {start_edge.id}")
    print(f"  Start node: {start_edge.start_node}, End node: {start_edge.end_node}")
    print(f"  Edge length: {start_edge.length:.2f}m")
    print(f"  Distance to edge: {start_distance:.2f}m")
    print(f"  Closest point on edge: ({start_point_on_edge.lat:.6f}, {start_point_on_edge.lon:.6f})")
    print()
    
    print(f"End edge: {end_edge.id}")
    print(f"  Start node: {end_edge.start_node}, End node: {end_edge.end_node}")
    print(f"  Edge length: {end_edge.length:.2f}m")
    print(f"  Distance to edge: {end_distance:.2f}m")
    print(f"  Closest point on edge: ({end_point_on_edge.lat:.6f}, {end_point_on_edge.lon:.6f})")
    print()
    
    # Check which nodes are selected as "nearest"
    print("Checking nearest node selections...")
    start_nearest_node, start_distance_to_node = start_edge.get_nearest_node_via_edge(start_point)
    end_nearest_node, end_distance_to_node = end_edge.get_nearest_node_via_edge(end_point)
    
    print(f"Start point -> nearest node: {start_nearest_node} (distance: {start_distance_to_node:.2f}m)")
    print(f"End point -> nearest node: {end_nearest_node} (distance: {end_distance_to_node:.2f}m)")
    print()
    
    # Check the coordinates of these nodes
    start_node_coord = calculator.graph.nodes[start_nearest_node]
    end_node_coord = calculator.graph.nodes[end_nearest_node]
    
    print(f"Start nearest node {start_nearest_node} coordinates: ({start_node_coord.lat:.6f}, {start_node_coord.lon:.6f})")
    print(f"End nearest node {end_nearest_node} coordinates: ({end_node_coord.lat:.6f}, {end_node_coord.lon:.6f})")
    print()
    
    # Check both nodes on the end edge
    end_start_node_coord = calculator.graph.nodes[end_edge.start_node]
    end_end_node_coord = calculator.graph.nodes[end_edge.end_node]
    
    print(f"End edge start node {end_edge.start_node}: ({end_start_node_coord.lat:.6f}, {end_start_node_coord.lon:.6f})")
    print(f"End edge end node {end_edge.end_node}: ({end_end_node_coord.lat:.6f}, {end_end_node_coord.lon:.6f})")
    print()
    
    # Calculate distances to both end nodes via edge
    print("Calculating distances to both end edge nodes...")
    
    # Get positions for end point on edge
    closest_point, _, seg_idx, ratio = end_edge.get_closest_point_with_position(end_point)
    print(f"End point closest to edge at segment {seg_idx}, ratio {ratio:.3f}")
    print(f"Closest point: ({closest_point.lat:.6f}, {closest_point.lon:.6f})")
    print()
    
    # Calculate distance to start node via edge manually
    distance_to_start_via_edge = 0.0
    for i in range(seg_idx, 0, -1):
        curr_coord = end_edge.coordinates[i]
        prev_coord = end_edge.coordinates[i - 1]
        distance_to_start_via_edge += ((curr_coord[0] - prev_coord[0])**2 + (curr_coord[1] - prev_coord[1])**2)**0.5 * 111000  # rough conversion
    
    # Add distance from closest point to segment start
    if seg_idx > 0:
        segment_start = Point(end_edge.coordinates[seg_idx][0], end_edge.coordinates[seg_idx][1])
        distance_to_start_via_edge += closest_point.distance_to(segment_start)
    
    # Calculate distance to end node via edge manually
    distance_to_end_via_edge = 0.0
    for i in range(seg_idx, len(end_edge.coordinates) - 1):
        curr_coord = end_edge.coordinates[i]
        next_coord = end_edge.coordinates[i + 1]
        distance_to_end_via_edge += ((curr_coord[0] - next_coord[0])**2 + (curr_coord[1] - next_coord[1])**2)**0.5 * 111000  # rough conversion
    
    # Add distance from closest point to segment end
    if seg_idx + 1 < len(end_edge.coordinates):
        segment_end = Point(end_edge.coordinates[seg_idx + 1][0], end_edge.coordinates[seg_idx + 1][1])
        distance_to_end_via_edge += closest_point.distance_to(segment_end)
    
    print(f"Distance to start node ({end_edge.start_node}) via edge: {distance_to_start_via_edge:.2f}m")
    print(f"Distance to end node ({end_edge.end_node}) via edge: {distance_to_end_via_edge:.2f}m")
    print(f"Selected node: {end_nearest_node} with distance {end_distance_to_node:.2f}m")
    print()
    
    # Test the geometry from node to point
    print("Testing geometry from node to end point...")
    try:
        geometry, distance = end_edge.get_geometry_from_node(end_nearest_node, end_point)
        print(f"Geometry from node {end_nearest_node} to end point:")
        print(f"  Points: {len(geometry)}")
        print(f"  Distance: {distance:.2f}m")
        if len(geometry) >= 2:
            print(f"  First point: ({geometry[0][0]:.6f}, {geometry[0][1]:.6f})")
            print(f"  Last point: ({geometry[-1][0]:.6f}, {geometry[-1][1]:.6f})")
    except Exception as e:
        print(f"Error getting geometry: {e}")
    
    print()
    
    # Check straight-line distance for comparison
    straight_distance = end_node_coord.distance_to(end_point)
    print(f"Straight-line distance from selected node to end point: {straight_distance:.2f}m")
    print(f"Via-edge distance: {end_distance_to_node:.2f}m")
    print(f"Ratio (via-edge / straight): {end_distance_to_node / straight_distance:.2f}")

if __name__ == "__main__":
    debug_route_issue()