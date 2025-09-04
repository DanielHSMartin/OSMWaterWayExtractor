#!/usr/bin/env python3
"""
Test script to reproduce the waterway route calculation bug
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from waterway_route_calculator import WaterwayRouteCalculator

def test_route_bug():
    """Test the specific case that shows the bug"""
    
    # Create the calculator with the Brazil data
    calculator = WaterwayRouteCalculator('brazil-latest.nodes.json.gz', 'brazil-latest.edges.json.gz')
    
    # Use the coordinates from the problem statement as strings (like command line input)
    from waterway_route_calculator import parse_coordinate_pairs
    waypoints = parse_coordinate_pairs([
        "-14.52107623303884,-49.5431544874199",
        "-14.42317971535585,-49.56648632971223"
    ])
    
    print("Testing route calculation with coordinates from bug report...")
    print(f"Waypoint 1: {waypoints[0]}")
    print(f"Waypoint 2: {waypoints[1]}")
    print()
    
    # Calculate the route
    route_result = calculator.calculate_route(waypoints)
    
    print(f"Route calculated with {len(route_result['segments'])} segments")
    print(f"Total distance: {route_result['total_distance_m']:.2f}m")
    
    # Print details of each segment to analyze the issue
    for i, segment in enumerate(route_result['segments']):
        print(f"Segment {i+1}: {len(segment['geometry'])} points, {segment['distance_m']:.2f}m")
        if len(segment['geometry']) >= 2:
            start_coord = segment['geometry'][0]
            end_coord = segment['geometry'][-1]
            print(f"  Start: ({start_coord[0]:.6f}, {start_coord[1]:.6f})")
            print(f"  End: ({end_coord[0]:.6f}, {end_coord[1]:.6f})")

if __name__ == "__main__":
    test_route_bug()