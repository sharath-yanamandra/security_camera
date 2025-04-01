#!/usr/bin/env python3
"""
Main module for the Security Camera Monitoring System
Integrates all use cases:
1. Entry Monitoring (stairs, gatherings, vehicles)
2. Reception Monitoring (occupancy, gate jumping)
3. Datacenter Entry Monitoring (unauthorized access, emergency buttons)
4. Datacenter Zone Monitoring (zone transitions)
5. Parking Monitoring (illegal parking, exit blocking)
"""

import os
import sys
import time
import argparse
from pathlib import Path
import json
import concurrent.futures
import multiprocessing

# Import all use cases
from entry_monitoring import EntryMonitor
from reception_monitoring import ReceptionMonitor
from datacenter_entry_monitoring import DatacenterEntryMonitor
from datacenter_zone_monitoring import DatacenterZoneMonitor
from parking_monitoring import ParkingMonitor

# Import utils for configuration
from utils import configure_models, load_rules


class SecurityMonitoringSystem:
    """Main system integrating all security camera monitoring use cases"""
    
    def __init__(self, config_file="config.json", rules_file="rules.json", zones_file="zones.json"):
        """
        Initialize the security monitoring system
        
        Args:
            config_file: Path to configuration file
            rules_file: Path to rules file
            zones_file: Path to zones file
        """
        # Store file paths
        self.config_file = config_file
        self.rules_file = rules_file
        self.zones_file = zones_file
        
        # Load configuration
        self.config = configure_models(config_file)
        self.rules = load_rules(rules_file)
        
        # Create output directory
        self.output_dir = Path(self.config.get("output_dir", "outputs"))
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Map use cases to their monitors
        self.use_cases = {
            1: {
                'name': 'Entry Monitoring',
                'camera_id': 'entry',
                'monitor_class': EntryMonitor,
                'monitor': None
            },
            2: {
                'name': 'Reception Monitoring',
                'camera_id': 'reception',
                'monitor_class': ReceptionMonitor,
                'monitor': None
            },
            3: {
                'name': 'Datacenter Entry Monitoring',
                'camera_id': 'datacenter_entry',
                'monitor_class': DatacenterEntryMonitor,
                'monitor': None
            },
            4: {
                'name': 'Datacenter Zone Monitoring',
                'camera_id': 'datacenter_inside',
                'monitor_class': DatacenterZoneMonitor,
                'monitor': None
            },
            5: {
                'name': 'Parking Monitoring',
                'camera_id': 'parking',
                'monitor_class': ParkingMonitor,
                'monitor': None
            }
        }
        
        print(f"Security Monitoring System initialized with {len(self.use_cases)} use cases")
    
    def initialize_monitor(self, use_case_id):
        """
        Initialize a specific monitor for a use case
        
        Args:
            use_case_id: ID of use case to initialize
            
        Returns:
            Initialized monitor instance
        """
        if use_case_id not in self.use_cases:
            print(f"Error: Use case {use_case_id} not implemented")
            return None
        
        use_case = self.use_cases[use_case_id]
        camera_id = use_case['camera_id']
        
        # Create the monitor
        monitor = use_case['monitor_class'](
            config_file=self.config_file,
            rules_file=self.rules_file,
            zones_file=self.zones_file,
            camera_id=camera_id
        )
        
        # Store the monitor
        self.use_cases[use_case_id]['monitor'] = monitor
        
        return monitor
    
    def run_use_case(self, use_case_id, input_video=None, output_video=None, skip_frames=2):
        """
        Run a specific use case with the given input video
        
        Args:
            use_case_id: ID of use case to run
            input_video: Path to input video file (if None, use stream_url from config)
            output_video: Path to output video file (if None, generate one)
            skip_frames: Process every Nth frame for better performance
            
        Returns:
            True if successful, False otherwise
        """
        if use_case_id not in self.use_cases:
            print(f"Error: Use case {use_case_id} not implemented")
            return False
        
        use_case = self.use_cases[use_case_id]
        camera_id = use_case['camera_id']
        
        # Get monitor
        monitor = use_case.get('monitor')
        if monitor is None:
            monitor = self.initialize_monitor(use_case_id)
            if monitor is None:
                return False
        
        # Get input video
        if input_video is None:
            camera_config = self.config.get("cameras", {}).get(camera_id, {})
            input_video = camera_config.get("stream_url")
            if not input_video:
                print(f"Error: No input video specified for use case {use_case_id}")
                return False
        
        # Generate output video path if not provided
        if output_video is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_video = self.output_dir / f"usecase{use_case_id}_{timestamp}.mp4"
        
        # Run the monitor
        print(f"Running Use Case {use_case_id}: {use_case['name']}")
        print(f"  Input: {input_video}")
        print(f"  Output: {output_video}")
        
        try:
            frames_processed = monitor.process_video(input_video, output_video, skip_frames=skip_frames)
            print(f"Completed Use Case {use_case_id}: processed {frames_processed} frames")
            return True
        except Exception as e:
            print(f"Error running Use Case {use_case_id}: {e}")
            return False
    
    def run_all_use_cases(self, video_mapping=None, output_dir=None, parallel=False, skip_frames=2):
        """
        Run all specified use cases
        
        Args:
            video_mapping: Dictionary mapping use case ID to input video path
            output_dir: Directory for output videos
            parallel: Whether to run use cases in parallel
            skip_frames: Process every Nth frame
            
        Returns:
            Dictionary of results by use case ID
        """
        # Create output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        else:
            output_dir = self.output_dir
        
        # Create default video mapping if not provided
        if video_mapping is None:
            video_mapping = {}
            for use_case_id, use_case in self.use_cases.items():
                camera_id = use_case['camera_id']
                camera_config = self.config.get("cameras", {}).get(camera_id, {})
                stream_url = camera_config.get("stream_url")
                if stream_url:
                    video_mapping[use_case_id] = stream_url
        
        results = {}
        
        if parallel and len(video_mapping) > 1:
            # Run use cases in parallel
            print(f"Running {len(video_mapping)} use cases in parallel")
            
            # Define worker function for parallel execution
            def run_use_case_worker(use_case_id, input_video):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_video = output_dir / f"usecase{use_case_id}_{timestamp}.mp4"
                result = self.run_use_case(use_case_id, input_video, output_video, skip_frames)
                return use_case_id, result
            
            # Run in parallel using ProcessPoolExecutor
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(video_mapping), multiprocessing.cpu_count())) as executor:
                futures = []
                for use_case_id, input_video in video_mapping.items():
                    future = executor.submit(run_use_case_worker, use_case_id, input_video)
                    futures.append(future)
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    use_case_id, result = future.result()
                    results[use_case_id] = result
        else:
            # Run use cases sequentially
            print(f"Running {len(video_mapping)} use cases sequentially")
            for use_case_id, input_video in video_mapping.items():
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_video = output_dir / f"usecase{use_case_id}_{timestamp}.mp4"
                results[use_case_id] = self.run_use_case(use_case_id, input_video, output_video, skip_frames)
        
        return results


def main():
    """Main entry point for the security monitoring system"""
    parser = argparse.ArgumentParser(description="Security Camera Monitoring System")
    
    # Global arguments
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--rules", type=str, default="rules.json", help="Path to rules file")
    parser.add_argument("--zones", type=str, default="zones.json", help="Path to zones file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for output videos")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--parallel", action="store_true", help="Run use cases in parallel")
    
    # Use case selection arguments
    parser.add_argument("--all", action="store_true", help="Run all use cases")
    parser.add_argument("--use-case", type=int, choices=range(1, 6), 
                        help="Run a specific use case (1-5)")
    
    # Input video arguments
    parser.add_argument("--input1", type=str, help="Input video for Use Case 1 (Entry)")
    parser.add_argument("--input2", type=str, help="Input video for Use Case 2 (Reception)")
    parser.add_argument("--input3", type=str, help="Input video for Use Case 3 (Datacenter Entry)")
    parser.add_argument("--input4", type=str, help="Input video for Use Case 4 (Datacenter Inside)")
    parser.add_argument("--input5", type=str, help="Input video for Use Case 5 (Parking)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the system
    system = SecurityMonitoringSystem(
        config_file=args.config,
        rules_file=args.rules,
        zones_file=args.zones
    )
    
    # Create video mapping from arguments
    video_mapping = {}
    if args.input1:
        video_mapping[1] = args.input1
    if args.input2:
        video_mapping[2] = args.input2
    if args.input3:
        video_mapping[3] = args.input3
    if args.input4:
        video_mapping[4] = args.input4
    if args.input5:
        video_mapping[5] = args.input5
    
    # Run selected use cases
    if args.use_case:
        # Run a specific use case
        if args.use_case in video_mapping:
            output_video = Path(args.output_dir) / f"usecase{args.use_case}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            system.run_use_case(args.use_case, video_mapping[args.use_case], output_video, args.skip_frames)
        else:
            print(f"Error: No input video specified for Use Case {args.use_case}")
            print(f"Please use --input{args.use_case} to specify an input video")
    elif args.all or video_mapping:
        # Run all use cases or those with input videos specified
        system.run_all_use_cases(
            video_mapping=video_mapping if video_mapping else None,
            output_dir=args.output_dir,
            parallel=args.parallel,
            skip_frames=args.skip_frames
        )
    else:
        print("No use cases selected to run.")
        print("Use --use-case X to run a specific use case, or --all to run all use cases.")
        print("For each use case, you need to specify an input video using --inputX arguments.")


if __name__ == "__main__":
    main()