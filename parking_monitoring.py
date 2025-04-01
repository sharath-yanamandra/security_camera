"""
Use Case 5: Parking Monitoring
- Detection of vehicles parked outside designated zones
- Detection of vehicles blocking exit paths
- Detection of suspicious activities in parking areas
"""

import cv2
import numpy as np
import time
import json
import os
from pathlib import Path
from datetime import datetime

from utils import (
    VideoProcessor, ObjectTracker, ModelLoader, ZoneManager,
    format_duration, draw_text_with_background, draw_zone,
    is_object_stationary, configure_models, load_rules,
    create_event_metadata, VEHICLE_CLASSES
)


class ParkingMonitor(VideoProcessor):
    """
    Monitor for parking area events:
    - Vehicles parked in unauthorized areas
    - Vehicles blocking exit paths
    - Suspicious activities in parking areas
    """
    
    def __init__(self, config_file="config.json", rules_file="rules.json", zones_file="zones.json", camera_id="parking"):
        """
        Initialize the Parking monitor
        
        Args:
            config_file: Path to configuration file
            rules_file: Path to rules file
            zones_file: Path to zones file
            camera_id: Camera identifier used in configuration
        """
        # Load configuration
        self.config = configure_models(config_file)
        self.rules = load_rules(rules_file)
        self.camera_id = camera_id
        
        # Initialize base class
        super().__init__(self.config.get("output_dir", "outputs"))
        
        # Get camera-specific settings
        self.camera_config = self.config.get("cameras", {}).get(camera_id, {})
        self.camera_rules = self.rules.get(camera_id, self.rules.get("default", {}))
        
        # Load detection model
        detection_model = self.config.get("models", {}).get("detection", "yolov8l.pt")
        device = self.config.get("device", "cpu")
        confidence = self.config.get("confidence", 0.5)
        
        self.detection_model = ModelLoader.load_model(detection_model, device)
        self.detection_model.conf = confidence
        
        # Load zones
        self.zone_manager = ZoneManager(zones_file)
        
        # Define zone colors
        self.zone_colors = {
            "parking_zones": (0, 255, 0),      # Green
            "exit_zones": (0, 0, 255),         # Red
            "no_parking": (255, 0, 0)          # Red
        }
        
        # Create object tracker
        processing_config = self.config.get("processing", {})
        self.vehicle_tracker = ObjectTracker(
            max_disappeared=processing_config.get("max_disappeared", 30),
            max_distance=processing_config.get("tracking_distance", 50)
        )
        
        # Initialize detection state
        self.violations = {
            "illegal_parking": set(),  # Set of vehicle IDs parked in unauthorized zones
            "exit_blocking": set(),    # Set of vehicle IDs blocking exits
            "suspicious": set()        # Set of vehicle IDs with suspicious activity
        }
        
        # Get rule parameters
        rule_params = self.camera_rules.get("improper_parking", {}).get("parameters", {})
        self.min_stationary_time = rule_params.get("min_stationary_time", 30)
        self.movement_threshold = rule_params.get("movement_threshold", 5)
        
        rule_params = self.camera_rules.get("exit_blocking", {}).get("parameters", {})
        self.min_overlap_ratio = rule_params.get("min_overlap_ratio", 0.3)
        
        rule_params = self.camera_rules.get("suspicious_activity", {}).get("parameters", {})
        self.rapid_movement_threshold = rule_params.get("rapid_movement_threshold", 50)
        self.loitering_time = rule_params.get("loitering_time", 120)
        
        # Session statistics
        self.stats = {
            "illegal_parking_violations": 0,
            "exit_blocking_violations": 0,
            "suspicious_activity_count": 0,
            "total_vehicles_detected": 0
        }
        
        # Initialize alert state
        self.alerts = {
            "illegal_parking": {
                "active": False,
                "start_time": None,
                "message": "ALERT: Vehicle parked in unauthorized area!",
                "duration": 5.0
            },
            "exit_blocking": {
                "active": False,
                "start_time": None,
                "message": "ALERT: Vehicle blocking exit path!",
                "duration": 5.0
            },
            "suspicious_activity": {
                "active": False,
                "start_time": None,
                "message": "ALERT: Suspicious activity detected!",
                "duration": 5.0
            }
        }
        
        self.start_time = time.time()
        
        print(f"Parking monitoring initialized for camera: {camera_id}")
    
    def _get_zones_from_config(self):
        """Extract parking zones from configuration"""
        zones = self.zone_manager.zones.get(self.camera_id, {})
        
        parking_zones = []
        if "parking_zones" in zones:
            for zone in zones["parking_zones"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Parking Zone")
                    parking_zones.append((*coords, name))
        
        exit_zones = []
        if "exit_zones" in zones:
            for zone in zones["exit_zones"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Exit Path")
                    exit_zones.append((*coords, name))
        
        no_parking_zones = []
        if "no_parking" in zones:
            for zone in zones["no_parking"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "No Parking")
                    no_parking_zones.append((*coords, name))
        
        return parking_zones, exit_zones, no_parking_zones
    
    def detect_vehicles(self, frame, detection_results):
        """
        Detect and track vehicles in the parking area
        Check for parking violations and exit blocking
        
        Args:
            frame: Current video frame
            detection_results: Results from detection model
            
        Returns:
            Tuple of (frame, vehicle_count, violations)
        """
        # Get zones
        parking_zones, exit_zones, no_parking_zones = self._get_zones_from_config()
        
        # Process detection results to find vehicles
        vehicles = []
        
        if detection_results:
            for r in detection_results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0].item())
                    
                    # Only interested in vehicle classes
                    if cls not in VEHICLE_CLASSES:
                        continue
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox = (x1, y1, x2, y2)
                    confidence = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    
                    # Get vehicle type and center point
                    vehicle_type = VEHICLE_CLASSES[cls]
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Add to vehicles list
                    vehicles.append({
                        "bbox": bbox,
                        "center": center,
                        "confidence": confidence,
                        "type": vehicle_type,
                        "class_id": cls
                    })
        
        # Update vehicle tracker
        vehicle_detections = [(v["bbox"], v["type"], v["confidence"]) for v in vehicles]
        vehicle_objects = self.vehicle_tracker.update(vehicle_detections)
        
        # Update total vehicles count
        self.stats["total_vehicles_detected"] = len(vehicle_objects)
        
        # Clear previous violations
        current_illegal_parking = set()
        current_exit_blocking = set()
        current_suspicious = set()
        
        # Check each vehicle for violations
        for vehicle_id, vehicle_data in vehicle_objects.items():
            bbox = vehicle_data["bbox"]
            vehicle_type = vehicle_data["type"]
            
            # Check if vehicle is stationary (parked)
            is_stationary, stationary_duration = is_object_stationary(
                vehicle_data, 
                time_threshold=self.min_stationary_time,
                movement_threshold=self.movement_threshold
            )
            
            if is_stationary:
                # Check if in proper parking zone
                in_parking_zone = False
                for zone in parking_zones:
                    overlap = self.zone_manager.object_zone_overlap(bbox, zone)
                    if overlap >= self.min_overlap_ratio:
                        in_parking_zone = True
                        break
                
                # Check if in no-parking zone
                in_no_parking_zone = False
                for zone in no_parking_zones:
                    overlap = self.zone_manager.object_zone_overlap(bbox, zone)
                    if overlap >= self.min_overlap_ratio:
                        in_no_parking_zone = True
                        break
                
                # Check if blocking exit
                blocking_exit = False
                blocked_exit_name = None
                for zone in exit_zones:
                    overlap = self.zone_manager.object_zone_overlap(bbox, zone)
                    if overlap >= self.min_overlap_ratio:
                        blocking_exit = True
                        blocked_exit_name = zone[4] if len(zone) > 4 else "Exit"
                        break
                
                # Detect parking violations
                if (not in_parking_zone) or in_no_parking_zone:
                    # Illegal parking violation
                    current_illegal_parking.add(vehicle_id)
                    
                    # Check if this is a new violation
                    if vehicle_id not in self.violations["illegal_parking"]:
                        self.violations["illegal_parking"].add(vehicle_id)
                        self.stats["illegal_parking_violations"] += 1
                        
                        # Update alert state
                        if not self.alerts["illegal_parking"]["active"]:
                            self.alerts["illegal_parking"]["active"] = True
                            self.alerts["illegal_parking"]["start_time"] = time.time()
                            
                            # Save event screenshot
                            if self.config.get("events", {}).get("save_screenshots", True):
                                event_metadata = create_event_metadata(
                                    frame, "illegal_parking", vehicle_id, None, 
                                    vehicle_data["confidence"], {
                                        "vehicle_type": vehicle_type,
                                        "duration": stationary_duration,
                                        "in_no_parking": in_no_parking_zone
                                    }
                                )
                                self.save_screenshot(frame, "illegal_parking")
                
                # Detect exit blocking
                if blocking_exit:
                    # Exit blocking violation
                    current_exit_blocking.add(vehicle_id)
                    
                    # Check if this is a new violation
                    if vehicle_id not in self.violations["exit_blocking"]:
                        self.violations["exit_blocking"].add(vehicle_id)
                        self.stats["exit_blocking_violations"] += 1
                        
                        # Update alert state
                        if not self.alerts["exit_blocking"]["active"]:
                            self.alerts["exit_blocking"]["active"] = True
                            self.alerts["exit_blocking"]["start_time"] = time.time()
                            self.alerts["exit_blocking"]["message"] = f"ALERT: Vehicle blocking {blocked_exit_name}!"
                            
                            # Save event screenshot
                            if self.config.get("events", {}).get("save_screenshots", True):
                                event_metadata = create_event_metadata(
                                    frame, "exit_blocking", vehicle_id, blocked_exit_name, 
                                    vehicle_data["confidence"], {
                                        "vehicle_type": vehicle_type,
                                        "duration": stationary_duration
                                    }
                                )
                                self.save_screenshot(frame, "exit_blocking")
            else:
                # Check for suspicious activity (rapid movement)
                positions = vehicle_data.get("positions", [])
                if len(positions) >= 2:
                    # Calculate average velocity
                    total_distance = 0
                    for i in range(1, len(positions)):
                        prev_pos = positions[i-1]
                        curr_pos = positions[i]
                        distance = np.sqrt(
                            (curr_pos[0] - prev_pos[0])**2 + 
                            (curr_pos[1] - prev_pos[1])**2
                        )
                        total_distance += distance
                    
                    avg_velocity = total_distance / (len(positions) - 1)
                    
                    # Check if velocity is suspiciously high
                    if avg_velocity > self.rapid_movement_threshold:
                        current_suspicious.add(vehicle_id)
                        
                        # Check if this is a new detection
                        if vehicle_id not in self.violations["suspicious"]:
                            self.violations["suspicious"].add(vehicle_id)
                            self.stats["suspicious_activity_count"] += 1
                            
                            # Update alert state
                            if not self.alerts["suspicious_activity"]["active"]:
                                self.alerts["suspicious_activity"]["active"] = True
                                self.alerts["suspicious_activity"]["start_time"] = time.time()
                                
                                # Save event screenshot
                                if self.config.get("events", {}).get("save_screenshots", True):
                                    event_metadata = create_event_metadata(
                                        frame, "suspicious_activity", vehicle_id, None, 
                                        vehicle_data["confidence"], {
                                            "vehicle_type": vehicle_type,
                                            "velocity": avg_velocity
                                        }
                                    )
                                    self.save_screenshot(frame, "suspicious_activity")
            
            # Visualize vehicle with tracking info
            x1, y1, x2, y2 = bbox
            
            # Determine color based on violations
            if vehicle_id in current_exit_blocking:
                color = (0, 0, 255)  # Red for exit blocking
            elif vehicle_id in current_illegal_parking:
                color = (0, 165, 255)  # Orange for illegal parking
            elif vehicle_id in current_suspicious:
                color = (255, 0, 255)  # Magenta for suspicious activity
            else:
                color = (0, 255, 0)  # Green for compliant vehicles
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw vehicle ID and type
            status_text = f"ID:{vehicle_id} {vehicle_type}"
            if is_stationary:
                status_text += f" ({format_duration(stationary_duration)})"
            
            draw_text_with_background(
                frame, status_text, (x1, y1 - 10),
                color=color, bg_alpha=0.7
            )
            
            # Draw violation status if applicable
            if vehicle_id in current_illegal_parking:
                draw_text_with_background(
                    frame, "ILLEGAL PARKING", (x1, y2 + 15),
                    font_scale=0.5, color=color, bg_alpha=0.7
                )
            elif vehicle_id in current_exit_blocking:
                draw_text_with_background(
                    frame, "BLOCKING EXIT", (x1, y2 + 15),
                    font_scale=0.5, color=color, bg_alpha=0.7
                )
            elif vehicle_id in current_suspicious:
                draw_text_with_background(
                    frame, "SUSPICIOUS ACTIVITY", (x1, y2 + 15),
                    font_scale=0.5, color=color, bg_alpha=0.7
                )
        
        # Update violations tracking
        self.violations["illegal_parking"] = current_illegal_parking
        self.violations["exit_blocking"] = current_exit_blocking
        self.violations["suspicious"] = current_suspicious
        
        # Return summary
        all_violations = {
            "illegal_parking": len(current_illegal_parking),
            "exit_blocking": len(current_exit_blocking),
            "suspicious_activity": len(current_suspicious)
        }
        
        return frame, len(vehicle_objects), all_violations
    
    def process_frame(self, frame):
        """
        Process a video frame to detect events
        
        Args:
            frame: Video frame to process
            
        Returns:
            Tuple of (processed_frame, detection_results)
        """
        # Create a copy of the frame for visualization
        result_frame = frame.copy()
        
        # Run object detection
        detection_results = self.detection_model(frame)
        
        # Draw zones
        self._draw_zones(result_frame)
        
        # Detect vehicles and violations
        result_frame, vehicle_count, violations = self.detect_vehicles(
            result_frame, detection_results
        )
        
        # Draw status and alerts
        result_frame = self._draw_status_and_alerts(result_frame, violations)
        
        # Return processed frame and detection results
        detection_info = {
            "vehicle_count": vehicle_count,
            "violations": violations
        }
        
        return result_frame, detection_info
    
    def _draw_zones(self, frame):
        """Draw monitoring zones on the frame"""
        zones = self.zone_manager.zones.get(self.camera_id, {})
        
        # Draw parking zones
        if "parking_zones" in zones:
            for zone in zones["parking_zones"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Parking Zone")
                    draw_zone(frame, (*coords, name), self.zone_colors["parking_zones"], 0.3, name)
        
        # Draw exit zones
        if "exit_zones" in zones:
            for zone in zones["exit_zones"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Exit Path")
                    draw_zone(frame, (*coords, name), self.zone_colors["exit_zones"], 0.3, name)
        
        # Draw no-parking zones
        if "no_parking" in zones:
            for zone in zones["no_parking"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "No Parking")
                    draw_zone(frame, (*coords, name), self.zone_colors["no_parking"], 0.3, name)
        
        return frame
    
    def _draw_status_and_alerts(self, frame, violations):
        """
        Draw status information and alerts on the frame
        
        Args:
            frame: Video frame to draw on
            violations: Current violations count dictionary
            
        Returns:
            Frame with status and alerts drawn
        """
        h, w = frame.shape[:2]
        current_time = time.time()
        
        # Draw session time
        session_time = current_time - self.start_time
        time_str = format_duration(session_time)
        draw_text_with_background(
            frame, f"Session time: {time_str}", (10, 30),
            font_scale=0.7, color=(255, 255, 255), bg_alpha=0.7
        )
        
        # Draw violation counts
        y_offset = 70
        
        # Illegal parking violations
        color = (0, 165, 255) if violations["illegal_parking"] > 0 else (255, 255, 255)
        draw_text_with_background(
            frame, f"Illegal parking: {violations['illegal_parking']} current, {self.stats['illegal_parking_violations']} total", 
            (10, y_offset), font_scale=0.7, color=color, bg_alpha=0.7
        )
        y_offset += 40
        
        # Exit blocking violations
        color = (0, 0, 255) if violations["exit_blocking"] > 0 else (255, 255, 255)
        draw_text_with_background(
            frame, f"Exit blocking: {violations['exit_blocking']} current, {self.stats['exit_blocking_violations']} total", 
            (10, y_offset), font_scale=0.7, color=color, bg_alpha=0.7
        )
        y_offset += 40
        
        # Suspicious activity
        color = (255, 0, 255) if violations["suspicious_activity"] > 0 else (255, 255, 255)
        draw_text_with_background(
            frame, f"Suspicious activity: {violations['suspicious_activity']} current, {self.stats['suspicious_activity_count']} total", 
            (10, y_offset), font_scale=0.7, color=color, bg_alpha=0.7
        )
        y_offset += 40
        
        # Total vehicles
        draw_text_with_background(
            frame, f"Total vehicles: {self.stats['total_vehicles_detected']}", 
            (10, y_offset), font_scale=0.7, color=(255, 255, 255), bg_alpha=0.7
        )
        
        # Process alerts
        alert_y = h - 100  # Start alerts at the bottom of the frame
        
        for alert_type, alert in self.alerts.items():
            if alert["active"]:
                # Check if alert duration has expired
                if current_time - alert["start_time"] > alert["duration"]:
                    alert["active"] = False
                else:
                    # Draw alert
                    cv2.rectangle(frame, (0, alert_y), (w, alert_y + 40), (0, 0, 200), -1)
                    cv2.putText(frame, alert["message"], (20, alert_y + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    alert_y -= 50  # Move up for next alert
        
        return frame


# Run as standalone module
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parking Monitoring (Use Case 5)")
    parser.add_argument("--input", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video file")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--rules", type=str, default="rules.json", help="Path to rules file")
    parser.add_argument("--zones", type=str, default="zones.json", help="Path to zones file")
    parser.add_argument("--camera", type=str, default="parking", help="Camera identifier in config")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame")
    
    args = parser.parse_args()
    
    # Create parking monitor
    monitor = ParkingMonitor(
        config_file=args.config,
        rules_file=args.rules,
        zones_file=args.zones,
        camera_id=args.camera
    )
    
    # If input is provided, process the video
    if args.input:
        monitor.process_video(args.input, args.output, skip_frames=args.skip_frames)
    else:
        print("No input video specified. Use --input to specify a video file.")