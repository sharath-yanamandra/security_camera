"""
Use Case 2: Reception Monitoring
- Enforce maximum occupancy limit (10 people)
- Detect gate jumping attempts
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
    configure_models, load_rules, create_event_metadata
)


class ReceptionMonitor(VideoProcessor):
    """
    Monitor for reception area events:
    - Maximum occupancy enforcement
    - Gate jumping detection
    """
    
    def __init__(self, config_file="config.json", rules_file="rules.json", zones_file="zones.json", camera_id="reception"):
        """
        Initialize the Reception monitor
        
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
        
        # Load detection models
        detection_model = self.config.get("models", {}).get("detection", "yolov8l.pt")
        pose_model = self.config.get("models", {}).get("pose", "yolov8l-pose.pt")
        device = self.config.get("device", "cpu")
        confidence = self.config.get("confidence", 0.5)
        
        self.detection_model = ModelLoader.load_model(detection_model, device)
        self.detection_model.conf = confidence
        
        self.pose_model = ModelLoader.load_model(pose_model, device)
        self.pose_model.conf = confidence
        
        # Load zones
        self.zone_manager = ZoneManager(zones_file)
        
        # Define zone colors
        self.zone_colors = {
            "reception_area": (0, 255, 0),  # Green
            "gate_area": (255, 0, 0)        # Red
        }
        
        # Create object tracker
        processing_config = self.config.get("processing", {})
        self.person_tracker = ObjectTracker(
            max_disappeared=processing_config.get("max_disappeared", 30),
            max_distance=processing_config.get("tracking_distance", 50)
        )
        
        # Initialize detection state
        self.gate_jump_history = {}  # person ID -> jump data
        
        # Get rule parameters
        rule_params = self.camera_rules.get("max_occupancy", {}).get("parameters", {})
        self.max_people = rule_params.get("max_people", 10)
        
        rule_params = self.camera_rules.get("gate_jumping", {}).get("parameters", {})
        self.vertical_velocity_threshold = rule_params.get("vertical_velocity_threshold", 15)
        self.jump_height_threshold = rule_params.get("jump_height_threshold", 0.3)
        
        # Session statistics
        self.stats = {
            "max_occupancy_violations": 0,
            "gate_jumping_violations": 0,
            "peak_occupancy": 0
        }
        
        # Initialize alert state
        self.alerts = {
            "max_occupancy": {
                "active": False,
                "start_time": None,
                "message": f"ALERT: Reception over capacity! (>{self.max_people} people)",
                "duration": 5.0
            },
            "gate_jumping": {
                "active": False,
                "start_time": None,
                "message": "ALERT: Gate jumping attempt detected!",
                "duration": 5.0
            }
        }
        
        self.start_time = time.time()
        
        print(f"Reception monitoring initialized for camera: {camera_id}")
    
    def _get_zones_from_config(self):
        """Extract reception and gate zones from configuration"""
        zones = self.zone_manager.zones.get(self.camera_id, {})
        
        reception_zones = []
        if "reception_area" in zones:
            for zone in zones["reception_area"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Reception")
                    reception_zones.append((*coords, name))
        
        gate_zones = []
        if "gate_area" in zones:
            for zone in zones["gate_area"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Gate")
                    gate_zones.append((*coords, name))
        
        return reception_zones, gate_zones
    
    def detect_people(self, frame, detection_results, pose_results):
        """
        Detect and track people in the reception area and monitor gate activity
        
        Args:
            frame: Current video frame
            detection_results: Results from detection model
            pose_results: Results from pose model
            
        Returns:
            Tuple of (frame, people_count, over_capacity, gate_jumping_detected)
        """
        # Get reception and gate zones
        reception_zones, gate_zones = self._get_zones_from_config()
        
        # Process detection results to find people
        people = []
        
        if detection_results:
            for r in detection_results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0].item())
                    
                    # Only interested in persons (class 0)
                    if cls != 0:
                        continue
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox = (x1, y1, x2, y2)
                    confidence = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    
                    # Get center point
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Check if in reception area
                    in_reception = False
                    reception_name = None
                    
                    for zone in reception_zones:
                        if self.zone_manager.is_in_zone(center, zone):
                            in_reception = True
                            reception_name = zone[4] if len(zone) > 4 else "Reception"
                            break
                    
                    # Check if in gate area
                    in_gate = False
                    gate_name = None
                    
                    for zone in gate_zones:
                        if self.zone_manager.is_in_zone(center, zone):
                            in_gate = True
                            gate_name = zone[4] if len(zone) > 4 else "Gate"
                            break
                    
                    # Add to people list
                    people.append({
                        "bbox": bbox,
                        "center": center,
                        "confidence": confidence,
                        "in_reception": in_reception,
                        "reception_name": reception_name,
                        "in_gate": in_gate,
                        "gate_name": gate_name
                    })
        
        # Update person tracker
        person_detections = [(p["bbox"], "person", p["confidence"]) for p in people]
        person_objects = self.person_tracker.update(person_detections)
        
        # Count people in reception area
        people_in_reception = sum(1 for person in people if person["in_reception"])
        
        # Update peak occupancy
        self.stats["peak_occupancy"] = max(self.stats["peak_occupancy"], people_in_reception)
        
        # Check for maximum occupancy violation
        over_capacity = people_in_reception > self.max_people
        if over_capacity and not self.alerts["max_occupancy"]["active"]:
            self.alerts["max_occupancy"]["active"] = True
            self.alerts["max_occupancy"]["start_time"] = time.time()
            self.stats["max_occupancy_violations"] += 1
            
            # Save event screenshot
            if self.config.get("events", {}).get("save_screenshots", True):
                event_metadata = create_event_metadata(
                    frame, "max_occupancy", None, None, 
                    1.0, {"person_count": people_in_reception, "max_allowed": self.max_people}
                )
                self.save_screenshot(frame, "max_occupancy")
        
        # Detect gate jumping
        gate_jumping_detected = False
        person_jumping_id = None
        
        if pose_results:
            for person in pose_results:
                if not hasattr(person, 'keypoints') or not hasattr(person, 'boxes'):
                    continue
                
                # Get bounding boxes and keypoints
                for idx, (box, keypoints) in enumerate(zip(person.boxes, person.keypoints)):
                    if not box or len(box.xyxy) == 0:
                        continue
                        
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox = (x1, y1, x2, y2)
                    
                    # Get center point
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Check if in gate area
                    in_gate = False
                    for zone in gate_zones:
                        if self.zone_manager.is_in_zone(center, zone):
                            in_gate = True
                            break
                            
                    if not in_gate:
                        continue
                    
                    # Get keypoints
                    kpts = keypoints.data[0].cpu().numpy()
                    
                    # Find corresponding person ID
                    person_id = None
                    min_distance = float('inf')
                    
                    for pid, p_data in person_objects.items():
                        p_center = p_data["centroid"]
                        distance = np.sqrt((center[0] - p_center[0])**2 + (center[1] - p_center[1])**2)
                        if distance < min_distance and distance < 50:  # 50 pixel threshold
                            person_id = pid
                            min_distance = distance
                    
                    if person_id is None:
                        continue
                    
                    # Analyze vertical movement (for jump detection)
                    if person_id not in self.gate_jump_history:
                        self.gate_jump_history[person_id] = {
                            "positions": [],
                            "times": [],
                            "jump_detected": False,
                            "jump_time": None
                        }
                    
                    # Get height of person
                    height = y2 - y1
                    
                    # Add current position
                    self.gate_jump_history[person_id]["positions"].append(y1)
                    self.gate_jump_history[person_id]["times"].append(time.time())
                    
                    # Limit history length
                    max_history = 10
                    if len(self.gate_jump_history[person_id]["positions"]) > max_history:
                        self.gate_jump_history[person_id]["positions"].pop(0)
                        self.gate_jump_history[person_id]["times"].pop(0)
                    
                    # Need at least 3 positions to detect jumping
                    positions = self.gate_jump_history[person_id]["positions"]
                    times = self.gate_jump_history[person_id]["times"]
                    
                    if len(positions) >= 3:
                        # Calculate vertical velocities
                        velocities = []
                        for i in range(1, len(positions)):
                            time_diff = times[i] - times[i-1]
                            if time_diff > 0:  # Avoid division by zero
                                # Note: y decreases as you go up the image
                                # Positive velocity means moving up
                                velocity = (positions[i-1] - positions[i]) / time_diff
                                velocities.append(velocity)
                        
                        # Check for substantial upward movement (jumping)
                        if velocities and max(velocities) > self.vertical_velocity_threshold:
                            # Check total height change
                            height_change = max(positions) - min(positions)
                            if height_change > height * self.jump_height_threshold:
                                # Jump detected!
                                if not self.gate_jump_history[person_id]["jump_detected"]:
                                    self.gate_jump_history[person_id]["jump_detected"] = True
                                    self.gate_jump_history[person_id]["jump_time"] = time.time()
                                    gate_jumping_detected = True
                                    person_jumping_id = person_id
                                    
                                    # Update alert state
                                    if not self.alerts["gate_jumping"]["active"]:
                                        self.alerts["gate_jumping"]["active"] = True
                                        self.alerts["gate_jumping"]["start_time"] = time.time()
                                        self.stats["gate_jumping_violations"] += 1
                                        
                                        # Save event screenshot
                                        if self.config.get("events", {}).get("save_screenshots", True):
                                            event_metadata = create_event_metadata(
                                                frame, "gate_jumping", person_id, None, 
                                                1.0, {"velocity": max(velocities), "height_change": height_change}
                                            )
                                            self.save_screenshot(frame, "gate_jumping")
                    
                    # Reset jump detection after a timeout
                    if (self.gate_jump_history[person_id]["jump_detected"] and 
                        time.time() - self.gate_jump_history[person_id]["jump_time"] > 3.0):
                        self.gate_jump_history[person_id]["jump_detected"] = False
        
        # Visualize people with tracking info
        for person_id, person_data in person_objects.items():
            bbox = person_data["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Find if person is in reception or at gate
            in_reception = False
            in_gate = False
            
            center = person_data["centroid"]
            
            for zone in reception_zones:
                if self.zone_manager.is_in_zone(center, zone):
                    in_reception = True
                    break
                    
            for zone in gate_zones:
                if self.zone_manager.is_in_zone(center, zone):
                    in_gate = True
                    break
            
            # Determine color based on location and activity
            is_jumping = person_id in self.gate_jump_history and self.gate_jump_history[person_id]["jump_detected"]
            
            if is_jumping:
                color = (0, 0, 255)  # Red for jumping
            elif in_gate:
                color = (255, 165, 0)  # Orange for gate area
            elif in_reception:
                color = (0, 255, 0)  # Green for reception area
            else:
                color = (255, 255, 255)  # White for outside areas
                
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw person ID and status
            status_text = f"ID:{person_id}"
            if is_jumping:
                status_text += " JUMPING!"
            elif in_gate:
                status_text += " (Gate)"
            elif in_reception:
                status_text += " (Reception)"
            
            draw_text_with_background(
                frame, status_text, (x1, y1 - 10),
                color=color, bg_alpha=0.7
            )
        
        return frame, people_in_reception, over_capacity, gate_jumping_detected
    
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
        
        # Run pose estimation
        pose_results = self.pose_model(frame)
        
        # Draw zones
        self._draw_zones(result_frame)
        
        # Detect and track people
        result_frame, people_count, over_capacity, gate_jumping = self.detect_people(
            result_frame, detection_results, pose_results
        )
        
        # Draw status and alerts
        result_frame = self._draw_status_and_alerts(result_frame, people_count)
        
        # Return processed frame and detection results
        detection_info = {
            "people_count": people_count,
            "over_capacity": over_capacity,
            "gate_jumping": gate_jumping,
            "max_allowed": self.max_people
        }
        
        return result_frame, detection_info
    
    def _draw_zones(self, frame):
        """Draw monitoring zones on the frame"""
        zones = self.zone_manager.zones.get(self.camera_id, {})
        
        # Draw reception zones
        if "reception_area" in zones:
            for zone in zones["reception_area"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Reception")
                    draw_zone(frame, (*coords, name), self.zone_colors["reception_area"], 0.3, "Reception: " + name)
        
        # Draw gate zones
        if "gate_area" in zones:
            for zone in zones["gate_area"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Gate")
                    draw_zone(frame, (*coords, name), self.zone_colors["gate_area"], 0.3, "Gate: " + name)
        
        return frame
    
    def _draw_status_and_alerts(self, frame, current_people_count):
        """
        Draw status information and alerts on the frame
        
        Args:
            frame: Video frame to draw on
            current_people_count: Current number of people in reception
            
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
        
        # Draw people count with capacity indicator
        count_color = (0, 0, 255) if current_people_count > self.max_people else (0, 255, 0)
        draw_text_with_background(
            frame, f"People in reception: {current_people_count}/{self.max_people}", 
            (10, 70), font_scale=0.7, color=count_color, bg_alpha=0.7
        )
        
        # Draw capacity status bar
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = 90
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Fill based on capacity
        fill_width = min(bar_width, int(bar_width * current_people_count / self.max_people))
        fill_color = (0, 0, 255) if current_people_count > self.max_people else (0, 255, 0)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), fill_color, -1)
        
        # Draw statistics
        y_offset = 130
        
        # Max occupancy violations
        color = (0, 0, 255) if self.stats["max_occupancy_violations"] > 0 else (255, 255, 255)
        draw_text_with_background(
            frame, f"Capacity violations: {self.stats['max_occupancy_violations']}", 
            (10, y_offset), font_scale=0.7, color=color, bg_alpha=0.7
        )
        y_offset += 40
        
        # Gate jumping violations
        color = (0, 0, 255) if self.stats["gate_jumping_violations"] > 0 else (255, 255, 255)
        draw_text_with_background(
            frame, f"Gate jumping attempts: {self.stats['gate_jumping_violations']}", 
            (10, y_offset), font_scale=0.7, color=color, bg_alpha=0.7
        )
        y_offset += 40
        
        # Peak occupancy
        draw_text_with_background(
            frame, f"Peak occupancy: {self.stats['peak_occupancy']}", 
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
    
    parser = argparse.ArgumentParser(description="Reception Area Monitoring (Use Case 2)")
    parser.add_argument("--input", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video file")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--rules", type=str, default="rules.json", help="Path to rules file")
    parser.add_argument("--zones", type=str, default="zones.json", help="Path to zones file")
    parser.add_argument("--camera", type=str, default="reception", help="Camera identifier in config")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame")
    
    args = parser.parse_args()
    
    # Create reception monitor
    monitor = ReceptionMonitor(
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