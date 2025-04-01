"""
Use Case 1: Entry Monitoring
- Detection of people sitting on stairs
- Detection of gatherings
- Vehicle entry monitoring (trucks should not be parked)
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


class EntryMonitor(VideoProcessor):
    """
    Monitor for entry area events:
    - People sitting on stairs
    - Gatherings of people
    - Vehicle entry with truck parking detection
    """
    
    def __init__(self, config_file="config.json", rules_file="rules.json", zones_file="zones.json", camera_id="entry"):
        """
        Initialize the Entry Area monitor
        
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
            "stairs": (0, 0, 255),     # Red
            "gathering": (0, 255, 255), # Yellow
            "vehicle_entry": (0, 255, 0)  # Green
        }
        
        # Create object tracker
        processing_config = self.config.get("processing", {})
        self.person_tracker = ObjectTracker(
            max_disappeared=processing_config.get("max_disappeared", 30),
            max_distance=processing_config.get("tracking_distance", 50)
        )
        self.vehicle_tracker = ObjectTracker(
            max_disappeared=processing_config.get("max_disappeared", 30),
            max_distance=processing_config.get("tracking_distance", 50)
        )
        
        # Initialize detection state
        self.sitting_detections = {}  # person ID -> sitting data
        self.gathering_history = []
        self.vehicle_detections = {}  # vehicle ID -> vehicle data
        
        # Session statistics
        self.stats = {
            "sitting_violations": 0,
            "gathering_violations": 0,
            "truck_parking_violations": 0,
            "total_vehicles_detected": 0
        }
        
        # Initialize alert state
        self.alerts = {
            "sitting_on_stairs": {
                "active": False,
                "start_time": None,
                "message": "ALERT: Person sitting on stairs!",
                "duration": 5.0
            },
            "gathering": {
                "active": False,
                "start_time": None,
                "message": "ALERT: Gathering detected!",
                "duration": 5.0
            },
            "truck_parked": {
                "active": False,
                "start_time": None,
                "message": "ALERT: Truck parked in no-parking zone!",
                "duration": 5.0
            }
        }
        
        self.start_time = time.time()
        
        print(f"Entry monitoring initialized for camera: {camera_id}")
        
        # Get rule parameters
        rule_params = self.camera_rules.get("sitting_on_stairs", {}).get("parameters", {})
        self.sitting_min_confidence = rule_params.get("min_detection_confidence", 0.6)
        self.sitting_min_duration = rule_params.get("min_duration", 5)
        
        rule_params = self.camera_rules.get("gathering", {}).get("parameters", {})
        self.gathering_threshold = rule_params.get("people_threshold", 3)
        self.gathering_min_duration = rule_params.get("min_duration", 10)
        
        rule_params = self.camera_rules.get("parked_truck", {}).get("parameters", {})
        self.truck_stationary_time = rule_params.get("min_stationary_time", 10)
        self.movement_threshold = rule_params.get("movement_threshold", 5)
    
    def detect_sitting_on_stairs(self, pose_results, frame):
        """
        Detect people sitting on stairs using pose estimation
        
        Args:
            pose_results: Results from pose estimation model
            frame: Current video frame
            
        Returns:
            Tuple of (frame, sitting_detections)
        """
        # Get stairs zones
        stair_zones = self.zone_manager.zones.get(self.camera_id, {}).get("stairs", [])
        
        # Extract stair coordinates
        stairs_areas = []
        for zone in stair_zones:
            if "coordinates" in zone:
                coords = zone["coordinates"]
                name = zone.get("name", "Stairs")
                stairs_areas.append((*coords, name))
        
        # Process pose results
        sitting_people = []
        
        if pose_results:
            for result in pose_results:
                if not hasattr(result, 'keypoints') or not hasattr(result, 'boxes'):
                    continue
                
                # Get bounding boxes and keypoints
                for idx, (box, keypoints) in enumerate(zip(result.boxes, result.keypoints)):
                    if not box or len(box.xyxy) == 0:
                        continue
                        
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox = (x1, y1, x2, y2)
                    confidence = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    
                    # Skip if confidence is too low
                    if confidence < self.sitting_min_confidence:
                        continue
                    
                    # Get center point
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Check if in stairs area
                    in_stairs_area = False
                    stairs_name = None
                    
                    for stairs in stairs_areas:
                        if self.zone_manager.is_in_zone(center, stairs):
                            in_stairs_area = True
                            stairs_name = stairs[4] if len(stairs) > 4 else "Stairs"
                            break
                    
                    if not in_stairs_area:
                        continue
                    
                    # Get pose keypoints
                    kpts = keypoints.data[0].cpu().numpy()
                    
                    # Check for sitting posture
                    # Keypoints: 11-left hip, 12-right hip, 13-left knee, 14-right knee
                    is_sitting = False
                    
                    if len(kpts) >= 17:  # YOLOv8 Pose has 17 keypoints
                        left_hip = kpts[11][:2] if kpts[11][2] > 0.5 else None
                        right_hip = kpts[12][:2] if kpts[12][2] > 0.5 else None
                        left_knee = kpts[13][:2] if kpts[13][2] > 0.5 else None
                        right_knee = kpts[14][:2] if kpts[14][2] > 0.5 else None
                        
                        # Check if required keypoints are detected
                        if left_hip is not None and right_hip is not None and left_knee is not None and right_knee is not None:
                            # Simple heuristic: in sitting position, knees are typically at same height or higher than hips
                            if (left_knee[1] <= left_hip[1] + 10) or (right_knee[1] <= right_hip[1] + 10):
                                is_sitting = True
                    
                    if is_sitting:
                        # Get or assign person ID
                        detections = [(bbox, "person", confidence)]
                        person_objects = self.person_tracker.update(detections)
                        
                        # Get the ID of the first (and only) person
                        person_id = list(person_objects.keys())[0]
                        
                        # Add to sitting detections
                        sitting_people.append({
                            "person_id": person_id,
                            "bbox": bbox,
                            "center": center,
                            "confidence": confidence,
                            "stairs_name": stairs_name
                        })
                        
                        # Update sitting detection state
                        if person_id not in self.sitting_detections:
                            self.sitting_detections[person_id] = {
                                "first_detected": time.time(),
                                "last_detected": time.time(),
                                "stairs_name": stairs_name,
                                "violation_reported": False
                            }
                        else:
                            self.sitting_detections[person_id]["last_detected"] = time.time()
                            self.sitting_detections[person_id]["stairs_name"] = stairs_name
                        
                        # Check if sitting duration exceeds threshold
                        sitting_duration = time.time() - self.sitting_detections[person_id]["first_detected"]
                        
                        if sitting_duration >= self.sitting_min_duration and not self.sitting_detections[person_id]["violation_reported"]:
                            # Record violation
                            self.stats["sitting_violations"] += 1
                            self.sitting_detections[person_id]["violation_reported"] = True
                            
                            # Create alert
                            self.alerts["sitting_on_stairs"]["active"] = True
                            self.alerts["sitting_on_stairs"]["start_time"] = time.time()
                            
                            # Save event screenshot
                            if self.config.get("events", {}).get("save_screenshots", True):
                                event_metadata = create_event_metadata(
                                    frame, "sitting_on_stairs", person_id, stairs_name, 
                                    confidence, {"duration": sitting_duration}
                                )
                                self.save_screenshot(frame, "sitting_on_stairs")
                        
                        # Draw sitting indication
                        color = (0, 0, 255)  # Red for sitting violation
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw sitting duration
                        duration_text = f"Sitting: {format_duration(sitting_duration)}"
                        draw_text_with_background(
                            frame, duration_text, (x1, y2 + 20), 
                            color=color, bg_alpha=0.7
                        )
        
        # Process expired sitting detections
        current_time = time.time()
        expired_ids = []
        
        for person_id, detection in self.sitting_detections.items():
            # If not detected in this frame
            if person_id not in [p["person_id"] for p in sitting_people]:
                # If not seen recently, remove
                if current_time - detection["last_detected"] > 5.0:
                    expired_ids.append(person_id)
        
        # Remove expired detections
        for person_id in expired_ids:
            del self.sitting_detections[person_id]
        
        return frame, sitting_people
    
    def detect_gathering(self, detection_results, frame):
        """
        Detect gatherings of people
        
        Args:
            detection_results: Results from detection model
            frame: Current video frame
            
        Returns:
            Tuple of (frame, is_gathering_detected)
        """
        # Get gathering zones
        gathering_zones = self.zone_manager.zones.get(self.camera_id, {}).get("gathering", [])
        
        # Extract gathering areas
        gathering_areas = []
        for zone in gathering_zones:
            if "coordinates" in zone:
                coords = zone["coordinates"]
                name = zone.get("name", "Gathering Area")
                gathering_areas.append((*coords, name))
        
        # Process detection results
        persons = []
        
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
                    
                    # Check if in gathering area
                    in_gathering_area = False
                    area_name = None
                    
                    for area in gathering_areas:
                        if self.zone_manager.is_in_zone(center, area):
                            in_gathering_area = True
                            area_name = area[4] if len(area) > 4 else "Gathering Area"
                            break
                    
                    if in_gathering_area:
                        # Add to persons list
                        persons.append({
                            "bbox": bbox,
                            "center": center,
                            "confidence": confidence,
                            "area_name": area_name
                        })
        
        # Update gathering history
        self.gathering_history.append(len(persons))
        if len(self.gathering_history) > 30:  # Track last 30 frames
            self.gathering_history.pop(0)
        
        # Check if we have a consistent gathering
        avg_persons = sum(self.gathering_history) / max(1, len(self.gathering_history))
        is_gathering = avg_persons >= self.gathering_threshold
        
        # Update alert state
        if is_gathering and not self.alerts["gathering"]["active"]:
            self.alerts["gathering"]["active"] = True
            self.alerts["gathering"]["start_time"] = time.time()
            self.stats["gathering_violations"] += 1
            
            # Save event screenshot
            if self.config.get("events", {}).get("save_screenshots", True):
                event_metadata = create_event_metadata(
                    frame, "gathering", None, area_name if 'area_name' in locals() else None, 
                    1.0, {"person_count": len(persons)}
                )
                self.save_screenshot(frame, "gathering")
        
        # If gathering detected, visualize
        if is_gathering and persons:
            # Calculate the centroid of all people
            center_x = sum(p["center"][0] for p in persons) // len(persons)
            center_y = sum(p["center"][1] for p in persons) // len(persons)
            
            # Calculate average distance from center (for circle radius)
            distances = [np.sqrt((p["center"][0] - center_x)**2 + (p["center"][1] - center_y)**2) for p in persons]
            radius = max(30, int(sum(distances) / len(distances)) + 20)
            
            # Draw circle around gathering
            cv2.circle(frame, (center_x, center_y), radius, (0, 165, 255), 2)
            
            # Draw text for gathering
            gathering_text = f"Gathering: {len(persons)} people"
            draw_text_with_background(
                frame, gathering_text, (center_x - 100, center_y + radius + 20),
                color=(0, 165, 255), bg_alpha=0.7
            )
        
        return frame, is_gathering
    
    def detect_vehicles(self, detection_results, frame):
        """
        Detect vehicles and check for parked trucks
        
        Args:
            detection_results: Results from detection model
            frame: Current video frame
            
        Returns:
            Tuple of (frame, vehicles_detected, truck_parked)
        """
        # Get vehicle entry zones
        vehicle_zones = self.zone_manager.zones.get(self.camera_id, {}).get("vehicle_entry", [])
        
        # Extract vehicle areas
        vehicle_areas = []
        for zone in vehicle_zones:
            if "coordinates" in zone:
                coords = zone["coordinates"]
                name = zone.get("name", "Vehicle Area")
                vehicle_areas.append((*coords, name))
        
        # Process detection results
        vehicles = []
        truck_parked = False
        
        if detection_results:
            for r in detection_results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0].item())
                    
                    # Check if this is a vehicle class we care about
                    if cls not in VEHICLE_CLASSES:
                        continue
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox = (x1, y1, x2, y2)
                    confidence = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    
                    # Get center point
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Check if in vehicle area
                    in_vehicle_area = False
                    area_name = None
                    
                    for area in vehicle_areas:
                        if self.zone_manager.is_in_zone(center, area):
                            in_vehicle_area = True
                            area_name = area[4] if len(area) > 4 else "Vehicle Area"
                            break
                    
                    # Get vehicle type
                    vehicle_type = VEHICLE_CLASSES[cls]
                    
                    # Add to vehicles list
                    vehicles.append({
                        "bbox": bbox,
                        "center": center,
                        "confidence": confidence,
                        "area_name": area_name if in_vehicle_area else None,
                        "in_vehicle_area": in_vehicle_area,
                        "type": vehicle_type,
                        "class_id": cls
                    })
        
        # Update vehicle tracker
        vehicle_detections = [(v["bbox"], v["type"], v["confidence"]) for v in vehicles]
        vehicle_objects = self.vehicle_tracker.update(vehicle_detections)
        
        # Check for parked trucks
        for vehicle_id, vehicle_data in vehicle_objects.items():
            vehicle_type = vehicle_data["type"]
            
            # Check if this is a truck
            if vehicle_type == "truck":
                # Check if stationary
                is_stationary, stationary_duration = is_object_stationary(
                    vehicle_data, 
                    time_threshold=self.truck_stationary_time,
                    movement_threshold=self.movement_threshold
                )
                
                if is_stationary:
                    truck_parked = True
                    
                    # Update alert state if not already active
                    if not self.alerts["truck_parked"]["active"]:
                        self.alerts["truck_parked"]["active"] = True
                        self.alerts["truck_parked"]["start_time"] = time.time()
                        self.stats["truck_parking_violations"] += 1
                        
                        # Save event screenshot
                        if self.config.get("events", {}).get("save_screenshots", True):
                            event_metadata = create_event_metadata(
                                frame, "truck_parked", vehicle_id, 
                                vehicle_data.get("area_name"), 
                                vehicle_data["confidence"], 
                                {"duration": stationary_duration}
                            )
                            self.save_screenshot(frame, "truck_parked")
                    
                    # Draw truck parking violation
                    x1, y1, x2, y2 = vehicle_data["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Draw violation text
                    violation_text = f"TRUCK PARKED: {format_duration(stationary_duration)}"
                    draw_text_with_background(
                        frame, violation_text, (x1, y1 - 10),
                        color=(0, 0, 255), bg_alpha=0.7
                    )
        
        # Visualize all vehicles
        for vehicle_id, vehicle_data in vehicle_objects.items():
            # Skip if already visualized as parking violation
            if vehicle_data["type"] == "truck" and truck_parked:
                continue
            
            x1, y1, x2, y2 = vehicle_data["bbox"]
            vehicle_type = vehicle_data["type"]
            
            # Determine color based on vehicle type
            if vehicle_type == "truck":
                color = (255, 165, 0)  # Orange for trucks
            else:
                color = (0, 255, 0)  # Green for other vehicles
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw vehicle info
            is_stationary, stationary_duration = is_object_stationary(vehicle_data)
            duration_text = format_duration(stationary_duration) if is_stationary else "moving"
            info_text = f"{vehicle_type.upper()} ID:{vehicle_id} ({duration_text})"
            
            draw_text_with_background(
                frame, info_text, (x1, y1 - 10),
                color=color, bg_alpha=0.7
            )
        
        # Update statistics
        self.stats["total_vehicles_detected"] = len(vehicle_objects)
        
        return frame, vehicles, truck_parked
    
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
        
        # Detect sitting on stairs
        result_frame, sitting_people = self.detect_sitting_on_stairs(pose_results, result_frame)
        
        # Detect gatherings
        result_frame, gathering_detected = self.detect_gathering(detection_results, result_frame)
        
        # Detect vehicles and truck parking
        result_frame, vehicles, truck_parked = self.detect_vehicles(detection_results, result_frame)
        
        # Draw status and alerts
        result_frame = self._draw_status_and_alerts(result_frame)
        
        # Return processed frame and detection results
        detection_info = {
            "sitting_count": len(sitting_people),
            "gathering_detected": gathering_detected,
            "vehicle_count": len(vehicles),
            "truck_parked": truck_parked
        }
        
        return result_frame, detection_info
    
    def _draw_zones(self, frame):
        """Draw monitoring zones on the frame"""
        zones = self.zone_manager.zones.get(self.camera_id, {})
        
        # Draw stairs zones
        if "stairs" in zones:
            for zone in zones["stairs"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Stairs")
                    draw_zone(frame, (*coords, name), self.zone_colors["stairs"], 0.3, "Stairs: " + name)
        
        # Draw gathering zones
        if "gathering" in zones:
            for zone in zones["gathering"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Gathering Area")
                    draw_zone(frame, (*coords, name), self.zone_colors["gathering"], 0.3, "Gathering: " + name)
        
        # Draw vehicle zones
        if "vehicle_entry" in zones:
            for zone in zones["vehicle_entry"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Vehicle Area")
                    draw_zone(frame, (*coords, name), self.zone_colors["vehicle_entry"], 0.3, "Vehicle: " + name)
        
        return frame
    
    def _draw_status_and_alerts(self, frame):
        """Draw status information and alerts on the frame"""
        h, w = frame.shape[:2]
        current_time = time.time()
        
        # Draw session time
        session_time = current_time - self.start_time
        time_str = format_duration(session_time)
        draw_text_with_background(
            frame, f"Session time: {time_str}", (10, 30),
            font_scale=0.7, color=(255, 255, 255), bg_alpha=0.7
        )
        
        # Draw event counts
        y_offset = 70
        
        # Sitting violations
        color = (0, 0, 255) if self.stats["sitting_violations"] > 0 else (255, 255, 255)
        draw_text_with_background(
            frame, f"Sitting violations: {self.stats['sitting_violations']}", 
            (10, y_offset), font_scale=0.7, color=color, bg_alpha=0.7
        )
        y_offset += 40
        
        # Gathering violations
        color = (0, 0, 255) if self.stats["gathering_violations"] > 0 else (255, 255, 255)
        draw_text_with_background(
            frame, f"Gathering events: {self.stats['gathering_violations']}", 
            (10, y_offset), font_scale=0.7, color=color, bg_alpha=0.7
        )
        y_offset += 40
        
        # Truck parking violations
        color = (0, 0, 255) if self.stats["truck_parking_violations"] > 0 else (255, 255, 255)
        draw_text_with_background(
            frame, f"Truck parking violations: {self.stats['truck_parking_violations']}", 
            (10, y_offset), font_scale=0.7, color=color, bg_alpha=0.7
        )
        y_offset += 40
        
        # Vehicle count
        draw_text_with_background(
            frame, f"Total vehicles detected: {self.stats['total_vehicles_detected']}", 
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
    
    parser = argparse.ArgumentParser(description="Entry Area Monitoring (Use Case 1)")
    parser.add_argument("--input", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video file")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--rules", type=str, default="rules.json", help="Path to rules file")
    parser.add_argument("--zones", type=str, default="zones.json", help="Path to zones file")
    parser.add_argument("--camera", type=str, default="entry", help="Camera identifier in config")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame")
    
    args = parser.parse_args()
    
    # Create entry monitor
    monitor = EntryMonitor(
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

    '''
    how to run --> python3 datacenter_zone_monitoring.py --input "test_videos/entry.mp4"  --output "results/output_entry.mp4"
    
    '''