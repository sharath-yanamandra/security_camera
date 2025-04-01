"""
Use Case 4: Inside Datacenter Monitoring
- Detection of people coming from unauthorized zones (e.g., fire service areas)
- Monitoring movement between zones and transitions
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


class DatacenterZoneMonitor(VideoProcessor):
    """
    Monitor for inside datacenter zones:
    - Track people movements between zones
    - Detect unauthorized zone transitions
    """
    
    def __init__(self, config_file="config.json", rules_file="rules.json", zones_file="zones.json", camera_id="datacenter_inside"):
        """
        Initialize the Datacenter Zone monitor
        
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
            "authorized": (0, 255, 0),     # Green
            "unauthorized": (0, 0, 255),   # Red
            "entry_points": (255, 165, 0)  # Orange
        }
        
        # Create object tracker
        processing_config = self.config.get("processing", {})
        self.person_tracker = ObjectTracker(
            max_disappeared=processing_config.get("max_disappeared", 30),
            max_distance=processing_config.get("tracking_distance", 50)
        )
        
        # Initialize detection state
        self.zone_transitions = {}  # Dict to track person zone transitions
        
        # Get rule parameters
        rule_params = self.camera_rules.get("unauthorized_zone_entry", {}).get("parameters", {})
        self.min_transition_confidence = rule_params.get("min_transition_confidence", 0.8)
        
        # Session statistics
        self.stats = {
            "unauthorized_transitions": 0,
            "zone_occupancy": {},  # Dict mapping zone names to current occupancy
            "total_transitions": 0
        }
        
        # Initialize zone occupancy counts
        for zone_type in ["authorized", "unauthorized", "entry_points"]:
            zones = self.zone_manager.zones.get(self.camera_id, {}).get(zone_type, [])
            for zone in zones:
                if "name" in zone:
                    self.stats["zone_occupancy"][zone["name"]] = 0
        
        # Initialize alert state
        self.alerts = {
            "unauthorized_entry": {
                "active": False,
                "start_time": None,
                "person_id": None,
                "message": "ALERT: Person entered from unauthorized zone!",
                "from_zone": None,
                "to_zone": None,
                "duration": 5.0
            }
        }
        
        self.start_time = time.time()
        
        print(f"Datacenter zone monitoring initialized for camera: {camera_id}")
    
    def _get_zones_from_config(self):
        """Extract datacenter zones from configuration"""
        zones = self.zone_manager.zones.get(self.camera_id, {})
        
        authorized_zones = []
        if "authorized" in zones:
            for zone in zones["authorized"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Authorized Zone")
                    authorized_zones.append((*coords, name))
        
        unauthorized_zones = []
        if "unauthorized" in zones:
            for zone in zones["unauthorized"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Unauthorized Zone")
                    unauthorized_zones.append((*coords, name))
        
        entry_points = []
        if "entry_points" in zones:
            for zone in zones["entry_points"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Entry Point")
                    entry_points.append((*coords, name))
        
        return authorized_zones, unauthorized_zones, entry_points
    
    def _get_current_zone(self, person_center):
        """
        Determine which zone a person is in based on center point
        
        Args:
            person_center: (x, y) center point of person
            
        Returns:
            Tuple of (zone_type, zone_name) or (None, None) if not in any zone
        """
        authorized_zones, unauthorized_zones, entry_points = self._get_zones_from_config()
        
        # Check entry points first (they take precedence)
        for zone in entry_points:
            if self.zone_manager.is_in_zone(person_center, zone):
                return "entry_points", zone[4] if len(zone) > 4 else "Entry Point"
        
        # Check authorized zones
        for zone in authorized_zones:
            if self.zone_manager.is_in_zone(person_center, zone):
                return "authorized", zone[4] if len(zone) > 4 else "Authorized"
        
        # Check unauthorized zones
        for zone in unauthorized_zones:
            if self.zone_manager.is_in_zone(person_center, zone):
                return "unauthorized", zone[4] if len(zone) > 4 else "Unauthorized"
        
        # Not in any defined zone
        return None, None
    
    def _update_zone_occupancy(self, old_zone_name, new_zone_name):
        """
        Update zone occupancy counts when a person changes zones
        
        Args:
            old_zone_name: Previous zone name or None
            new_zone_name: New zone name or None
        """
        # Decrement old zone count if it exists
        if old_zone_name and old_zone_name in self.stats["zone_occupancy"]:
            self.stats["zone_occupancy"][old_zone_name] = max(0, self.stats["zone_occupancy"][old_zone_name] - 1)
        
        # Increment new zone count if it exists
        if new_zone_name and new_zone_name in self.stats["zone_occupancy"]:
            self.stats["zone_occupancy"][new_zone_name] += 1
    
    def _check_unauthorized_transition(self, from_zone_type, from_zone_name, to_zone_type, to_zone_name):
        """
        Check if a zone transition is unauthorized
        
        Args:
            from_zone_type: Previous zone type or None
            from_zone_name: Previous zone name or None
            to_zone_type: New zone type or None
            to_zone_name: New zone name or None
            
        Returns:
            Boolean indicating if transition is unauthorized
        """
        # Person exists from an unauthorized zone into an authorized zone
        if from_zone_type == "unauthorized" and to_zone_type == "authorized":
            return True
        
        # Person appears directly in authorized zone without going through entry point
        if (from_zone_type is None or from_zone_type == "unauthorized") and to_zone_type == "authorized":
            return True
        
        return False
    
    def detect_people_and_transitions(self, frame, detection_results, pose_results):
        """
        Detect and track people inside the datacenter
        Monitor zone transitions with focus on unauthorized movements
        
        Args:
            frame: Current video frame
            detection_results: Results from detection model
            pose_results: Results from pose model (for better tracking)
            
        Returns:
            Tuple of (frame, person_count, unauthorized_transitions)
        """
        current_time = time.time()
        unauthorized_transitions = []
        
        # Prepare object detections
        all_persons = []
        
        # Process detection results
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
                    
                    # Add to people list
                    all_persons.append({
                        "bbox": bbox,
                        "center": center,
                        "confidence": confidence,
                        "keypoints": None
                    })
        
        # Process pose results to get keypoints (for better tracking)
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
                    
                    # Get center point
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Get keypoints
                    kpts = keypoints.data[0].cpu().numpy() if hasattr(keypoints, 'data') else None
                    
                    # Find if this person already exists in our detection list
                    best_match_idx = None
                    best_match_score = float('inf')
                    
                    for i, p in enumerate(all_persons):
                        p_center = p["center"]
                        dist = np.sqrt((center[0] - p_center[0])**2 + (center[1] - p_center[1])**2)
                        if dist < 50 and dist < best_match_score:  # 50 pixel threshold
                            best_match_idx = i
                            best_match_score = dist
                    
                    if best_match_idx is not None:
                        # Update existing detection with keypoints
                        all_persons[best_match_idx]["keypoints"] = kpts
                    else:
                        # Add as a new detection
                        all_persons.append({
                            "bbox": bbox,
                            "center": center,
                            "confidence": box.conf[0].item() if hasattr(box, 'conf') else 0.5,
                            "keypoints": kpts
                        })
        
        # Update person tracker
        person_detections = [(p["bbox"], "person", p["confidence"]) for p in all_persons]
        person_objects = self.person_tracker.update(person_detections)
        
        # Check for zone transitions
        for person_id, person_data in person_objects.items():
            bbox = person_data["bbox"]
            center = person_data["centroid"]
            
            # Get current zone
            zone_type, zone_name = self._get_current_zone(center)
            
            # Initialize transition tracking if needed
            if person_id not in self.zone_transitions:
                self.zone_transitions[person_id] = {
                    "current_zone_type": zone_type,
                    "current_zone_name": zone_name,
                    "previous_zone_type": None,
                    "previous_zone_name": None,
                    "entry_time": current_time,
                    "path": []  # List of (zone_type, zone_name, timestamp) tuples
                }
                
                # Add initial zone to path
                if zone_type and zone_name:
                    self.zone_transitions[person_id]["path"].append((zone_type, zone_name, current_time))
                    
                    # Update zone occupancy
                    self._update_zone_occupancy(None, zone_name)
            else:
                # Check for zone change
                current_zone_type = self.zone_transitions[person_id]["current_zone_type"]
                current_zone_name = self.zone_transitions[person_id]["current_zone_name"]
                
                if zone_type != current_zone_type or zone_name != current_zone_name:
                    # Zone transition detected
                    previous_zone_type = current_zone_type
                    previous_zone_name = current_zone_name
                    
                    # Update transition data
                    self.zone_transitions[person_id]["previous_zone_type"] = previous_zone_type
                    self.zone_transitions[person_id]["previous_zone_name"] = previous_zone_name
                    self.zone_transitions[person_id]["current_zone_type"] = zone_type
                    self.zone_transitions[person_id]["current_zone_name"] = zone_name
                    self.zone_transitions[person_id]["entry_time"] = current_time
                    
                    # Add to path history
                    if zone_type and zone_name:
                        self.zone_transitions[person_id]["path"].append((zone_type, zone_name, current_time))
                        
                        # Limit path history length
                        if len(self.zone_transitions[person_id]["path"]) > 20:
                            self.zone_transitions[person_id]["path"].pop(0)
                    
                    # Update zone occupancy
                    self._update_zone_occupancy(previous_zone_name, zone_name)
                    
                    # Increment total transitions counter
                    self.stats["total_transitions"] += 1
                    
                    # Check if transition is unauthorized
                    is_unauthorized = self._check_unauthorized_transition(
                        previous_zone_type, previous_zone_name, zone_type, zone_name
                    )
                    
                    if is_unauthorized:
                        # Record unauthorized transition
                        self.stats["unauthorized_transitions"] += 1
                        
                        transition_info = {
                            "person_id": person_id,
                            "from_type": previous_zone_type,
                            "from_name": previous_zone_name,
                            "to_type": zone_type,
                            "to_name": zone_name,
                            "timestamp": current_time
                        }
                        
                        unauthorized_transitions.append(transition_info)
                        
                        # Update alert state
                        if not self.alerts["unauthorized_entry"]["active"]:
                            self.alerts["unauthorized_entry"]["active"] = True
                            self.alerts["unauthorized_entry"]["start_time"] = current_time
                            self.alerts["unauthorized_entry"]["person_id"] = person_id
                            self.alerts["unauthorized_entry"]["from_zone"] = previous_zone_name or "Unknown"
                            self.alerts["unauthorized_entry"]["to_zone"] = zone_name
                            self.alerts["unauthorized_entry"]["message"] = (
                                f"ALERT: Person {person_id} entered {zone_name} from {previous_zone_name or 'Unknown'}!"
                            )
                            
                            # Save event screenshot
                            if self.config.get("events", {}).get("save_screenshots", True):
                                event_metadata = create_event_metadata(
                                    frame, "unauthorized_zone_entry", person_id, 
                                    f"{previous_zone_name or 'Unknown'} to {zone_name}", 
                                    person_data["confidence"], transition_info
                                )
                                self.save_screenshot(frame, "unauthorized_zone_entry")
        
        # Visualize people with tracking and zone info
        for person_id, person_data in person_objects.items():
            bbox = person_data["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Skip if no zone transition data
            if person_id not in self.zone_transitions:
                continue
            
            # Get zone information
            transition_data = self.zone_transitions[person_id]
            zone_type = transition_data["current_zone_type"]
            zone_name = transition_data["current_zone_name"]
            
            # Determine color based on zone
            if zone_type == "unauthorized":
                color = (0, 0, 255)  # Red for unauthorized zone
            elif zone_type == "authorized":
                color = (0, 255, 0)  # Green for authorized zone
            elif zone_type == "entry_points":
                color = (255, 165, 0)  # Orange for entry points
            else:
                color = (255, 255, 255)  # White for outside zones
            
            # Check if this person had an unauthorized transition
            for transition in unauthorized_transitions:
                if transition["person_id"] == person_id:
                    color = (0, 0, 255)  # Red for unauthorized transition
                    break
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw person ID and zone
            status_text = f"ID:{person_id}"
            if zone_name:
                status_text += f" ({zone_name})"
            
            draw_text_with_background(
                frame, status_text, (x1, y1 - 10),
                color=color, bg_alpha=0.7
            )
            
            # Draw time in zone if available
            if transition_data["entry_time"]:
                time_in_zone = current_time - transition_data["entry_time"]
                time_text = f"Time: {format_duration(time_in_zone)}"
                
                draw_text_with_background(
                    frame, time_text, (x1, y2 + 15),
                    font_scale=0.5, color=color, bg_alpha=0.7
                )
            
            # Draw path for persons who had unauthorized transitions
            path_person = False
            for transition in unauthorized_transitions:
                if transition["person_id"] == person_id:
                    path_person = True
                    break
            
            if path_person and self.alerts["unauthorized_entry"]["active"]:
                self._draw_person_path(frame, person_id)
        
        return frame, len(person_objects), unauthorized_transitions
    
    def _draw_person_path(self, frame, person_id):
        """
        Draw the path history of a person
        
        Args:
            frame: Video frame to draw on
            person_id: ID of the person whose path to draw
            
        Returns:
            Frame with path drawn
        """
        if person_id not in self.zone_transitions:
            return frame
        
        path = self.zone_transitions[person_id]["path"]
        if len(path) < 2:
            return frame
        
        # Get zone configurations for zone centers
        authorized_zones, unauthorized_zones, entry_points = self._get_zones_from_config()
        all_zones = []
        
        for zone in authorized_zones:
            all_zones.append(("authorized", zone))
        
        for zone in unauthorized_zones:
            all_zones.append(("unauthorized", zone))
        
        for zone in entry_points:
            all_zones.append(("entry_points", zone))
        
        # Draw connections between zone transitions
        for i in range(1, len(path)):
            prev_zone_type, prev_zone_name, prev_time = path[i-1]
            curr_zone_type, curr_zone_name, curr_time = path[i]
            
            # Find zone centers
            prev_center = None
            curr_center = None
            
            for zone_type, zone in all_zones:
                zone_name = zone[4] if len(zone) > 4 else zone_type
                if zone_type == prev_zone_type and zone_name == prev_zone_name:
                    prev_center = ((zone[0] + zone[2]) // 2, (zone[1] + zone[3]) // 2)
                
                if zone_type == curr_zone_type and zone_name == curr_zone_name:
                    curr_center = ((zone[0] + zone[2]) // 2, (zone[1] + zone[3]) // 2)
            
            # If we found both centers, draw the connection
            if prev_center and curr_center:
                # Change color based on transition type
                is_unauthorized = self._check_unauthorized_transition(
                    prev_zone_type, prev_zone_name, curr_zone_type, curr_zone_name
                )
                
                color = (0, 0, 255) if is_unauthorized else (255, 255, 255)
                
                # Draw arrow between zone centers
                cv2.arrowedLine(frame, prev_center, curr_center, color, 2, tipLength=0.2)
                
                # Draw timestamp at middle of arrow
                mid_x = (prev_center[0] + curr_center[0]) // 2
                mid_y = (prev_center[1] + curr_center[1]) // 2
                
                time_str = datetime.fromtimestamp(curr_time).strftime("%H:%M:%S")
                
                draw_text_with_background(
                    frame, time_str, (mid_x, mid_y),
                    font_scale=0.4, color=color, bg_alpha=0.7
                )
        
        return frame
    
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
        
        # Detect people and zone transitions
        result_frame, person_count, unauthorized_transitions = self.detect_people_and_transitions(
            result_frame, detection_results, pose_results
        )
        
        # Draw status and alerts
        result_frame = self._draw_status_and_alerts(result_frame)
        
        # Return processed frame and detection results
        detection_info = {
            "person_count": person_count,
            "unauthorized_transitions": len(unauthorized_transitions)
        }
        
        return result_frame, detection_info
    
    def _draw_zones(self, frame):
        """Draw monitoring zones on the frame"""
        zones = self.zone_manager.zones.get(self.camera_id, {})
        
        # Draw authorized zones
        if "authorized" in zones:
            for zone in zones["authorized"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Authorized")
                    draw_zone(frame, (*coords, name), self.zone_colors["authorized"], 0.2, name)
                    
                    # Draw occupancy if available
                    if name in self.stats["zone_occupancy"]:
                        occupancy = self.stats["zone_occupancy"][name]
                        draw_text_with_background(
                            frame, f"Occupancy: {occupancy}", 
                            (coords[0] + 10, coords[1] + 40),
                            font_scale=0.5, color=self.zone_colors["authorized"], bg_alpha=0.7
                        )
        
        # Draw unauthorized zones
        if "unauthorized" in zones:
            for zone in zones["unauthorized"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Unauthorized")
                    draw_zone(frame, (*coords, name), self.zone_colors["unauthorized"], 0.2, name)
                    
                    # Draw occupancy if available
                    if name in self.stats["zone_occupancy"]:
                        occupancy = self.stats["zone_occupancy"][name]
                        draw_text_with_background(
                            frame, f"Occupancy: {occupancy}", 
                            (coords[0] + 10, coords[1] + 40),
                            font_scale=0.5, color=self.zone_colors["unauthorized"], bg_alpha=0.7
                        )
        
        # Draw entry points
        if "entry_points" in zones:
            for zone in zones["entry_points"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Entry Point")
                    draw_zone(frame, (*coords, name), self.zone_colors["entry_points"], 0.2, name)
                    
                    # Draw occupancy if available
                    if name in self.stats["zone_occupancy"]:
                        occupancy = self.stats["zone_occupancy"][name]
                        draw_text_with_background(
                            frame, f"Occupancy: {occupancy}", 
                            (coords[0] + 10, coords[1] + 40),
                            font_scale=0.5, color=self.zone_colors["entry_points"], bg_alpha=0.7
                        )
        
        return frame
    
    def _draw_status_and_alerts(self, frame):
        """
        Draw status information and alerts on the frame
        
        Args:
            frame: Video frame to draw on
            
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
        
        # Draw statistics
        y_offset = 70
        
        # Unauthorized transitions
        color = (0, 0, 255) if self.stats["unauthorized_transitions"] > 0 else (255, 255, 255)
        draw_text_with_background(
            frame, f"Unauthorized transitions: {self.stats['unauthorized_transitions']}", 
            (10, y_offset), font_scale=0.7, color=color, bg_alpha=0.7
        )
        y_offset += 40
        
        # Total transitions
        draw_text_with_background(
            frame, f"Total transitions: {self.stats['total_transitions']}", 
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
    
    parser = argparse.ArgumentParser(description="Datacenter Zone Monitoring (Use Case 4)")
    parser.add_argument("--input", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video file")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--rules", type=str, default="rules.json", help="Path to rules file")
    parser.add_argument("--zones", type=str, default="zones.json", help="Path to zones file")
    parser.add_argument("--camera", type=str, default="datacenter_inside", help="Camera identifier in config")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame")
    
    args = parser.parse_args()
    
    # Create datacenter zone monitor
    monitor = DatacenterZoneMonitor(
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