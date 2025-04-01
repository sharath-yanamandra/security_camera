"""
Use Case 3: Datacenter Entry Monitoring
- Detection of unauthorized access to restricted zones
- Detection of emergency door button activation
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


class DatacenterEntryMonitor(VideoProcessor):
    """
    Monitor for datacenter entry area:
    - Unauthorized access to restricted zones
    - Emergency door button activation
    """
    
    def __init__(self, config_file="config.json", rules_file="rules.json", zones_file="zones.json", camera_id="datacenter_entry"):
        """
        Initialize the Datacenter Entry monitor
        
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
        device = self.config.get("device", "cpu")
        confidence = self.config.get("confidence", 0.5)
        
        self.detection_model = ModelLoader.load_model(detection_model, device)
        self.detection_model.conf = confidence
        
        # Load zones
        self.zone_manager = ZoneManager(zones_file)
        
        # Define zone colors
        self.zone_colors = {
            "authorized": (0, 255, 0),     # Green
            "unauthorized": (0, 0, 255),   # Red
            "emergency": (255, 0, 0)       # Red
        }
        
        # Create object tracker
        processing_config = self.config.get("processing", {})
        self.person_tracker = ObjectTracker(
            max_disappeared=processing_config.get("max_disappeared", 30),
            max_distance=processing_config.get("tracking_distance", 50)
        )
        
        # Initialize detection state
        self.unauthorized_access = set()  # Set of person IDs in unauthorized areas
        self.emergency_button_press = {}  # Dict to track emergency button presses
        
        # Get rule parameters
        rule_params = self.camera_rules.get("unauthorized_access", {}).get("parameters", {})
        self.min_overlap_ratio = rule_params.get("min_overlap_ratio", 0.5)
        
        rule_params = self.camera_rules.get("emergency_button", {}).get("parameters", {})
        self.emergency_detection_threshold = rule_params.get("detection_threshold", 0.7)
        self.button_pressed_duration = rule_params.get("button_pressed_duration", 1.0)
        
        # Session statistics
        self.stats = {
            "unauthorized_access_violations": 0,
            "emergency_button_activations": 0,
            "current_unauthorized_count": 0,
            "zone_entries": {}  # Dict mapping zone names to entry counts
        }
        
        # Initialize zone entry counts
        for zone_type in ["authorized", "unauthorized"]:
            zones = self.zone_manager.zones.get(self.camera_id, {}).get(zone_type, [])
            for zone in zones:
                if "name" in zone:
                    self.stats["zone_entries"][zone["name"]] = 0
        
        # Initialize alert state
        self.alerts = {
            "unauthorized_access": {
                "active": False,
                "start_time": None,
                "message": "ALERT: Unauthorized access detected!",
                "duration": 5.0
            },
            "emergency_button": {
                "active": False,
                "start_time": None,
                "message": "ALERT: Emergency button activated!",
                "duration": 5.0
            }
        }
        
        self.start_time = time.time()
        
        print(f"Datacenter entry monitoring initialized for camera: {camera_id}")
    
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
        
        emergency_zones = []
        if "emergency" in zones:
            for zone in zones["emergency"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Emergency Button")
                    emergency_zones.append((*coords, name))
        
        return authorized_zones, unauthorized_zones, emergency_zones
    
    def detect_people(self, frame, detection_results):
        """
        Detect and track people in the datacenter entry area
        Check for unauthorized access to restricted zones
        
        Args:
            frame: Current video frame
            detection_results: Results from detection model
            
        Returns:
            Tuple of (frame, person_count, unauthorized_count)
        """
        # Get zones
        authorized_zones, unauthorized_zones, _ = self._get_zones_from_config()
        
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
                    
                    # Add to people list
                    people.append({
                        "bbox": bbox,
                        "center": center,
                        "confidence": confidence
                    })
        
        # Update person tracker
        person_detections = [(p["bbox"], "person", p["confidence"]) for p in people]
        person_objects = self.person_tracker.update(person_detections)
        
        # Check for zone presence
        current_unauthorized = set()
        
        for person_id, person_data in person_objects.items():
            bbox = person_data["bbox"]
            center = person_data["centroid"]
            
            # Check if in authorized zone
            in_authorized = False
            authorized_zone_name = None
            
            for zone in authorized_zones:
                if self.zone_manager.is_in_zone(center, zone):
                    in_authorized = True
                    authorized_zone_name = zone[4] if len(zone) > 4 else "Authorized"
                    break
            
            # Check if in unauthorized zone
            in_unauthorized = False
            unauthorized_zone_name = None
            
            for zone in unauthorized_zones:
                overlap = self.zone_manager.object_zone_overlap(bbox, zone)
                if overlap >= self.min_overlap_ratio:
                    in_unauthorized = True
                    unauthorized_zone_name = zone[4] if len(zone) > 4 else "Unauthorized"
                    break
            
            # Update zone entry stats if this is a new entry
            if in_authorized and authorized_zone_name:
                if not hasattr(person_data, 'last_zone') or person_data.get('last_zone') != authorized_zone_name:
                    person_data['last_zone'] = authorized_zone_name
                    if authorized_zone_name in self.stats["zone_entries"]:
                        self.stats["zone_entries"][authorized_zone_name] += 1
            
            if in_unauthorized and unauthorized_zone_name:
                if not hasattr(person_data, 'last_zone') or person_data.get('last_zone') != unauthorized_zone_name:
                    person_data['last_zone'] = unauthorized_zone_name
                    if unauthorized_zone_name in self.stats["zone_entries"]:
                        self.stats["zone_entries"][unauthorized_zone_name] += 1
                
                # Add to current unauthorized set
                current_unauthorized.add(person_id)
                
                # Check if this is a new unauthorized access
                if person_id not in self.unauthorized_access:
                    self.unauthorized_access.add(person_id)
                    self.stats["unauthorized_access_violations"] += 1
                    
                    # Update alert state
                    if not self.alerts["unauthorized_access"]["active"]:
                        self.alerts["unauthorized_access"]["active"] = True
                        self.alerts["unauthorized_access"]["start_time"] = time.time()
                        
                        # Save event screenshot
                        if self.config.get("events", {}).get("save_screenshots", True):
                            event_metadata = create_event_metadata(
                                frame, "unauthorized_access", person_id, unauthorized_zone_name, 
                                person_data["confidence"], {"overlap": overlap}
                            )
                            self.save_screenshot(frame, "unauthorized_access")
        
        # Update unauthorized count
        self.stats["current_unauthorized_count"] = len(current_unauthorized)
        
        # Visualize people with tracking info
        for person_id, person_data in person_objects.items():
            bbox = person_data["bbox"]
            x1, y1, x2, y2 = bbox
            center = person_data["centroid"]
            
            # Determine if in a specific zone
            in_authorized = False
            in_unauthorized = False
            zone_name = None
            
            for zone in authorized_zones:
                if self.zone_manager.is_in_zone(center, zone):
                    in_authorized = True
                    zone_name = zone[4] if len(zone) > 4 else "Authorized"
                    break
                    
            for zone in unauthorized_zones:
                overlap = self.zone_manager.object_zone_overlap(bbox, zone)
                if overlap >= self.min_overlap_ratio:
                    in_unauthorized = True
                    zone_name = zone[4] if len(zone) > 4 else "Unauthorized"
                    break
            
            # Determine color based on location
            if in_unauthorized or person_id in self.unauthorized_access:
                color = (0, 0, 255)  # Red for unauthorized access
            elif in_authorized:
                color = (0, 255, 0)  # Green for authorized zone
            else:
                color = (255, 255, 255)  # White for outside areas
                
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw person ID and status
            status_text = f"ID:{person_id}"
            if zone_name:
                status_text += f" ({zone_name})"
            
            draw_text_with_background(
                frame, status_text, (x1, y1 - 10),
                color=color, bg_alpha=0.7
            )
        
        return frame, len(person_objects), len(current_unauthorized)
    
    def detect_emergency_button(self, frame, detection_results):
        """
        Detect interaction with emergency door button
        
        Args:
            frame: Current video frame
            detection_results: Results from detection model
            
        Returns:
            Tuple of (frame, button_pressed)
        """
        # Get emergency button zones
        _, _, emergency_zones = self._get_zones_from_config()
        
        if not emergency_zones:
            return frame, False
        
        button_pressed = False
        
        # Look for people near emergency buttons
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
                    
                    # Get center and hand points (approximated)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Approximate right hand position (25% from right edge, 40% from top)
                    right_hand = (int(x1 + (x2 - x1) * 0.75), int(y1 + (y2 - y1) * 0.4))
                    
                    # Approximate left hand position (25% from left edge, 40% from top)
                    left_hand = (int(x1 + (x2 - x1) * 0.25), int(y1 + (y2 - y1) * 0.4))
                    
                    # Check hand positions against emergency button zones
                    for zone in emergency_zones:
                        # Check if either hand is in an emergency button zone
                        right_in_button = self.zone_manager.is_in_zone(right_hand, zone)
                        left_in_button = self.zone_manager.is_in_zone(left_hand, zone)
                        
                        if right_in_button or left_in_button:
                            button_name = zone[4] if len(zone) > 4 else "Emergency Button"
                            
                            # Track button press
                            if button_name not in self.emergency_button_press:
                                self.emergency_button_press[button_name] = {
                                    "start_time": time.time(),
                                    "active": True
                                }
                            else:
                                # Check if pressed long enough
                                press_duration = time.time() - self.emergency_button_press[button_name]["start_time"]
                                
                                if press_duration >= self.button_pressed_duration and self.emergency_button_press[button_name]["active"]:
                                    # Button activated!
                                    button_pressed = True
                                    self.emergency_button_press[button_name]["active"] = False  # Prevent multiple activations
                                    self.stats["emergency_button_activations"] += 1
                                    
                                    # Update alert state
                                    if not self.alerts["emergency_button"]["active"]:
                                        self.alerts["emergency_button"]["active"] = True
                                        self.alerts["emergency_button"]["start_time"] = time.time()
                                        self.alerts["emergency_button"]["message"] = f"ALERT: {button_name} activated!"
                                        
                                        # Save event screenshot
                                        if self.config.get("events", {}).get("save_screenshots", True):
                                            event_metadata = create_event_metadata(
                                                frame, "emergency_button", None, button_name, 
                                                1.0, {"duration": press_duration}
                                            )
                                            self.save_screenshot(frame, "emergency_button")
                            
                            # Draw hand position and interaction
                            hand_point = right_hand if right_in_button else left_hand
                            cv2.circle(frame, hand_point, 5, (0, 0, 255), -1)
                            
                            # Draw line from hand to button center
                            button_center = (
                                (zone[0] + zone[2]) // 2,
                                (zone[1] + zone[3]) // 2
                            )
                            cv2.line(frame, hand_point, button_center, (0, 0, 255), 2)
                            
                            # Draw button interaction text
                            if button_name in self.emergency_button_press:
                                press_duration = time.time() - self.emergency_button_press[button_name]["start_time"]
                                button_text = f"Button press: {press_duration:.1f}s"
                                draw_text_with_background(
                                    frame, button_text, (zone[0], zone[1] - 10),
                                    color=(0, 0, 255), bg_alpha=0.7
                                )
        
        # Reset emergency button state if no interaction for a while
        buttons_to_reset = []
        current_time = time.time()
        
        for button_name, button_data in self.emergency_button_press.items():
            if current_time - button_data["start_time"] > 2.0:  # Reset after 2 seconds of no interaction
                buttons_to_reset.append(button_name)
        
        for button_name in buttons_to_reset:
            self.emergency_button_press[button_name]["active"] = True  # Allow new activations
            if current_time - self.emergency_button_press[button_name]["start_time"] > 5.0:  # Remove after 5 seconds
                del self.emergency_button_press[button_name]
                
        return frame, button_pressed
    
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
        
        # Detect people and unauthorized access
        result_frame, person_count, unauthorized_count = self.detect_people(
            result_frame, detection_results
        )
        
        # Detect emergency button activation
        result_frame, button_pressed = self.detect_emergency_button(
            result_frame, detection_results
        )
        
        # Draw status and alerts
        result_frame = self._draw_status_and_alerts(result_frame)
        
        # Return processed frame and detection results
        detection_info = {
            "person_count": person_count,
            "unauthorized_count": unauthorized_count,
            "button_pressed": button_pressed
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
                    draw_zone(frame, (*coords, name), self.zone_colors["authorized"], 0.3, name)
        
        # Draw unauthorized zones
        if "unauthorized" in zones:
            for zone in zones["unauthorized"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Unauthorized")
                    draw_zone(frame, (*coords, name), self.zone_colors["unauthorized"], 0.3, name)
        
        # Draw emergency zones
        if "emergency" in zones:
            for zone in zones["emergency"]:
                if "coordinates" in zone:
                    coords = zone["coordinates"]
                    name = zone.get("name", "Emergency Button")
                    draw_zone(frame, (*coords, name), self.zone_colors["emergency"], 0.3, name)
        
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
        
        # Draw violation counts
        y_offset = 70
        
        # Unauthorized access violations
        color = (0, 0, 255) if self.stats["unauthorized_access_violations"] > 0 else (255, 255, 255)
        draw_text_with_background(
            frame, f"Unauthorized access violations: {self.stats['unauthorized_access_violations']}", 
            (10, y_offset), font_scale=0.7, color=color, bg_alpha=0.7
        )
        y_offset += 40
        
        # Current unauthorized count
        color = (0, 0, 255) if self.stats["current_unauthorized_count"] > 0 else (255, 255, 255)
        draw_text_with_background(
            frame, f"Current unauthorized count: {self.stats['current_unauthorized_count']}", 
            (10, y_offset), font_scale=0.7, color=color, bg_alpha=0.7
        )
        y_offset += 40
        
        # Emergency button activations
        color = (0, 0, 255) if self.stats["emergency_button_activations"] > 0 else (255, 255, 255)
        draw_text_with_background(
            frame, f"Emergency button activations: {self.stats['emergency_button_activations']}", 
            (10, y_offset), font_scale=0.7, color=color, bg_alpha=0.7
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
    
    parser = argparse.ArgumentParser(description="Datacenter Entry Monitoring (Use Case 3)")
    parser.add_argument("--input", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video file")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--rules", type=str, default="rules.json", help="Path to rules file")
    parser.add_argument("--zones", type=str, default="zones.json", help="Path to zones file")
    parser.add_argument("--camera", type=str, default="datacenter_entry", help="Camera identifier in config")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame")
    
    args = parser.parse_args()
    
    # Create datacenter entry monitor
    monitor = DatacenterEntryMonitor(
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