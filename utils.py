"""
Common utilities for the security camera monitoring system.
Provides functions for object detection, tracking, and visualization used across all use cases.
"""
import cv2
import numpy as np
import torch
import time
import os
from pathlib import Path
import json
from datetime import datetime
from ultralytics import YOLO

# Common constants
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Vehicle classes from COCO
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 8: 'boat'}

class VideoProcessor:
    """Base class for processing video streams"""
    
    def __init__(self, output_dir='outputs'):
        """
        Initialize the video processor
        
        Args:
            output_dir: Directory where output videos and screenshots will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.start_time = time.time()
        
    def process_video(self, video_path, output_path=None, skip_frames=1):
        """
        Process video file with frame skipping for better performance
        
        Args:
            video_path: Path to input video file
            output_path: Path to output video file (if None, no output is saved)
            skip_frames: Process every Nth frame for better performance
        
        Returns:
            Number of frames processed
        """
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return 0
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if output path is specified
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        result_frame = None
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                print(f"Processing frame {frame_count}", end='\r')
                
                # Process frame based on skip_frames ratio
                if frame_count % skip_frames == 0:
                    # This method should be implemented by derived classes
                    if hasattr(self, 'process_frame'):
                        result_frame, _ = self.process_frame(frame)
                    else:
                        result_frame = frame
                
                    # Write frame to output video
                    if writer:
                        writer.write(result_frame)
                    
                    # Display frame (optional)
                    cv2.imshow(self.__class__.__name__, result_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        break
                elif writer and result_frame is not None:
                    # For skipped frames, just write the last processed frame
                    writer.write(result_frame)
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        finally:
            # Release resources
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"\nProcessed {frame_count} frames")
            return frame_count
    
    def save_screenshot(self, frame, event_type):
        """
        Save a screenshot when an event is detected
        
        Args:
            frame: The frame to save
            event_type: Type of event (used in filename)
            
        Returns:
            Path to saved screenshot
        """
        screenshots_dir = self.output_dir / "screenshots"
        screenshots_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{event_type}_{timestamp}.jpg"
        filepath = screenshots_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"Screenshot saved: {filepath}")
        return filepath


class ObjectTracker:
    """Common tracking functionality for detected objects"""
    
    def __init__(self, max_disappeared=10, max_distance=50):
        """
        Initialize object tracker
        
        Args:
            max_disappeared: Maximum frames an object can disappear before being removed
            max_distance: Maximum pixel distance for centroid matching
        """
        self.next_object_id = 0
        self.objects = {}  # Dictionary: ID -> object data
        self.disappeared = {}  # Dictionary: ID -> frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid, bbox, obj_type, confidence=1.0):
        """
        Register a new object
        
        Args:
            centroid: (x, y) center point of object
            bbox: (x1, y1, x2, y2) bounding box
            obj_type: Type of object (e.g., "person", "car")
            confidence: Detection confidence
            
        Returns:
            ID of the registered object
        """
        object_id = self.next_object_id
        self.objects[object_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'type': obj_type,
            'confidence': confidence,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'positions': [centroid],
            'history': [(bbox, time.time())],
        }
        self.disappeared[object_id] = 0
        self.next_object_id += 1
        return object_id
    
    def deregister(self, object_id):
        """
        Remove an object from tracking
        
        Args:
            object_id: ID of object to remove
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        """
        Update tracked objects with new detections
        
        Args:
            detections: List of (bbox, type, confidence) tuples
            
        Returns:
            Dictionary mapping object IDs to updated object data
        """
        # If no objects are being tracked, register all detections
        if len(self.objects) == 0:
            for bbox, obj_type, confidence in detections:
                x1, y1, x2, y2 = bbox
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                self.register(centroid, bbox, obj_type, confidence)
        
        # If no detections, increment disappeared counter for all objects
        elif len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        # Otherwise, match existing objects with new detections
        else:
            # Get centroids of current objects
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[object_id]['centroid'] for object_id in object_ids]
            
            # Calculate centroids of new detections
            detection_centroids = []
            for bbox, _, _ in detections:
                x1, y1, x2, y2 = bbox
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                detection_centroids.append(centroid)
            
            # Calculate distance matrix between existing objects and new detections
            distances = np.zeros((len(object_centroids), len(detection_centroids)))
            for i, object_centroid in enumerate(object_centroids):
                for j, detection_centroid in enumerate(detection_centroids):
                    d = np.sqrt(
                        (object_centroid[0] - detection_centroid[0])**2 + 
                        (object_centroid[1] - detection_centroid[1])**2
                    )
                    distances[i, j] = d
            
            # Use the Hungarian algorithm to solve the assignment problem
            from scipy.optimize import linear_sum_assignment
            matched_rows, matched_cols = linear_sum_assignment(distances)
            
            # Keep track of which objects and detections we've already handled
            used_object_indices = set()
            used_detection_indices = set()
            
            # Loop through all matched pairs
            for (row_idx, col_idx) in zip(matched_rows, matched_cols):
                # If the distance exceeds our max distance, this isn't a match
                if distances[row_idx, col_idx] > self.max_distance:
                    continue
                
                # Get the object ID of the matched object
                object_id = object_ids[row_idx]
                
                # Get the detection details of the matched detection
                bbox, obj_type, confidence = detections[col_idx]
                centroid = detection_centroids[col_idx]
                
                # Don't match objects of different types (e.g., don't match a person with a car)
                if obj_type != self.objects[object_id]['type']:
                    continue
                
                # Update the object's data
                self.objects[object_id]['centroid'] = centroid
                self.objects[object_id]['bbox'] = bbox
                self.objects[object_id]['confidence'] = confidence
                self.objects[object_id]['last_seen'] = time.time()
                self.objects[object_id]['positions'].append(centroid)
                self.objects[object_id]['history'].append((bbox, time.time()))
                
                # Limit history length
                if len(self.objects[object_id]['positions']) > 30:
                    self.objects[object_id]['positions'].pop(0)
                if len(self.objects[object_id]['history']) > 30:
                    self.objects[object_id]['history'].pop(0)
                
                # Reset the disappeared counter
                self.disappeared[object_id] = 0
                
                # Mark this object and detection as used
                used_object_indices.add(row_idx)
                used_detection_indices.add(col_idx)
            
            # Handle unmatched objects
            unused_object_rows = set(range(len(object_centroids))) - used_object_indices
            for row_idx in unused_object_rows:
                object_id = object_ids[row_idx]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Handle unmatched detections
            unused_detection_cols = set(range(len(detection_centroids))) - used_detection_indices
            for col_idx in unused_detection_cols:
                bbox, obj_type, confidence = detections[col_idx]
                centroid = detection_centroids[col_idx]
                self.register(centroid, bbox, obj_type, confidence)
        
        # Return updated objects
        return self.objects


class ModelLoader:
    """Utility for loading and caching ML models"""
    
    _model_cache = {}
    
    @classmethod
    def load_model(cls, model_path, device=None):
        """
        Load a YOLOv8 model with caching
        
        Args:
            model_path: Path to model weights
            device: Device to run on ('cuda' or 'cpu')
            
        Returns:
            YOLO model instance
        """
        # Default device selection
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Use cached model if available
        cache_key = f"{model_path}_{device}"
        if cache_key in cls._model_cache:
            return cls._model_cache[cache_key]
        
        # Ensure model file exists
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found locally. Will use default from Ultralytics.")
        
        # Load the model
        model = YOLO(model_path)
        
        # Cache the model
        cls._model_cache[cache_key] = model
        return model


class ZoneManager:
    """Manage detection zones and check for presence in zones"""
    
    def __init__(self, zones_file=None):
        """
        Initialize zone manager
        
        Args:
            zones_file: Path to zones.json file
        """
        self.zones = {}
        if zones_file:
            self.load_zones(zones_file)
    
    def load_zones(self, zones_file):
        """
        Load zones from a JSON file
        
        Args:
            zones_file: Path to zones.json file
        """
        try:
            with open(zones_file, 'r') as f:
                zones_data = json.load(f)
                
            self.zones = {}
            for camera_id, camera_zones in zones_data.items():
                self.zones[camera_id] = {}
                for zone_type, zones in camera_zones.items():
                    self.zones[camera_id][zone_type] = zones
                    
            print(f"Loaded {sum(len(zones) for camera in self.zones.values() for zones in camera.values())} zones from {zones_file}")
        except Exception as e:
            print(f"Error loading zones from {zones_file}: {e}")
    
    def is_in_zone(self, point, zone):
        """
        Check if a point is in a rectangular zone
        
        Args:
            point: (x, y) coordinates
            zone: (x1, y1, x2, y2, [name]) zone definition
            
        Returns:
            Boolean indicating if point is in zone
        """
        x, y = point
        x1, y1, x2, y2 = zone[:4]
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def object_zone_overlap(self, bbox, zone):
        """
        Calculate the overlap percentage between object and zone
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            zone: (x1, y1, x2, y2, [name]) zone definition
            
        Returns:
            Overlap percentage (0.0 to 1.0)
        """
        # Calculate box area
        x1, y1, x2, y2 = bbox
        box_area = (x2 - x1) * (y2 - y1)
        
        # Calculate zone area
        zx1, zy1, zx2, zy2 = zone[:4]
        
        # Calculate overlap coordinates
        overlap_x1 = max(x1, zx1)
        overlap_y1 = max(y1, zy1)
        overlap_x2 = min(x2, zx2)
        overlap_y2 = min(y2, zy2)
        
        # Check if there's overlap
        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            return 0.0
        
        # Calculate overlap area
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        
        # Return overlap percentage relative to the box
        return overlap_area / box_area


def format_duration(seconds):
    """
    Format seconds into a human-readable duration string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string like "5m 30s" or "45s"
    """
    if seconds is None:
        return "N/A"
    
    minutes, seconds = divmod(int(seconds), 60)
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def draw_text_with_background(frame, text, position, font_scale=0.5, color=(255, 255, 255), 
                             thickness=1, bg_color=(0, 0, 0), bg_alpha=0.5, padding=5):
    """
    Draw text with a semi-transparent background
    
    Args:
        frame: Image to draw on
        text: Text to display
        position: (x, y) coordinates
        font_scale: Font scale
        color: Text color
        thickness: Line thickness
        bg_color: Background color
        bg_alpha: Background transparency (0-1)
        padding: Padding around text
        
    Returns:
        Frame with text drawn
    """
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # Calculate background rectangle
    x, y = position
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + padding
    
    # Create overlay for background
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    # Blend with original frame
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    
    # Draw text
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    return frame


def configure_models(config_file):
    """
    Load model configuration from config.json
    
    Args:
        config_file: Path to config.json
        
    Returns:
        Dictionary with model paths and settings
    """
    default_config = {
        "models": {
            "detection": "yolov8l.pt",
            "pose": "yolov8l-pose.pt"
        },
        "device": "cpu",
        "confidence": 0.5,
        "output_dir": "outputs"
    }
    
    if not os.path.exists(config_file):
        print(f"Config file {config_file} not found, using defaults.")
        return default_config
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        print(f"Error loading config from {config_file}: {e}")
        return default_config


def load_rules(rules_file):
    """
    Load detection rules from rules.json
    
    Args:
        rules_file: Path to rules.json
        
    Returns:
        Dictionary of rules by camera and event type
    """
    default_rules = {
        "default": {
            "max_people": 10,
            "sitting_on_stairs": True,
            "gathering_threshold": 3,
            "parked_vehicle_time": 10,
            "movement_threshold": 5
        }
    }
    
    if not os.path.exists(rules_file):
        print(f"Rules file {rules_file} not found, using defaults.")
        return default_rules
    
    try:
        with open(rules_file, 'r') as f:
            rules = json.load(f)
        print(f"Loaded rules from {rules_file}")
        return rules
    except Exception as e:
        print(f"Error loading rules from {rules_file}: {e}")
        return default_rules


def create_event_metadata(frame, event_type, object_id=None, zone_name=None, confidence=1.0, additional_info=None):
    """
    Create metadata for event logging
    
    Args:
        frame: The frame where the event occurred
        event_type: Type of event
        object_id: ID of the object causing the event
        zone_name: Name of the zone where the event occurred
        confidence: Confidence score
        additional_info: Any additional information
        
    Returns:
        Dictionary with event metadata
    """
    timestamp = datetime.now().isoformat()
    
    metadata = {
        "event_type": event_type,
        "timestamp": timestamp,
        "object_id": object_id,
        "zone_name": zone_name,
        "confidence": confidence,
        "frame_dimensions": (frame.shape[1], frame.shape[0]),  # width, height
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    return metadata


def draw_zone(frame, zone, color, alpha=0.3, label=None):
    """
    Draw a zone on the frame with transparency
    
    Args:
        frame: Image to draw on
        zone: (x1, y1, x2, y2, [name]) zone definition
        color: RGB color tuple
        alpha: Transparency (0-1)
        label: Optional label to show
        
    Returns:
        Frame with zone drawn
    """
    x1, y1, x2, y2 = zone[:4]
    name = zone[4] if len(zone) > 4 else label or ""
    
    # Draw with semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label if provided
    if name:
        draw_text_with_background(frame, name, (x1 + 10, y1 + 20), 
                                 font_scale=0.6, color=color, bg_alpha=0.5)
    
    return frame


def is_object_stationary(object_data, time_threshold=5.0, movement_threshold=10.0):
    """
    Determine if an object has been stationary for a period of time
    
    Args:
        object_data: Object tracking data containing position history
        time_threshold: Time (in seconds) required to be considered stationary
        movement_threshold: Movement threshold (in pixels) below which object is considered stationary
        
    Returns:
        Tuple: (is_stationary, duration_stationary)
    """
    history = object_data.get('history', [])
    
    # Need at least 2 history points
    if len(history) < 2:
        return False, 0
    
    # Check time span of history
    current_time = time.time()
    oldest_time = history[0][1]
    newest_time = history[-1][1]
    
    # Calculate maximum distance moved in recent history
    max_distance = 0
    for i in range(1, len(history)):
        prev_box, _ = history[i-1]
        curr_box, _ = history[i]
        
        prev_center = ((prev_box[0] + prev_box[2]) // 2, (prev_box[1] + prev_box[3]) // 2)
        curr_center = ((curr_box[0] + curr_box[2]) // 2, (curr_box[1] + curr_box[3]) // 2)
        
        distance = np.sqrt((prev_center[0] - curr_center[0])**2 + (prev_center[1] - curr_center[1])**2)
        max_distance = max(max_distance, distance)
    
    # Check if stationary
    is_stationary = (newest_time - oldest_time >= time_threshold) and (max_distance <= movement_threshold)
    
    # Calculate duration
    duration = current_time - oldest_time if is_stationary else 0
    
    return is_stationary, duration
