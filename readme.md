# Security Camera Monitoring System

A modular system for security camera monitoring with multiple use cases:

1. **Entry Monitoring**: Detection of people sitting on stairs, gatherings, and vehicle entry (with truck parking restrictions)
2. **Reception Monitoring**: Enforcement of maximum occupancy limits and detection of gate jumping attempts
3. **Datacenter Entry Monitoring**: Detection of unauthorized access to restricted zones and emergency button activations
4. **Datacenter Zone Monitoring**: Tracking of people moving between zones with focus on unauthorized transitions
5. **Parking Monitoring**: Detection of vehicles parked outside designated zones, exit blocking, and suspicious activities

## Features

- Real-time video processing with object detection and tracking
- Zone-based monitoring with configurable rules
- Violation detection and alert generation
- Automated screenshot capture for security events
- Integrated dashboard with real-time statistics
- Parallel or sequential processing of multiple camera feeds
- Modular architecture for easy extension

## Installation

### Prerequisites

- Python 3.8+
- OpenCV 4.5+
- PyTorch 1.9+
- CUDA support (optional, for GPU acceleration)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/security-monitoring.git
   cd security-monitoring
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the YOLOv8 models (automatically downloaded on first run, or manually):
   ```
   # Detection model
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
   
   # Pose model
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt
   ```

## Configuration

The system uses three main configuration files:

1. **config.json**: General system configuration including model paths, camera settings, etc.
2. **zones.json**: Zone definitions for each camera and use case
3. **rules.json**: Rules and thresholds for event detection

Modify these files to match your environment and requirements.

## Usage

### Running a Single Use Case

```bash
# Run entry monitoring
python entry_monitoring.py --input path/to/video.mp4 --output results/entry_output.mp4

# Run reception monitoring
python reception_monitoring.py --input path/to/video.mp4 --output results/reception_output.mp4

# Run datacenter entry monitoring
python datacenter_entry_monitoring.py --input path/to/video.mp4 --output results/dc_entry_output.mp4

# Run datacenter zone monitoring
python datacenter_zone_monitoring.py --input path/to/video.mp4 --output results/dc_zone_output.mp4

# Run parking monitoring
python parking_monitoring.py --input path/to/video.mp4 --output results/parking_output.mp4
```

### Running All Use Cases Together

```bash
# Run all use cases with different video inputs
python main.py --input1 entry.mp4 --input2 reception.mp4 --input3 dc_entry.mp4 --input4 dc_inside.mp4 --input5 parking.mp4 --output-dir results/

# Run specific use case
python main.py --use-case 2 --input2 reception.mp4 --output-dir results/

# Run with parallel processing
python main.py --all --parallel --output-dir results/
```

### Advanced Usage

```bash
# Use GPU acceleration
python main.py --all --config gpu_config.json --output-dir results/

# Define custom zones
python datacenter_entry_monitoring.py --input video.mp4 --zones custom_zones.json

# Apply specific rules
python parking_monitoring.py --input video.mp4 --rules custom_rules.json
```

## Project Structure

```
security-monitoring/
├── config.json              # Main configuration file
├── rules.json               # Event detection rules
├── zones.json               # Zone definitions
├── main.py                  # Main system controller
├── utils.py                 # Common utilities
├── entry_monitoring.py      # Use Case 1: Entry monitoring
├── reception_monitoring.py  # Use Case 2: Reception monitoring
├── datacenter_entry_monitoring.py  # Use Case 3: DC entry monitoring
├── datacenter_zone_monitoring.py   # Use Case 4: DC zone monitoring
├── parking_monitoring.py    # Use Case 5: Parking monitoring
├── outputs/                 # Generated output videos and screenshots
├── models/                  # Pretrained ML models
└── README.md                # Project documentation
```

## Future Work

- Database integration with the provided schema
- Web interface for real-time monitoring
- Alert notification system (email, SMS, etc.)
- Additional detection scenarios and use cases
- Integration with physical security systems

## License

This project is licensed under the MIT License - see the LICENSE file for details.
