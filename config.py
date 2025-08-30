"""
VisionGuard AI Configuration Settings
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ALERTS_DIR = DATA_DIR / "alerts"
FACES_DIR = DATA_DIR / "authorized_faces"
VIDEOS_DIR = DATA_DIR / "sample_videos"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, ALERTS_DIR, FACES_DIR, VIDEOS_DIR, LOGS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = 'gemini-1.5-pro'

# Video Processing Configuration
VIDEO_PROCESSING = {
    'supported_formats': ['.mp4', '.avi', '.mov', '.mkv', '.wmv'],
    'max_file_size_mb': 100,
    'frame_extraction_interval': 30,  # Extract every 30th frame
    'max_frames_to_process': 200,     # Limit for performance
    'resize_frame_width': 640,        # Resize for faster processing
}

# Face Recognition Configuration
FACE_RECOGNITION = {
    'similarity_threshold': 0.6,      # Face matching threshold
    'blur_factor': 15,                # Blur intensity for unauthorized faces
    'max_faces_per_frame': 10,        # Limit faces processed per frame
    'face_detection_model': 'hog',    # 'hog' or 'cnn' (cnn is more accurate but slower)
    'clustering_threshold': 0.4,      # Threshold for semi-supervised clustering
    'min_cluster_size': 3,            # Minimum faces to form a cluster
    'semi_supervised_learning': True, # Enable automatic identity clustering
}

# Alert Configuration
ALERTS = {
    'priority_levels': {
        1: {'label': 'Low', 'color': '#28a745', 'emoji': 'ℹ️'},
        2: {'label': 'Medium', 'color': '#17a2b8', 'emoji': '⚠️'},
        3: {'label': 'High', 'color': '#ffc107', 'emoji': '🚨'},
        4: {'label': 'Critical', 'color': '#fd7e14', 'emoji': '🔥'},
        5: {'label': 'Emergency', 'color': '#dc3545', 'emoji': '🚫'},
    },
    'auto_cleanup_days': 30,          # Auto-delete alerts older than 30 days
    'max_alerts_display': 100,        # Maximum alerts to show in dashboard
}

# YOLO Configuration
YOLO = {
    'model_path': MODELS_DIR / 'yolov8n.pt',  # YOLOv8 nano for speed
    'confidence_threshold': 0.4,
    'iou_threshold': 0.45,
    'classes_of_interest': [0, 1, 2, 3, 5, 7, 24, 28, 31],  # Person, bicycle, car, motorcycle, bus, truck, backpack, suitcase, handbag
    'campus_objects': {
        0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
        1: "bicycle", 24: "backpack", 67: "cell phone", 73: "laptop",
        28: "suitcase", 31: "handbag", 39: "bottle"
    }
}

# Campus Zone Configuration
CAMPUS_ZONES = {
    "main_gate": {
        "name": "Main Campus Gate",
        "allowed_objects": ["person", "car", "bicycle", "motorcycle"],
        "restricted_objects": ["truck", "bus"],
        "high_risk_hours": [(22, 6)],  # 10PM to 6AM
        "max_loitering_time": 180,
        "visitor_escort_required": False,
        "alert_multipliers": {
            "unauthorized_vehicle": 1.5,
            "after_hours": 2.0,
            "loitering": 1.2
        }
    },
    "construction_site": {
        "name": "Campus Construction Zone", 
        "allowed_objects": ["person", "truck", "car"],
        "restricted_objects": ["bicycle", "motorcycle"],
        "high_risk_hours": [(18, 7), (12, 13)],
        "max_loitering_time": 600,
        "visitor_escort_required": True,
        "alert_multipliers": {
            "no_safety_gear": 3.0,
            "unauthorized_access": 2.5,
            "restricted_vehicle": 2.0
        }
    },
    "library": {
        "name": "Academic Library",
        "allowed_objects": ["person", "backpack", "laptop"],
        "restricted_objects": ["bicycle", "car", "motorcycle"],
        "high_risk_hours": [(23, 6)],
        "max_loitering_time": 7200,
        "visitor_escort_required": False,
        "alert_multipliers": {
            "noise_disturbance": 1.5,
            "unauthorized_entry": 2.0
        }
    },
    "parking_lot": {
        "name": "Student Parking Area",
        "allowed_objects": ["person", "car", "bicycle", "motorcycle"],
        "restricted_objects": ["truck"],
        "high_risk_hours": [(22, 6)],
        "max_loitering_time": 300,
        "visitor_escort_required": False,
        "alert_multipliers": {
            "vehicle_break_in": 2.0,
            "unauthorized_parking": 1.2
        }
    },
    "dormitory": {
        "name": "Student Residence Hall",
        "allowed_objects": ["person"],
        "restricted_objects": ["car", "truck", "motorcycle"],
        "high_risk_hours": [(2, 6)],
        "max_loitering_time": 600,
        "visitor_escort_required": True,
        "alert_multipliers": {
            "unauthorized_entry": 3.0,
            "visitor_unescorted": 2.0
        }
    }
}

# Audio Anomaly Detection Configuration
AUDIO_DETECTION = {
    'enable_audio_analysis': True,
    'anomaly_types': [
        'glass_breaking', 'shouting', 'machinery_noise', 'vehicle_alarm',
        'loud_music', 'metal_clanging', 'running_footsteps', 'power_tool_noise',
        'door_slamming', 'emergency_siren'
    ],
    'zone_audio_patterns': {
        'main_gate': ['vehicle_alarm', 'shouting', 'metal_clanging'],
        'construction_site': ['machinery_noise', 'power_tool_noise', 'metal_clanging'],
        'library': ['loud_music', 'shouting', 'door_slamming'],
        'parking_lot': ['vehicle_alarm', 'glass_breaking', 'running_footsteps'],
        'dormitory': ['loud_music', 'shouting', 'door_slamming', 'glass_breaking']
    },
    'correlation_threshold': 0.5,  # Audio-visual correlation strength threshold
}

# Logging Configuration
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': LOGS_DIR / 'visionguard.log',
    'max_file_size_mb': 10,
    'backup_count': 5,
}

# Streamlit Configuration
STREAMLIT = {
    'page_title': 'VisionGuard AI - Smart Campus Security',
    'page_icon': '🛡️',
    'layout': 'wide',
    'theme': {
        'primary_color': '#1e3a8a',
        'background_color': '#ffffff',
        'secondary_background_color': '#f0f2f6',
        'text_color': '#262730',
    }
}

# Performance Configuration
PERFORMANCE = {
    'max_concurrent_video_processing': 2,
    'gemini_request_timeout': 30,
    'face_recognition_batch_size': 10,
    'alert_processing_batch_size': 5,
}

# Demo/Development Configuration
DEMO = {
    'enable_mock_detections': True,
    'mock_detection_probability': 0.3,  # 30% chance of generating mock detection
    'sample_authorized_faces': [
        'John_Doe.jpg',
        'Jane_Smith.jpg', 
        'Security_Guard.jpg',
        'Admin_Staff.jpg'
    ],
}

# Security Configuration
SECURITY = {
    'encrypt_face_embeddings': False,   # For production, consider encryption
    'audit_log_enabled': True,
    'max_login_attempts': 3,            # For future auth implementation
    'session_timeout_minutes': 60,
}

def get_config_summary():
    """Get a summary of current configuration for display."""
    return {
        'data_directory': str(DATA_DIR),
        'authorized_faces_dir': str(FACES_DIR),
        'alerts_directory': str(ALERTS_DIR),
        'gemini_configured': bool(GEMINI_API_KEY),
        'video_formats_supported': VIDEO_PROCESSING['supported_formats'],
        'face_threshold': FACE_RECOGNITION['similarity_threshold'],
        'clustering_enabled': FACE_RECOGNITION['semi_supervised_learning'],
        'yolo_model': YOLO['model_path'].name,
        'campus_zones': len(CAMPUS_ZONES),
        'audio_detection_enabled': AUDIO_DETECTION['enable_audio_analysis'],
        'max_file_size': f"{VIDEO_PROCESSING['max_file_size_mb']}MB",
    }

def validate_config():
    """Validate configuration and return any issues."""
    issues = []
    
    # Check API key
    if not GEMINI_API_KEY:
        issues.append("GEMINI_API_KEY not set in environment variables")
    
    # Check directories
    for name, path in [
        ('Data', DATA_DIR),
        ('Alerts', ALERTS_DIR), 
        ('Faces', FACES_DIR),
        ('Videos', VIDEOS_DIR),
        ('Logs', LOGS_DIR),
    ]:
        if not path.exists():
            issues.append(f"{name} directory does not exist: {path}")
        elif not path.is_dir():
            issues.append(f"{name} path is not a directory: {path}")
    
    # Validate zone configuration
    for zone_id, zone_config in CAMPUS_ZONES.items():
        if not zone_config.get('name'):
            issues.append(f"Zone {zone_id} missing name")
        if not zone_config.get('allowed_objects'):
            issues.append(f"Zone {zone_id} missing allowed_objects")
    
    # Validate YOLO configuration
    if not YOLO['model_path'].parent.exists():
        issues.append(f"YOLO models directory does not exist: {YOLO['model_path'].parent}")
    
    return issues

# Environment-specific overrides
if os.getenv('ENVIRONMENT') == 'development':
    VIDEO_PROCESSING['frame_extraction_interval'] = 60  # Slower for dev
    FACE_RECOGNITION['similarity_threshold'] = 0.5     # More lenient for testing
    DEMO['enable_mock_detections'] = True

elif os.getenv('ENVIRONMENT') == 'production':
    VIDEO_PROCESSING['frame_extraction_interval'] = 15  # More frequent
    FACE_RECOGNITION['similarity_threshold'] = 0.7     # More strict
    DEMO['enable_mock_detections'] = False
    LOGGING['level'] = 'WARNING'