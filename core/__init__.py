"""
VisionGuard AI Core Module

This module contains the core functionality for the VisionGuard AI system:
- Video processing and frame extraction
- Face verification and semi-supervised learning
- Object detection with YOLOv8
- Audio anomaly detection
- Gemini AI integration
- Alert management
"""

from .video_processor import VideoProcessor
from .face_verification import FaceVerificationSystem
from .detection_engine import ObjectDetectionEngine, ZoneContextManager
from .audio_anomaly_detector import AudioAnomalyDetector
from .gemini_client import GeminiSecurityAnalyzer
from .alert_manager import AlertManager

__version__ = "1.0.0"
__author__ = "VisionGuard AI Team"

# Core components available for import
__all__ = [
    "VideoProcessor",
    "FaceVerificationSystem", 
    "ObjectDetectionEngine",
    "ZoneContextManager",
    "AudioAnomalyDetector",
    "GeminiSecurityAnalyzer",
    "AlertManager"
]