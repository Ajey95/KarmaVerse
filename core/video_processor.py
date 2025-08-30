import cv2
import numpy as np
from datetime import datetime, timezone
import random
from typing import List, Dict, Optional, Tuple
import logging
from .face_verification import FaceVerificationSystem
from .detection_engine import ObjectDetectionEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video processing, frame extraction, face verification, object detection, and zone-aware event generation."""
    
    def __init__(self, camera_id: str = "Gate3"):
        self.camera_id = camera_id
        self.detection_types = [
            "loitering", "intrusion", "abandoned_object", 
            "unauthorized_entry", "crowd_formation", "suspicious_behavior",
            "unauthorized_vehicle_entry", "construction_safety_check", 
            "safety_violation", "visitor_entry", "after_hours_intrusion"
        ]
        
        # Campus locations mapped to zones
        self.campus_locations = {
            "Main Gate": "main_gate",
            "Construction Zone": "construction_site", 
            "Library": "library",
            "Parking Lot": "parking_lot",
            "Student Dormitory": "dormitory",
            "Admin Building": "library",  # Use library rules
            "Science Building": "library"
        }
        
        # Initialize detection systems
        self.face_verifier = FaceVerificationSystem()
        self.object_detector = ObjectDetectionEngine()
        
    def process_video_with_enhanced_detection(self, video_path: str, nth_frame: int = 30) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Process video with enhanced object detection, face verification, and zone-aware analysis.
        
        Args:
            video_path: Path to the video file
            nth_frame: Extract every nth frame
            
        Returns:
            Tuple of (processed_frames, detection_events)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            processed_frames = []
            detection_events = []
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Processing video with enhanced detection (YOLO + Face + Zone analysis)...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % nth_frame == 0:
                    # Simulate different campus locations for demo
                    current_location = random.choice(list(self.campus_locations.keys()))
                    zone_id = self.campus_locations[current_location]
                    
                    # Object detection
                    object_detections = self.object_detector.detect_objects(frame)
                    
                    # Face verification with location context
                    processed_frame, face_info = self.face_verifier.detect_and_verify_faces(
                        frame, 
                        location=current_location, 
                        timestamp=datetime.now(timezone.utc).isoformat()
                    )
                    
                    # Draw object detection boxes
                    if object_detections:
                        processed_frame = self.object_detector.draw_detections(processed_frame, object_detections)
                    
                    processed_frames.append(processed_frame)
                    
                    # Generate zone-aware detection event
                    timestamp = datetime.now(timezone.utc).isoformat()
                    time_seconds = frame_count / fps if fps > 0 else frame_count
                    
                    # Ensure face_info is properly formatted
                    if isinstance(face_info, list):
                        face_summary = self.face_verifier.get_face_match_summary(face_info)
                    else:
                        # Handle single face_info dict
                        face_summary = self.face_verifier.get_face_match_summary([face_info] if face_info else [])
                    
                    detection_event = self.object_detector.generate_zone_aware_event(
                        object_detections, 
                        face_summary,
                        zone_id,
                        current_location,
                        timestamp
                    )
                    
                    # Add video metadata
                    detection_event.update({
                        "video_time_seconds": time_seconds,
                        "frame_number": frame_count,
                        "processing_method": "enhanced_detection"
                    })
                    
                    detection_events.append(detection_event)
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Enhanced processing complete: {len(processed_frames)} frames, {len(detection_events)} events")
            return processed_frames, detection_events
            
        except Exception as e:
            logger.error(f"Error in enhanced video processing: {e}")
            return [], []
        
    def extract_frames(self, video_path: str, nth_frame: int = 30) -> List[np.ndarray]:
        """
        Extract every nth frame from video for processing.
        
        Args:
            video_path: Path to the video file
            nth_frame: Extract every nth frame (default: 30 = ~1 frame per second for 30fps video)
            
        Returns:
            List of extracted frames as numpy arrays
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
            logger.info(f"Extracting every {nth_frame}th frame")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % nth_frame == 0:
                    frames.append(frame)
                    logger.debug(f"Extracted frame {frame_count}")
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from {total_frames} total frames")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
    
    def process_video_with_face_verification(self, video_path: str, nth_frame: int = 30) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Legacy method - now redirects to enhanced detection.
        Kept for backwards compatibility.
        """
        return self.process_video_with_enhanced_detection(video_path, nth_frame)
    
    def generate_detection_from_faces(self, face_info: List[Dict], timestamp: str, 
                                    time_seconds: float, frame_number: int) -> Optional[Dict]:
        """
        Legacy method - Generate detection events based on face verification results.
        Now uses enhanced detection engine.
        """
        if not face_info:
            return None
        
        face_summary = self.face_verifier.get_face_match_summary(face_info)
        
        # Use object detector for enhanced event generation
        return self.object_detector.generate_zone_aware_event(
            [],  # No object detections in legacy mode
            face_summary,
            "main_gate",  # Default zone
            random.choice(list(self.campus_locations.keys())),
            timestamp
        )
    
    def generate_mock_detection(self, timestamp: str = None, time_seconds: float = None) -> Dict:
        """
        Generate mock detection data for testing purposes.
        
        Args:
            timestamp: Optional timestamp, will generate current time if None
            time_seconds: Time in seconds from video start
            
        Returns:
            Mock detection dictionary
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        
        # Mock behavior patterns
        behavior_patterns = {
            "loitering": {
                "descriptions": [
                    "Person stood at gate for 12 minutes",
                    "Individual lingering in parking area for extended period",
                    "Person pacing back and forth near entrance"
                ],
                "confidence_range": (0.85, 0.95)
            },
            "intrusion": {
                "descriptions": [
                    "Person climbing over fence",
                    "Unauthorized entry through side gate",
                    "Individual bypassing security checkpoint"
                ],
                "confidence_range": (0.90, 0.98)
            },
            "abandoned_object": {
                "descriptions": [
                    "Unattended bag left near entrance",
                    "Suspicious package detected in parking area",
                    "Object left behind after person departed"
                ],
                "confidence_range": (0.80, 0.92)
            },
            "crowd_formation": {
                "descriptions": [
                    "Large group gathering detected",
                    "Unusual crowd density near main gate",
                    "Multiple people congregating in restricted area"
                ],
                "confidence_range": (0.75, 0.88)
            }
        }
        
        # Select random event type and details
        event_type = random.choice(list(behavior_patterns.keys()))
        pattern = behavior_patterns[event_type]
        description = random.choice(pattern["descriptions"])
        confidence = round(random.uniform(*pattern["confidence_range"]), 2)
        
        # Mock face verification results (random for testing)
        mock_face_verification = {
            "total_faces": random.randint(0, 3),
            "authorized_faces": random.randint(0, 2),
            "unauthorized_faces": random.randint(0, 2),
            "authorized_names": []
        }
        
        if mock_face_verification["authorized_faces"] > 0:
            mock_names = ["John_Doe", "Jane_Smith", "Mike_Johnson", "Sarah_Wilson"]
            mock_face_verification["authorized_names"] = random.sample(
                mock_names, min(mock_face_verification["authorized_faces"], len(mock_names))
            )
        
        return {
            "camera_id": self.camera_id,
            "timestamp": timestamp,
            "video_time_seconds": time_seconds or 0,
            "frame_number": random.randint(100, 10000),
            "event_type": event_type,
            "description": description,
            "model_confidence": confidence,
            "location": random.choice(list(self.campus_locations.keys())),  # Fixed: use campus_locations
            "face_verification": mock_face_verification,
            "audio_event": random.choice(["none", "loud_noise", "shouting", "glass_breaking"]) if random.random() < 0.2 else "none"
        }
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get basic information about the video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video information
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"error": f"Cannot open video file: {video_path}"}
            
            info = {
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_seconds": 0
            }
            
            if info["fps"] > 0:
                info["duration_seconds"] = info["total_frames"] / info["fps"]
            
            cap.release()
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {"error": str(e)}
    
    def create_sample_authorized_faces(self):
        """Create sample authorized faces for testing (this would normally be done via admin interface)."""
        import os
        
        sample_dir = "data/authorized_faces/"
        os.makedirs(sample_dir, exist_ok=True)
        
        logger.info("To add authorized faces:")
        logger.info(f"1. Place face images in {sample_dir}")
        logger.info("2. Name files as: PersonName.jpg (e.g., John_Doe.jpg)")
        logger.info("3. Restart the application to load new faces")
        
        # Create a sample instruction file
        with open(os.path.join(sample_dir, "README.txt"), "w") as f:
            f.write("VisionGuard AI - Authorized Faces Setup\n")
            f.write("=====================================\n\n")
            f.write("To add authorized personnel:\n")
            f.write("1. Add clear face photos to this directory\n")
            f.write("2. Name files as: FirstName_LastName.jpg\n")
            f.write("3. Ensure photos show clear, front-facing faces\n")
            f.write("4. Supported formats: .jpg, .jpeg, .png, .bmp\n")
            f.write("5. Restart the application to load new faces\n\n")
            f.write("Example filenames:\n")
            f.write("- John_Doe.jpg\n")
            f.write("- Jane_Smith.png\n")
            f.write("- Security_Guard_Mike.jpg\n")