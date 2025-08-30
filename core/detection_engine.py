import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, time
import random

logger = logging.getLogger(__name__)

class ZoneContextManager:
    """Manages campus zones and contextual alerting rules."""
    
    def __init__(self):
        self.zones = {
            "main_gate": {
                "name": "Main Campus Gate",
                "allowed_objects": ["person", "car", "bicycle", "motorcycle"],
                "restricted_objects": ["truck", "bus"],  # Need special permission
                "high_risk_hours": [(22, 6)],  # 10PM to 6AM
                "max_loitering_time": 180,  # 3 minutes
                "visitor_escort_required": False,
                "safety_equipment_required": [],
                "alert_multipliers": {
                    "unauthorized_vehicle": 1.5,
                    "after_hours": 2.0,
                    "loitering": 1.2
                }
            },
            "construction_site": {
                "name": "Campus Construction Zone",
                "allowed_objects": ["person", "truck", "car"],
                "restricted_objects": ["bicycle", "motorcycle"],  # Safety hazard
                "high_risk_hours": [(18, 7), (12, 13)],  # After hours + lunch break
                "max_loitering_time": 600,  # 10 minutes for work
                "visitor_escort_required": True,
                "safety_equipment_required": ["helmet", "vest"],  # Would need custom detection
                "alert_multipliers": {
                    "no_safety_gear": 3.0,
                    "unauthorized_access": 2.5,
                    "restricted_vehicle": 2.0,
                    "after_hours": 1.8
                }
            },
            "library": {
                "name": "Academic Library",
                "allowed_objects": ["person", "backpack", "laptop"],
                "restricted_objects": ["bicycle", "car", "motorcycle"],
                "high_risk_hours": [(23, 6)],  # Late night
                "max_loitering_time": 7200,  # 2 hours (study time)
                "visitor_escort_required": False,
                "safety_equipment_required": [],
                "alert_multipliers": {
                    "noise_disturbance": 1.5,
                    "unauthorized_entry": 2.0,
                    "suspicious_behavior": 1.3
                }
            },
            "parking_lot": {
                "name": "Student Parking Area",
                "allowed_objects": ["person", "car", "bicycle", "motorcycle"],
                "restricted_objects": ["truck"],
                "high_risk_hours": [(22, 6)],
                "max_loitering_time": 300,  # 5 minutes
                "visitor_escort_required": False,
                "safety_equipment_required": [],
                "alert_multipliers": {
                    "vehicle_break_in": 2.0,
                    "unauthorized_parking": 1.2,
                    "after_hours": 1.5
                }
            },
            "dormitory": {
                "name": "Student Residence Hall",
                "allowed_objects": ["person"],
                "restricted_objects": ["car", "truck", "motorcycle"],  # No vehicles near dorms
                "high_risk_hours": [(2, 6)],  # Very late night
                "max_loitering_time": 600,  # 10 minutes (visiting)
                "visitor_escort_required": True,
                "safety_equipment_required": [],
                "alert_multipliers": {
                    "unauthorized_entry": 3.0,
                    "visitor_unescorted": 2.0,
                    "late_night_activity": 1.5
                }
            }
        }
    
    def get_zone_config(self, zone_name: str) -> Dict:
        """Get configuration for specific zone."""
        return self.zones.get(zone_name, self.zones["main_gate"])  # Default fallback
    
    def is_high_risk_time(self, zone_name: str, current_time: time = None) -> bool:
        """Check if current time is high-risk for zone."""
        if current_time is None:
            current_time = datetime.now().time()
        
        zone_config = self.get_zone_config(zone_name)
        current_hour = current_time.hour
        
        for start_hour, end_hour in zone_config["high_risk_hours"]:
            if start_hour > end_hour:  # Overnight period (e.g., 22 to 6)
                if current_hour >= start_hour or current_hour <= end_hour:
                    return True
            else:  # Same day period
                if start_hour <= current_hour <= end_hour:
                    return True
        
        return False
    
    def assess_object_risk(self, zone_name: str, detected_objects: List[str], 
                          face_info: Dict, current_time: time = None) -> Dict:
        """Assess risk level based on zone context, objects, and time."""
        zone_config = self.get_zone_config(zone_name)
        risk_factors = []
        base_priority = 2  # Default medium priority
        alert_multiplier = 1.0
        
        # Check restricted objects
        restricted_found = [obj for obj in detected_objects if obj in zone_config["restricted_objects"]]
        if restricted_found:
            risk_factors.append(f"Restricted objects detected: {', '.join(restricted_found)}")
            base_priority += 1
            alert_multiplier *= zone_config["alert_multipliers"].get("restricted_vehicle", 1.5)
        
        # Check time-based risk
        if self.is_high_risk_time(zone_name, current_time):
            risk_factors.append("Activity during high-risk hours")
            base_priority += 1
            alert_multiplier *= zone_config["alert_multipliers"].get("after_hours", 1.5)
        
        # Check unauthorized personnel
        if face_info.get("unauthorized_faces", 0) > 0:
            if zone_config["visitor_escort_required"]:
                risk_factors.append("Unauthorized person in restricted zone")
                base_priority += 2
                alert_multiplier *= zone_config["alert_multipliers"].get("unauthorized_access", 2.0)
            else:
                risk_factors.append("Unknown person detected")
                base_priority += 1
                alert_multiplier *= zone_config["alert_multipliers"].get("unauthorized_entry", 1.3)
        
        # Zone-specific rules
        if zone_name == "construction_site":
            # Construction safety rules
            if "person" in detected_objects:
                risk_factors.append("Person in construction zone - safety gear check required")
                base_priority += 1
                alert_multiplier *= zone_config["alert_multipliers"].get("no_safety_gear", 2.0)
        
        elif zone_name == "library":
            # Library-specific rules
            if "bicycle" in detected_objects or "motorcycle" in detected_objects:
                risk_factors.append("Vehicle not allowed in academic building")
                base_priority += 1
        
        # Cap priority at 5
        final_priority = min(5, base_priority)
        
        return {
            "zone_name": zone_config["name"],
            "risk_factors": risk_factors,
            "base_priority": base_priority,
            "final_priority": final_priority,
            "alert_multiplier": alert_multiplier,
            "is_high_risk_time": self.is_high_risk_time(zone_name, current_time),
            "zone_rules_triggered": len(risk_factors)
        }


class ObjectDetectionEngine:
    """Enhanced object detection using YOLOv8 with campus zone awareness."""
    
    def __init__(self, model_size: str = "n"):  # n=nano for speed, s=small for accuracy
        self.model_size = model_size
        self.model = None
        self.zone_manager = ZoneContextManager()
        self.confidence_threshold = 0.4
        self.iou_threshold = 0.45
        
        # COCO class names for campus-relevant objects
        self.campus_objects = {
            0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
            1: "bicycle", 24: "backpack", 67: "cell phone", 73: "laptop",
            28: "suitcase", 31: "handbag", 39: "bottle"
        }
        
        self.load_model()
    
    def load_model(self):
        """Load YOLOv8 model."""
        try:
            model_name = f"yolov8{self.model_size}.pt"
            logger.info(f"Loading YOLOv8 model: {model_name}")
            self.model = YOLO(model_name)
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            logger.info("Falling back to mock detection")
            self.model = None
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame using YOLOv8.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            return self._mock_object_detection(frame)
        
        try:
            # Run YOLOv8 inference
            results = self.model(frame, conf=self.confidence_threshold, iou=self.iou_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection info
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Only include campus-relevant objects
                        if class_id in self.campus_objects:
                            object_name = self.campus_objects[class_id]
                            
                            detection = {
                                'object_class': object_name,
                                'confidence': confidence,
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                'area': (x2 - x1) * (y2 - y1)
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return self._mock_object_detection(frame)
    
    def _mock_object_detection(self, frame: np.ndarray) -> List[Dict]:
        """Generate mock object detections for demo purposes."""
        height, width = frame.shape[:2]
        
        # Simulate realistic campus detections
        possible_objects = ["person", "car", "bicycle", "backpack", "truck", "motorcycle"]
        num_detections = random.randint(1, 4)
        
        detections = []
        for _ in range(num_detections):
            obj_class = random.choice(possible_objects)
            
            # Generate realistic bounding box
            x1 = random.randint(50, width - 200)
            y1 = random.randint(50, height - 150)
            x2 = x1 + random.randint(80, 200)
            y2 = y1 + random.randint(60, 150)
            
            detection = {
                'object_class': obj_class,
                'confidence': random.uniform(0.6, 0.95),
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'area': (x2 - x1) * (y2 - y1)
            }
            detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection boxes and labels on frame."""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            obj_class = detection['object_class']
            
            # Color coding for different object types
            colors = {
                'person': (0, 255, 0),      # Green
                'car': (255, 0, 0),         # Blue  
                'truck': (0, 0, 255),       # Red
                'bicycle': (255, 255, 0),   # Cyan
                'motorcycle': (255, 0, 255), # Magenta
                'backpack': (0, 255, 255),  # Yellow
            }
            
            color = colors.get(obj_class, (128, 128, 128))  # Default gray
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{obj_class}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def generate_zone_aware_event(self, detections: List[Dict], face_info: Dict, 
                                 zone_name: str, location: str, timestamp: str) -> Dict:
        """
        Generate security event with zone-aware contextual analysis.
        
        Args:
            detections: Object detection results
            face_info: Face verification results  
            zone_name: Campus zone identifier
            location: Specific location name
            timestamp: Event timestamp
            
        Returns:
            Enhanced security event with zone context
        """
        detected_objects = [det['object_class'] for det in detections]
        
        # Get zone-based risk assessment
        zone_assessment = self.zone_manager.assess_object_risk(
            zone_name, detected_objects, face_info
        )
        
        # Determine event type based on zone and objects
        event_type = self._determine_event_type(detections, face_info, zone_name, zone_assessment)
        
        # Generate contextual description
        description = self._generate_contextual_description(
            detections, face_info, zone_name, zone_assessment
        )
        
        # Calculate final confidence based on zone rules
        base_confidence = sum(det['confidence'] for det in detections) / len(detections) if detections else 0.7
        zone_confidence_modifier = min(1.2, zone_assessment['alert_multiplier'])
        final_confidence = min(0.98, base_confidence * zone_confidence_modifier)
        
        return {
            "camera_id": f"CAM_{zone_name.upper()}",
            "timestamp": timestamp,
            "event_type": event_type,
            "description": description,
            "model_confidence": round(final_confidence, 2),
            "location": location,
            "zone_context": {
                "zone_name": zone_assessment["zone_name"],
                "zone_id": zone_name,
                "risk_factors": zone_assessment["risk_factors"],
                "priority_boost": zone_assessment["final_priority"] - 2,  # Base was 2
                "is_high_risk_time": zone_assessment["is_high_risk_time"],
                "rules_triggered": zone_assessment["zone_rules_triggered"]
            },
            "detected_objects": [
                {
                    "class": det['object_class'],
                    "confidence": det['confidence'],
                    "bbox": det['bbox']
                } for det in detections
            ],
            "face_verification": face_info,
            "audio_event": self._simulate_audio_context(event_type, zone_name),
            "suggested_priority": zone_assessment["final_priority"]
        }
    
    def _determine_event_type(self, detections: List[Dict], face_info: Dict, 
                             zone_name: str, zone_assessment: Dict) -> str:
        """Determine event type based on zone context and detections."""
        detected_objects = [det['object_class'] for det in detections]
        
        # Enhanced gate/fence climbing detection
        if zone_name == "main_gate":
            # Check for climbing behavior indicators
            person_detections = [det for det in detections if det['object_class'] == 'person']
            
            for person_det in person_detections:
                # If person is detected at gate with unauthorized face, likely climbing
                if face_info.get("unauthorized_faces", 0) > 0:
                    bbox = person_det['bbox']
                    height_ratio = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])  # height/width ratio
                    
                    # Heuristic: unusual height ratio might indicate climbing
                    if height_ratio > 1.5 or person_det.get('confidence', 0) > 0.8:
                        return "gate_climbing_intrusion"
                        
            if any(obj in detected_objects for obj in ["truck", "bus"]):
                return "unauthorized_vehicle_entry"
            elif face_info.get("unauthorized_faces", 0) > 0:
                return "unauthorized_gate_access"
        
        elif zone_name == "construction_site":
            if "person" in detected_objects:
                if face_info.get("unauthorized_faces", 0) > 0:
                    return "unauthorized_construction_access"
                else:
                    return "construction_safety_check"
            elif any(obj in detected_objects for obj in ["bicycle", "motorcycle"]):
                return "safety_violation"
        
        elif zone_name == "main_gate":
            if any(obj in detected_objects for obj in ["truck", "bus"]):
                return "unauthorized_vehicle_entry"
            elif face_info.get("unauthorized_faces", 0) > 0:
                return "visitor_entry"
        
        elif zone_name == "dormitory":
            if face_info.get("unauthorized_faces", 0) > 0:
                return "unauthorized_dorm_access"
        
        elif zone_name == "library":
            if any(obj in detected_objects for obj in ["bicycle", "motorcycle"]):
                return "prohibited_vehicle"
        
        # General event types with enhanced intrusion detection
        if face_info.get("unauthorized_faces", 0) > 0:
            if zone_assessment["is_high_risk_time"]:
                return "after_hours_intrusion"
            else:
                # Check for potential climbing/intrusion behavior
                if "person" in detected_objects and len(detected_objects) == 1:
                    return "potential_intrusion"
                else:
                    return "unauthorized_person"
        
        if len(detected_objects) > 3:
            return "crowd_formation"
        
        return "routine_monitoring"
    
    def _generate_contextual_description(self, detections: List[Dict], face_info: Dict,
                                       zone_name: str, zone_assessment: Dict) -> str:
        """Generate contextual description based on zone and detections."""
        detected_objects = [det['object_class'] for det in detections]
        zone_config = self.zone_manager.get_zone_config(zone_name)
        
        descriptions = []
        
        # Object-based description
        if detected_objects:
            obj_counts = {}
            for obj in detected_objects:
                obj_counts[obj] = obj_counts.get(obj, 0) + 1
            
            obj_desc = ", ".join([f"{count} {obj}{'s' if count > 1 else ''}" 
                                 for obj, count in obj_counts.items()])
            descriptions.append(f"Detected: {obj_desc}")
        
        # Face verification context
        if face_info.get("unauthorized_faces", 0) > 0:
            descriptions.append(f"{face_info['unauthorized_faces']} unauthorized person(s)")
        if face_info.get("authorized_faces", 0) > 0:
            descriptions.append(f"{face_info['authorized_faces']} authorized person(s)")
        
        # Zone-specific context
        zone_desc = f"in {zone_config['name']}"
        if zone_assessment["is_high_risk_time"]:
            zone_desc += " during high-risk hours"
        
        descriptions.append(zone_desc)
        
        # Risk factors
        if zone_assessment["risk_factors"]:
            descriptions.append(f"Risk factors: {'; '.join(zone_assessment['risk_factors'][:2])}")
        
        return ". ".join(descriptions)
    
    def _simulate_audio_context(self, event_type: str, zone_name: str) -> str:
        """Simulate audio context based on event and zone."""
        audio_contexts = {
            "construction_site": ["machinery_noise", "drilling", "hammering", "vehicle_backup"],
            "main_gate": ["vehicle_engine", "gate_opening", "footsteps"],
            "library": ["quiet_ambiance", "page_turning", "whispers"],
            "parking_lot": ["car_doors", "engine_start", "footsteps"],
            "dormitory": ["door_closing", "conversations", "footsteps"]
        }
        
        # Event-specific audio
        if "unauthorized" in event_type or "intrusion" in event_type:
            return random.choice(["suspicious_movement", "quiet_footsteps", "door_rattling"])
        elif "vehicle" in event_type:
            return random.choice(["engine_noise", "vehicle_approach", "car_door"])
        elif "safety" in event_type:
            return random.choice(["warning_beep", "machinery_noise", "shouting"])
        
        # Zone default audio
        zone_audio = audio_contexts.get(zone_name, ["ambient_noise"])
        return random.choice(zone_audio)