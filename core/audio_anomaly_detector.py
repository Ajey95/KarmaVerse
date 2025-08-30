import numpy as np
import random
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class AudioAnomalyDetector:
    """
    Mock audio anomaly detection system for campus surveillance.
    In a real implementation, this would use PyAudio + MFCC + ML models.
    """
    
    def __init__(self):
        # Audio anomaly types for campus environment
        self.anomaly_types = {
            "glass_breaking": {
                "severity": 4,
                "description": "Glass breaking sound detected",
                "campus_context": "Possible vandalism or break-in",
                "response_required": "Immediate security response",
                "frequency_range": "high",
                "duration": "short"
            },
            "shouting": {
                "severity": 3,
                "description": "Loud shouting or yelling detected",
                "campus_context": "Possible altercation or emergency",
                "response_required": "Security assessment needed",
                "frequency_range": "mid",
                "duration": "variable"
            },
            "machinery_noise": {
                "severity": 2,
                "description": "Heavy machinery operation",
                "campus_context": "Construction or maintenance activity",
                "response_required": "Verify authorized work",
                "frequency_range": "low",
                "duration": "extended"
            },
            "vehicle_alarm": {
                "severity": 3,
                "description": "Vehicle alarm system activated",
                "campus_context": "Possible vehicle break-in or theft",
                "response_required": "Check parking areas",
                "frequency_range": "high",
                "duration": "extended"
            },
            "loud_music": {
                "severity": 2,
                "description": "Excessive noise levels detected",
                "campus_context": "Noise violation in quiet zone",
                "response_required": "Issue noise warning",
                "frequency_range": "full",
                "duration": "extended"
            },
            "metal_clanging": {
                "severity": 2,
                "description": "Metallic impact sounds",
                "campus_context": "Equipment damage or fence climbing",
                "response_required": "Investigate source",
                "frequency_range": "high",
                "duration": "intermittent"
            },
            "running_footsteps": {
                "severity": 2,
                "description": "Rapid footstep pattern",
                "campus_context": "Possible chase or emergency evacuation",
                "response_required": "Monitor situation",
                "frequency_range": "low",
                "duration": "short"
            },
            "power_tool_noise": {
                "severity": 1,
                "description": "Power tool operation detected",
                "campus_context": "Maintenance or construction work",
                "response_required": "Verify work authorization",
                "frequency_range": "high",
                "duration": "intermittent"
            },
            "door_slamming": {
                "severity": 2,
                "description": "Forceful door closure detected",
                "campus_context": "Possible forced entry or anger",
                "response_required": "Check building security",
                "frequency_range": "low",
                "duration": "short"
            },
            "emergency_siren": {
                "severity": 5,
                "description": "Emergency vehicle siren",
                "campus_context": "Emergency response in progress",
                "response_required": "Clear pathways, standby",
                "frequency_range": "full",
                "duration": "extended"
            }
        }
        
        # Zone-specific audio patterns
        self.zone_audio_patterns = {
            "main_gate": ["vehicle_alarm", "shouting", "metal_clanging"],
            "construction_site": ["machinery_noise", "power_tool_noise", "metal_clanging", "shouting"],
            "library": ["loud_music", "shouting", "door_slamming"],
            "parking_lot": ["vehicle_alarm", "glass_breaking", "running_footsteps"],
            "dormitory": ["loud_music", "shouting", "door_slamming", "glass_breaking"]
        }
        
        # Time-based audio likelihood
        self.time_based_likelihood = {
            "day": {  # 6 AM - 6 PM
                "machinery_noise": 0.3,
                "power_tool_noise": 0.4,
                "loud_music": 0.1,
                "shouting": 0.1,
                "vehicle_alarm": 0.2
            },
            "evening": {  # 6 PM - 10 PM
                "loud_music": 0.3,
                "shouting": 0.2,
                "vehicle_alarm": 0.2,
                "door_slamming": 0.2
            },
            "night": {  # 10 PM - 6 AM
                "glass_breaking": 0.4,
                "metal_clanging": 0.3,
                "running_footsteps": 0.3,
                "vehicle_alarm": 0.4,
                "shouting": 0.2
            }
        }
    
    def get_time_period(self, hour: int) -> str:
        """Determine time period based on hour."""
        if 6 <= hour < 18:
            return "day"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def simulate_audio_detection(self, zone_name: str, event_type: str = None, 
                               current_time: datetime = None) -> Optional[Dict]:
        """
        Simulate audio anomaly detection based on zone and context.
        
        Args:
            zone_name: Campus zone identifier
            event_type: Visual event type to correlate with
            current_time: Current timestamp
            
        Returns:
            Audio detection result or None
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Base probability of audio detection (15% chance)
        detection_probability = 0.15
        
        # Increase probability based on visual events
        if event_type:
            if "unauthorized" in event_type or "intrusion" in event_type:
                detection_probability = 0.4  # Higher chance of audio with suspicious activity
            elif "construction" in event_type or "safety" in event_type:
                detection_probability = 0.6  # Very likely to have audio in construction zones
            elif "vehicle" in event_type:
                detection_probability = 0.3  # Vehicles often make noise
        
        # Random decision on whether to detect audio
        if random.random() > detection_probability:
            return None
        
        # Select appropriate audio anomaly
        time_period = self.get_time_period(current_time.hour)
        zone_patterns = self.zone_audio_patterns.get(zone_name, ["shouting", "metal_clanging"])
        
        # Weight selection by time and zone
        possible_anomalies = []
        for anomaly in zone_patterns:
            if anomaly in self.time_based_likelihood[time_period]:
                likelihood = self.time_based_likelihood[time_period][anomaly]
                possible_anomalies.extend([anomaly] * int(likelihood * 10))  # Weight by likelihood
        
        if not possible_anomalies:
            possible_anomalies = zone_patterns
        
        selected_anomaly = random.choice(possible_anomalies)
        anomaly_info = self.anomaly_types[selected_anomaly]
        
        # Generate detection result
        confidence = random.uniform(0.7, 0.95)
        
        # Simulate audio features (in real implementation, these would be from MFCC analysis)
        audio_features = self._generate_mock_audio_features(selected_anomaly)
        
        return {
            "anomaly_type": selected_anomaly,
            "confidence": confidence,
            "severity": anomaly_info["severity"],
            "description": anomaly_info["description"],
            "campus_context": anomaly_info["campus_context"],
            "response_required": anomaly_info["response_required"],
            "audio_features": audio_features,
            "detection_timestamp": current_time.isoformat(),
            "zone_correlation": zone_name,
            "visual_event_correlation": event_type
        }
    
    def _generate_mock_audio_features(self, anomaly_type: str) -> Dict:
        """Generate mock audio features that would come from MFCC analysis."""
        anomaly_info = self.anomaly_types[anomaly_type]
        
        # Simulate realistic audio characteristics
        features = {
            "duration_seconds": random.uniform(0.5, 10.0),
            "peak_frequency_hz": self._get_frequency_range(anomaly_info["frequency_range"]),
            "amplitude_db": random.uniform(60, 95),  # Decibel level
            "spectral_centroid": random.uniform(1000, 4000),
            "zero_crossing_rate": random.uniform(0.1, 0.3),
            "mfcc_coefficients": [random.uniform(-10, 10) for _ in range(13)],  # 13 MFCC coefficients
            "spectral_rolloff": random.uniform(2000, 8000),
            "chroma_features": [random.uniform(0, 1) for _ in range(12)]
        }
        
        # Adjust features based on anomaly type
        if anomaly_type == "glass_breaking":
            features["peak_frequency_hz"] = random.uniform(3000, 8000)
            features["duration_seconds"] = random.uniform(0.5, 2.0)
            features["amplitude_db"] = random.uniform(75, 90)
        
        elif anomaly_type == "machinery_noise":
            features["peak_frequency_hz"] = random.uniform(100, 1000)
            features["duration_seconds"] = random.uniform(30, 300)
            features["amplitude_db"] = random.uniform(70, 85)
        
        elif anomaly_type == "shouting":
            features["peak_frequency_hz"] = random.uniform(200, 2000)
            features["duration_seconds"] = random.uniform(1, 5)
            features["amplitude_db"] = random.uniform(65, 85)
        
        return features
    
    def _get_frequency_range(self, range_type: str) -> float:
        """Get frequency value based on range type."""
        ranges = {
            "low": (50, 500),
            "mid": (500, 2000),
            "high": (2000, 8000),
            "full": (50, 8000)
        }
        min_freq, max_freq = ranges.get(range_type, (50, 8000))
        return random.uniform(min_freq, max_freq)
    
    def correlate_with_visual_event(self, visual_event: Dict, audio_detection: Dict) -> Dict:
        """
        Correlate audio detection with visual event for enhanced analysis.
        
        Args:
            visual_event: Visual detection event
            audio_detection: Audio anomaly detection
            
        Returns:
            Correlation analysis
        """
        correlation_strength = 0.0
        correlation_factors = []
        
        # Time correlation (if detected within same timeframe)
        correlation_strength += 0.3
        correlation_factors.append("Temporal correlation")
        
        # Location correlation
        if audio_detection.get("zone_correlation") == visual_event.get("zone_context", {}).get("zone_id"):
            correlation_strength += 0.4
            correlation_factors.append("Spatial correlation")
        
        # Event type correlation
        visual_event_type = visual_event.get("event_type", "")
        audio_anomaly = audio_detection.get("anomaly_type", "")
        
        # Define correlation rules
        correlations = {
            "glass_breaking": ["intrusion", "unauthorized", "break_in"],
            "shouting": ["crowd", "altercation", "emergency"],
            "machinery_noise": ["construction", "maintenance"],
            "vehicle_alarm": ["vehicle", "parking", "break_in"],
            "metal_clanging": ["fence", "climbing", "intrusion"]
        }
        
        for audio_type, visual_keywords in correlations.items():
            if audio_anomaly == audio_type:
                for keyword in visual_keywords:
                    if keyword in visual_event_type:
                        correlation_strength += 0.3
                        correlation_factors.append(f"Event-type correlation: {audio_type} + {keyword}")
                        break
        
        # Cap correlation at 1.0
        correlation_strength = min(1.0, correlation_strength)
        
        return {
            "correlation_strength": correlation_strength,
            "correlation_factors": correlation_factors,
            "is_correlated": correlation_strength > 0.5,
            "combined_severity": max(
                visual_event.get("suggested_priority", 2),
                audio_detection.get("severity", 2)
            ),
            "multimodal_confidence": (
                visual_event.get("model_confidence", 0.7) + 
                audio_detection.get("confidence", 0.7)
            ) / 2
        }
    
    def generate_audio_enhanced_alert(self, visual_event: Dict, 
                                    audio_detection: Optional[Dict] = None) -> Dict:
        """
        Generate enhanced alert combining visual and audio information.
        
        Args:
            visual_event: Visual detection event
            audio_detection: Optional audio detection
            
        Returns:
            Enhanced event with audio context
        """
        enhanced_event = visual_event.copy()
        
        if audio_detection:
            # Add audio information
            enhanced_event["audio_anomaly"] = audio_detection
            
            # Correlate audio and visual
            correlation = self.correlate_with_visual_event(visual_event, audio_detection)
            enhanced_event["audio_visual_correlation"] = correlation
            
            # Update event description with audio context
            original_desc = enhanced_event.get("description", "")
            audio_desc = f"Audio: {audio_detection['description']}"
            enhanced_event["description"] = f"{original_desc}. {audio_desc}"
            
            # Update priority if audio correlation is strong
            if correlation["is_correlated"]:
                enhanced_event["suggested_priority"] = correlation["combined_severity"]
                enhanced_event["model_confidence"] = correlation["multimodal_confidence"]
                
                # Add correlation note
                enhanced_event["zone_context"]["correlation_note"] = (
                    f"Audio-visual correlation: {correlation['correlation_strength']:.2f}"
                )
        else:
            # No audio detected
            enhanced_event["audio_anomaly"] = None
            enhanced_event["audio_visual_correlation"] = None
        
        return enhanced_event
    
    def get_zone_audio_report(self, zone_name: str, hours: int = 24) -> Dict:
        """
        Generate audio activity report for a specific zone.
        
        Args:
            zone_name: Zone to analyze
            hours: Hours to look back
            
        Returns:
            Audio activity summary
        """
        # This would query actual audio detection history in a real system
        # For demo, generate realistic summary
        
        zone_patterns = self.zone_audio_patterns.get(zone_name, [])
        
        simulated_detections = []
        for _ in range(random.randint(5, 15)):  # Simulate 5-15 detections
            anomaly_type = random.choice(zone_patterns)
            detection_time = datetime.now().hour - random.randint(0, hours)
            
            simulated_detections.append({
                "anomaly_type": anomaly_type,
                "severity": self.anomaly_types[anomaly_type]["severity"],
                "hour": max(0, detection_time)
            })
        
        # Analyze patterns
        anomaly_counts = {}
        severity_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        hourly_activity = {hour: 0 for hour in range(24)}
        
        for detection in simulated_detections:
            anomaly_type = detection["anomaly_type"]
            anomaly_counts[anomaly_type] = anomaly_counts.get(anomaly_type, 0) + 1
            severity_distribution[detection["severity"]] += 1
            hourly_activity[detection["hour"]] += 1
        
        return {
            "zone_name": zone_name,
            "analysis_period_hours": hours,
            "total_audio_detections": len(simulated_detections),
            "anomaly_type_distribution": anomaly_counts,
            "severity_distribution": severity_distribution,
            "hourly_activity_pattern": hourly_activity,
            "most_common_anomaly": max(anomaly_counts.items(), key=lambda x: x[1])[0] if anomaly_counts else None,
            "average_severity": sum(d["severity"] for d in simulated_detections) / len(simulated_detections) if simulated_detections else 0
        }