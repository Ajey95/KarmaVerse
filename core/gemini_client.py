import google.generativeai as genai
import json
import logging
import os
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class GeminiSecurityAnalyzer:
    """Handles communication with Gemini 2.5 Pro for security event analysis."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in .env file or pass it directly.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
        # System prompt for security analysis
        self.system_prompt = """You are VisionGuard AI, an advanced security analysis system for campus surveillance. 
Your job is to analyze security events and provide actionable intelligence to security teams.

For each security event you receive, provide a JSON response with exactly these fields:
- "alert": A clear, concise alert message (with appropriate emoji)
- "priority": Integer from 1-5 (1=low, 2=medium, 3=high, 4=critical, 5=emergency)
- "recommended_action": Specific action security should take
- "explanation": Brief reasoning for the alert level and recommended action
- "risk_factors": List of key risk factors identified
- "follow_up_required": Boolean indicating if follow-up investigation is needed

Consider these factors when analyzing:
- Event type and severity (gate climbing/intrusion = CRITICAL priority 4-5)
- Location sensitivity 
- Time of day/context
- Face verification results (unauthorized vs authorized persons)
- Historical patterns
- Campus security policies

SPECIAL ALERT TYPES:
- "gate_climbing_intrusion": Always priority 4-5, immediate response
- "unauthorized_gate_access": Priority 3-4 depending on context
- "fence_breach": Priority 4-5, security perimeter compromised

Be decisive but proportionate in your responses. Focus on actionable intelligence."""

    def analyze_security_event(self, detection_event: Dict) -> Dict:
        """
        Send detection event to Gemini for analysis and get structured response.
        
        Args:
            detection_event: Detection event dictionary from video processor
            
        Returns:
            Structured alert response from Gemini
        """
        try:
            # Format the detection event for Gemini
            prompt = self._format_event_prompt(detection_event)
            
            # Add delay to avoid rate limiting
            import time
            time.sleep(1)  # 1 second delay between requests
            
            # Generate response with timeout
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            alert_data = self._parse_gemini_response(response.text)
            
            # Add metadata
            alert_data['source_event'] = detection_event
            alert_data['analysis_timestamp'] = datetime.now().isoformat()
            alert_data['analyzer'] = 'Gemini-1.5-Pro'
            
            logger.info(f"Generated alert for {detection_event['event_type']} - Priority: {alert_data.get('priority', 'Unknown')}")
            return alert_data
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error analyzing security event: {error_msg}")
            
            # Check if it's a rate limit error
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                logger.warning("Rate limit detected, using fallback analysis")
                return self._create_fallback_alert(detection_event, "Rate limit exceeded - using fallback analysis")
            
            return self._create_fallback_alert(detection_event, str(e))
    
    def _format_event_prompt(self, event: Dict) -> str:
        """Format detection event into a prompt for Gemini."""
        
        # Extract key information
        event_type = event.get('event_type', 'unknown')
        description = event.get('description', 'No description')
        location = event.get('location', 'Unknown location')
        confidence = event.get('model_confidence', 0)
        timestamp = event.get('timestamp', 'Unknown time')
        
        # Face verification info
        face_info = event.get('face_verification', {})
        unauthorized_faces = face_info.get('unauthorized_faces', 0)
        authorized_faces = face_info.get('authorized_faces', 0)
        authorized_names = face_info.get('authorized_names', [])
        
        # Audio context
        audio_event = event.get('audio_event', 'none')
        
        prompt = f"""{self.system_prompt}

SECURITY EVENT ANALYSIS REQUEST:

Event Details:
- Type: {event_type}
- Description: {description}
- Location: {location}
- Timestamp: {timestamp}
- Detection Confidence: {confidence:.2f}

Face Verification Results:
- Authorized persons detected: {authorized_faces}
- Unauthorized persons detected: {unauthorized_faces}
- Known individuals: {', '.join(authorized_names) if authorized_names else 'None'}

Additional Context:
- Audio anomaly: {audio_event}
- Camera ID: {event.get('camera_id', 'Unknown')}

Please analyze this security event and respond with a JSON object containing your assessment."""

        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Parse Gemini's response and extract JSON."""
        try:
            # Try to find JSON in the response
            response_text = response_text.strip()
            
            # Look for JSON block
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                json_text = response_text[start:end].strip()
            elif '{' in response_text and '}' in response_text:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_text = response_text[start:end]
            else:
                # If no JSON found, create structured response from text
                return self._create_structured_response_from_text(response_text)
            
            # Parse JSON
            parsed = json.loads(json_text)
            
            # Validate required fields
            required_fields = ['alert', 'priority', 'recommended_action', 'explanation']
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = f"Missing {field}"
            
            # Ensure priority is integer between 1-5
            try:
                parsed['priority'] = max(1, min(5, int(parsed['priority'])))
            except (ValueError, TypeError):
                parsed['priority'] = 3
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            return self._create_structured_response_from_text(response_text)
    
    def _create_structured_response_from_text(self, text: str) -> Dict:
        """Create structured response when JSON parsing fails."""
        # Extract priority from text (basic heuristics)
        text_lower = text.lower()
        priority = 3  # default
        
        if any(word in text_lower for word in ['emergency', 'critical', 'immediate']):
            priority = 5
        elif any(word in text_lower for word in ['high', 'urgent', 'serious']):
            priority = 4
        elif any(word in text_lower for word in ['medium', 'moderate']):
            priority = 3
        elif any(word in text_lower for word in ['low', 'minor']):
            priority = 2
        
        return {
            'alert': f"⚠️ Security Alert - Analysis Available",
            'priority': priority,
            'recommended_action': "Review full analysis and take appropriate action",
            'explanation': text[:200] + "..." if len(text) > 200 else text,
            'risk_factors': ["Analysis parsing issue"],
            'follow_up_required': True,
            'raw_response': text
        }
    
    def _create_fallback_alert(self, event: Dict, error_msg: str) -> Dict:
        """Create fallback alert when Gemini API fails."""
        event_type = event.get('event_type', 'unknown')
        unauthorized_faces = event.get('face_verification', {}).get('unauthorized_faces', 0)
        location = event.get('location', 'Unknown')
        
        # Enhanced gate climbing detection in fallback
        if event_type == "gate_climbing_intrusion" or "climbing" in event_type:
            priority = 5  # Emergency
            alert = f"🚨 CRITICAL: Gate Climbing Detected at {location}"
            action = "IMMEDIATE RESPONSE - Security breach in progress"
            explanation = "Unauthorized individual detected climbing over security gate - Critical security perimeter breach"
            risk_factors = ["Security perimeter breach", "Unauthorized intrusion", "Active climbing behavior"]
            
        elif event_type in ['unauthorized_gate_access', 'potential_intrusion']:
            priority = 4  # Critical
            alert = f"🔥 CRITICAL: Unauthorized Gate Access - {location}"
            action = "Immediate security team dispatch required"
            explanation = "Unknown individual detected at secured gate entrance"
            risk_factors = ["Unauthorized access attempt", "Gate security breach"]
            
        elif unauthorized_faces > 0:
            priority = 4
            alert = f"🚨 Unauthorized Person Detected - {event_type.replace('_', ' ').title()}"
            action = "Immediate security response required"
            risk_factors = ["Unauthorized person", event_type]
            explanation = f"Unknown individual detected with {unauthorized_faces} unauthorized face(s)"
            
        elif event_type in ['intrusion', 'unauthorized_entry']:
            priority = 4
            alert = f"🚨 Security Breach - {event_type.replace('_', ' ').title()}"
            action = "Dispatch security team immediately"
            risk_factors = ["Security breach", event_type]
            explanation = "Potential security intrusion detected"
            
        elif event_type in ['loitering', 'suspicious_behavior']:
            priority = 3
            alert = f"⚠️ Suspicious Activity - {event_type.replace('_', ' ').title()}"
            action = "Monitor situation and prepare to intervene"
            risk_factors = ["Suspicious behavior", event_type]
            explanation = "Unusual behavior pattern detected requiring monitoring"
            
        else:
            priority = 2
            alert = f"ℹ️ Security Event - {event_type.replace('_', ' ').title()}"
            action = "Review and assess situation"
            risk_factors = [event_type]
            explanation = f"Security event detected: {event_type}"
        
        return {
            'alert': alert,
            'priority': priority,
            'recommended_action': action,
            'explanation': explanation,
            'risk_factors': risk_factors,
            'follow_up_required': priority >= 3,
            'source_event': event,
            'analysis_timestamp': datetime.now().isoformat(),
            'analyzer': 'Enhanced-Fallback-System',
            'error': error_msg
        }
    
    def batch_analyze_events(self, detection_events: List[Dict]) -> List[Dict]:
        """
        Analyze multiple detection events in batch.
        
        Args:
            detection_events: List of detection event dictionaries
            
        Returns:
            List of alert responses
        """
        alerts = []
        
        for event in detection_events:
            try:
                alert = self.analyze_security_event(event)
                alerts.append(alert)
            except Exception as e:
                logger.error(f"Error in batch analysis: {e}")
                fallback_alert = self._create_fallback_alert(event, str(e))
                alerts.append(fallback_alert)
        
        return alerts
    
    def generate_incident_report(self, alerts: List[Dict]) -> str:
        """
        Generate a comprehensive incident report from multiple alerts.
        
        Args:
            alerts: List of alert dictionaries
            
        Returns:
            Formatted incident report
        """
        try:
            # Summarize alerts
            total_alerts = len(alerts)
            priority_counts = {}
            event_types = {}
            
            for alert in alerts:
                priority = alert.get('priority', 3)
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
                
                event_type = alert.get('source_event', {}).get('event_type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            # Create summary prompt
            prompt = f"""Generate a professional security incident report based on the following alert summary:

Total Alerts: {total_alerts}
Priority Distribution: {priority_counts}
Event Types: {event_types}

Recent Alerts:
{json.dumps(alerts[-5:], indent=2)}

Please provide a concise but comprehensive incident report suitable for security management."""

            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating incident report: {e}")
            return f"Incident Report Generation Failed: {e}"
    
    def health_check(self) -> Dict:
        """Check if Gemini API is accessible and working."""
        try:
            test_prompt = "Respond with: {'status': 'healthy'}"
            response = self.model.generate_content(test_prompt)
            
            return {
                'status': 'healthy',
                'api_accessible': True,
                'model': 'gemini-2.5-pro',
                'test_response': response.text[:100]
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'api_accessible': False,
                'error': str(e)
            }