import json
import os
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class AlertManager:
    """Manages security alerts, storage, filtering, and reporting."""
    
    def __init__(self, alerts_dir: str = "data/alerts/"):
        self.alerts_dir = alerts_dir
        os.makedirs(alerts_dir, exist_ok=True)
        
        self.priority_colors = {
            1: "#28a745",  # Green - Low
            2: "#17a2b8",  # Blue - Medium
            3: "#ffc107",  # Yellow - High
            4: "#fd7e14",  # Orange - Critical
            5: "#dc3545"   # Red - Emergency
        }
        
        self.priority_labels = {
            1: "Low",
            2: "Medium", 
            3: "High",
            4: "Critical",
            5: "Emergency"
        }
        
        self.priority_emojis = {
            1: "ℹ️",
            2: "⚠️",
            3: "🚨",
            4: "🔥",
            5: "🚫"
        }
    
    def save_alert(self, alert: Dict) -> str:
        """
        Save alert to persistent storage.
        
        Args:
            alert: Alert dictionary from Gemini analysis
            
        Returns:
            Alert ID (filename)
        """
        try:
            # Generate unique alert ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            event_type = alert.get('source_event', {}).get('event_type', 'unknown')
            alert_id = f"{timestamp}_{event_type}"
            
            # Add alert metadata
            alert['alert_id'] = alert_id
            alert['created_at'] = datetime.now(timezone.utc).isoformat()
            alert['status'] = 'active'
            
            # Save to JSON file
            filename = f"{alert_id}.json"
            filepath = os.path.join(self.alerts_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(alert, f, indent=2, default=str)
            
            logger.info(f"Saved alert: {alert_id}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
            return ""
    
    def load_alerts(self, limit: Optional[int] = None, 
                   min_priority: Optional[int] = None) -> List[Dict]:
        """
        Load alerts from storage with optional filtering.
        
        Args:
            limit: Maximum number of alerts to return
            min_priority: Minimum priority level to include
            
        Returns:
            List of alert dictionaries, sorted by creation time (newest first)
        """
        try:
            alerts = []
            
            # Get all alert files
            for filename in os.listdir(self.alerts_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.alerts_dir, filename)
                    
                    try:
                        with open(filepath, 'r') as f:
                            alert = json.load(f)
                            
                        # Apply priority filter
                        if min_priority and alert.get('priority', 0) < min_priority:
                            continue
                            
                        alerts.append(alert)
                        
                    except Exception as e:
                        logger.warning(f"Error loading alert file {filename}: {e}")
            
            # Sort by creation time (newest first)
            alerts.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            # Apply limit
            if limit:
                alerts = alerts[:limit]
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error loading alerts: {e}")
            return []
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Dict]:
        """Get specific alert by ID."""
        try:
            filepath = os.path.join(self.alerts_dir, f"{alert_id}.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading alert {alert_id}: {e}")
        return None
    
    def update_alert_status(self, alert_id: str, status: str, notes: str = "") -> bool:
        """
        Update alert status (e.g., 'resolved', 'investigating', 'false_positive').
        
        Args:
            alert_id: Alert identifier
            status: New status
            notes: Optional notes about status change
            
        Returns:
            True if successful, False otherwise
        """
        try:
            alert = self.get_alert_by_id(alert_id)
            if not alert:
                return False
            
            alert['status'] = status
            alert['status_updated_at'] = datetime.now(timezone.utc).isoformat()
            alert['status_notes'] = notes
            
            # Save updated alert
            filepath = os.path.join(self.alerts_dir, f"{alert_id}.json")
            with open(filepath, 'w') as f:
                json.dump(alert, f, indent=2, default=str)
            
            logger.info(f"Updated alert {alert_id} status to: {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating alert status: {e}")
            return False
    
    def get_alert_statistics(self, days: int = 7) -> Dict:
        """
        Get alert statistics for the specified number of days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Statistics dictionary
        """
        try:
            alerts = self.load_alerts()
            
            # Filter by date range
            cutoff_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
            
            recent_alerts = []
            for alert in alerts:
                try:
                    alert_date = datetime.fromisoformat(alert.get('created_at', ''))
                    if alert_date >= cutoff_date:
                        recent_alerts.append(alert)
                except:
                    continue
            
            # Calculate statistics
            total_alerts = len(recent_alerts)
            priority_counts = {}
            event_type_counts = {}
            status_counts = {}
            location_counts = {}
            
            for alert in recent_alerts:
                # Priority distribution
                priority = alert.get('priority', 3)
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
                
                # Event type distribution
                event_type = alert.get('source_event', {}).get('event_type', 'unknown')
                event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
                
                # Status distribution
                status = alert.get('status', 'active')
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Location distribution
                location = alert.get('source_event', {}).get('location', 'unknown')
                location_counts[location] = location_counts.get(location, 0) + 1
            
            return {
                'period_days': days,
                'total_alerts': total_alerts,
                'priority_distribution': priority_counts,
                'event_type_distribution': event_type_counts,
                'status_distribution': status_counts,
                'location_distribution': location_counts,
                'average_alerts_per_day': round(total_alerts / max(days, 1), 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating alert statistics: {e}")
            return {'error': str(e)}
    
    def format_alert_for_display(self, alert: Dict) -> Dict:
        """
        Format alert for display in Streamlit with enhanced styling.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            Formatted alert with display properties
        """
        priority = alert.get('priority', 3)
        
        formatted = {
            'alert_id': alert.get('alert_id', 'unknown'),
            'timestamp': alert.get('analysis_timestamp', ''),
            'title': alert.get('alert', 'Security Alert'),
            'priority': priority,
            'priority_label': self.priority_labels.get(priority, 'Unknown'),
            'priority_color': self.priority_colors.get(priority, '#6c757d'),
            'priority_emoji': self.priority_emojis.get(priority, '❓'),
            'description': alert.get('explanation', ''),
            'recommended_action': alert.get('recommended_action', ''),
            'risk_factors': alert.get('risk_factors', []),
            'location': alert.get('source_event', {}).get('location', 'Unknown'),
            'event_type': alert.get('source_event', {}).get('event_type', 'unknown'),
            'confidence': alert.get('source_event', {}).get('model_confidence', 0),
            'face_info': alert.get('source_event', {}).get('face_verification', {}),
            'status': alert.get('status', 'active'),
            'follow_up_required': alert.get('follow_up_required', False),
            'analyzer': alert.get('analyzer', 'Unknown'),
            'raw_alert': alert  # Keep original for reference
        }
        
        # Format timestamp for display
        try:
            dt = datetime.fromisoformat(formatted['timestamp'].replace('Z', '+00:00'))
            formatted['display_time'] = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted['display_time'] = formatted['timestamp']
        
        return formatted
    
    def export_alerts_to_csv(self, alerts: List[Dict], filename: str = None) -> str:
        """
        Export alerts to CSV format.
        
        Args:
            alerts: List of alert dictionaries
            filename: Output filename (optional)
            
        Returns:
            Path to exported CSV file
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"security_alerts_{timestamp}.csv"
            
            # Flatten alert data for CSV
            csv_data = []
            for alert in alerts:
                row = {
                    'alert_id': alert.get('alert_id', ''),
                    'timestamp': alert.get('analysis_timestamp', ''),
                    'priority': alert.get('priority', ''),
                    'alert_message': alert.get('alert', ''),
                    'event_type': alert.get('source_event', {}).get('event_type', ''),
                    'location': alert.get('source_event', {}).get('location', ''),
                    'description': alert.get('source_event', {}).get('description', ''),
                    'confidence': alert.get('source_event', {}).get('model_confidence', ''),
                    'recommended_action': alert.get('recommended_action', ''),
                    'explanation': alert.get('explanation', ''),
                    'status': alert.get('status', ''),
                    'unauthorized_faces': alert.get('source_event', {}).get('face_verification', {}).get('unauthorized_faces', 0),
                    'authorized_faces': alert.get('source_event', {}).get('face_verification', {}).get('authorized_faces', 0),
                    'analyzer': alert.get('analyzer', '')
                }
                csv_data.append(row)
            
            # Create DataFrame and export
            df = pd.DataFrame(csv_data)
            output_path = os.path.join(self.alerts_dir, filename)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Exported {len(alerts)} alerts to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting alerts to CSV: {e}")
            return ""
    
    def cleanup_old_alerts(self, days_to_keep: int = 30) -> int:
        """
        Clean up alerts older than specified days.
        
        Args:
            days_to_keep: Number of days of alerts to retain
            
        Returns:
            Number of alerts deleted
        """
        try:
            cutoff_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_to_keep)
            
            deleted_count = 0
            
            for filename in os.listdir(self.alerts_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.alerts_dir, filename)
                    
                    try:
                        with open(filepath, 'r') as f:
                            alert = json.load(f)
                        
                        alert_date = datetime.fromisoformat(alert.get('created_at', ''))
                        
                        if alert_date < cutoff_date:
                            os.remove(filepath)
                            deleted_count += 1
                            logger.info(f"Deleted old alert: {filename}")
                            
                    except Exception as e:
                        logger.warning(f"Error processing alert file {filename}: {e}")
            
            logger.info(f"Cleanup completed: {deleted_count} old alerts deleted")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0