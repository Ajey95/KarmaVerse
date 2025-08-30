import streamlit as st
import os
import sys
import logging
import tempfile
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from core.video_processor import VideoProcessor
from core.gemini_client import GeminiSecurityAnalyzer
from core.alert_manager import AlertManager
from core.face_verification import FaceVerificationSystem
from core.audio_anomaly_detector import AudioAnomalyDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="VisionGuard AI - Smart Campus Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .alert-card {
        border-left: 4px solid;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
        border-radius: 5px;
    }
    
    .priority-1 { border-left-color: #28a745; }
    .priority-2 { border-left-color: #17a2b8; }
    .priority-3 { border-left-color: #ffc107; }
    .priority-4 { border-left-color: #fd7e14; }
    .priority-5 { border-left-color: #dc3545; }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-active { background-color: #dc3545; }
    .status-investigating { background-color: #ffc107; }
    .status-resolved { background-color: #28a745; }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'processed_videos' not in st.session_state:
        st.session_state.processed_videos = []
    if 'gemini_analyzer' not in st.session_state:
        st.session_state.gemini_analyzer = None
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = VideoProcessor()
    if 'alert_manager' not in st.session_state:
        st.session_state.alert_manager = AlertManager()
    if 'audio_detector' not in st.session_state:
        st.session_state.audio_detector = AudioAnomalyDetector()
    if 'blur_all_faces' not in st.session_state:
        st.session_state.blur_all_faces = False

def setup_gemini():
    """Setup Gemini analyzer with API key."""
    if st.session_state.gemini_analyzer is None:
        # Check for API key in environment or get from user
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key or api_key == 'your_gemini_api_key_here':
            # Only show input if not already in session state
            if 'gemini_api_input' not in st.session_state:
                st.session_state.gemini_api_input = ""
            
            if not st.session_state.gemini_api_input:
                st.sidebar.error("⚠️ Gemini API Key Required")
                api_key = st.sidebar.text_input(
                    "Enter Gemini API Key:", 
                    type="password",
                    key="gemini_api_key_input",
                    help="Get your API key from Google AI Studio"
                )
                st.session_state.gemini_api_input = api_key
            else:
                api_key = st.session_state.gemini_api_input
            
            if api_key:
                os.environ['GEMINI_API_KEY'] = api_key
            else:
                return False
        
        try:
            st.session_state.gemini_analyzer = GeminiSecurityAnalyzer(api_key)
            return True
        except Exception as e:
            st.sidebar.error(f"Failed to initialize Gemini: {e}")
            return False
    return True

def render_header():
    """Render the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ VisionGuard AI</h1>
        <p>Smart Campus Security with AI-Powered Threat Detection</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar_stats():
    """Render sidebar statistics."""
    st.sidebar.markdown("### 📊 System Status")
    
    # System health checks
    gemini_status = "🟢 Connected" if st.session_state.gemini_analyzer else "🔴 Not Connected"
    st.sidebar.markdown(f"**Gemini AI:** {gemini_status}")
    
    face_verifier = FaceVerificationSystem()
    authorized_count = len(face_verifier.authorized_encodings)
    unknown_clusters = len(face_verifier.face_clusters)
    
    st.sidebar.markdown(f"**Authorized Faces:** {authorized_count}")
    st.sidebar.markdown(f"**Unknown Clusters:** {unknown_clusters}")
    st.sidebar.markdown(f"**Learning Buffer:** {len(face_verifier.new_identity_buffer)}")
    
    # Alert statistics
    alert_stats = st.session_state.alert_manager.get_alert_statistics(days=7)
    st.sidebar.markdown("### 🚨 Alert Summary (7 days)")
    st.sidebar.metric("Total Alerts", alert_stats.get('total_alerts', 0))
    
    priority_dist = alert_stats.get('priority_distribution', {})
    for priority in [5, 4, 3, 2, 1]:
        if priority in priority_dist:
            label = st.session_state.alert_manager.priority_labels[priority]
            emoji = st.session_state.alert_manager.priority_emojis[priority]
            st.sidebar.metric(f"{emoji} {label}", priority_dist[priority])

def process_video_file(uploaded_file):
    """Process uploaded video file with enhanced detection and generate alerts."""
    if uploaded_file is None:
        return
    
    with st.spinner("🎬 Processing video with enhanced AI detection..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Get video info
            video_info = st.session_state.video_processor.get_video_info(tmp_path)
            st.success(f"📹 Video loaded: {video_info.get('total_frames', 0)} frames, {video_info.get('duration_seconds', 0):.1f}s")
            
            # Process video with enhanced detection
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Running YOLO object detection + face verification + zone analysis...")
            processed_frames, detection_events = st.session_state.video_processor.process_video_with_enhanced_detection(
                tmp_path, nth_frame=60  # Process every 60th frame for speed
            )
            progress_bar.progress(40)
            
            if detection_events:
                status_text.text("🎵 Adding audio anomaly detection...")
                
                # Enhance events with audio detection
                enhanced_events = []
                for event in detection_events:
                    # Simulate audio detection
                    audio_detection = st.session_state.audio_detector.simulate_audio_detection(
                        event.get("zone_context", {}).get("zone_id", "main_gate"),
                        event.get("event_type"),
                        datetime.now()
                    )
                    
                    # Enhance event with audio
                    enhanced_event = st.session_state.audio_detector.generate_audio_enhanced_alert(
                        event, audio_detection
                    )
                    enhanced_events.append(enhanced_event)
                
                progress_bar.progress(60)
                
                status_text.text("🤖 Analyzing with Gemini AI...")
                
                # Analyze events with Gemini
                if setup_gemini():
                    alerts = []
                    for i, event in enumerate(enhanced_events):
                        try:
                            alert = st.session_state.gemini_analyzer.analyze_security_event(event)
                            if alert:  # Check if alert is not None
                                alert_id = st.session_state.alert_manager.save_alert(alert)
                                alerts.append(alert)
                            else:
                                # Create fallback alert if Gemini returns None
                                fallback_alert = {
                                    'alert': f"🚨 {event.get('event_type', 'security_event').replace('_', ' ').title()}",
                                    'priority': 3,
                                    'recommended_action': 'Investigate security event',
                                    'explanation': event.get('description', 'Security event detected'),
                                    'source_event': event,
                                    'analysis_timestamp': datetime.now().isoformat(),
                                    'analyzer': 'Fallback-System'
                                }
                                alert_id = st.session_state.alert_manager.save_alert(fallback_alert)
                                alerts.append(fallback_alert)
                        except Exception as e:
                            logger.error(f"Error analyzing individual event: {e}")
                            # Create fallback alert for failed analysis
                            fallback_alert = {
                                'alert': f"🚨 {event.get('event_type', 'security_event').replace('_', ' ').title()}",
                                'priority': 3,
                                'recommended_action': 'Manual review required',
                                'explanation': f"Analysis failed: {str(e)}",
                                'source_event': event,
                                'analysis_timestamp': datetime.now().isoformat(),
                                'analyzer': 'Error-Fallback-System'
                            }
                            alert_id = st.session_state.alert_manager.save_alert(fallback_alert)
                            alerts.append(fallback_alert)
                        
                        progress = 60 + (40 * (i + 1) / len(enhanced_events))
                        progress_bar.progress(int(progress))
                    
                    st.session_state.alerts.extend(alerts)
                    status_text.text("✅ Enhanced AI analysis complete!")
                    
                    # Display results summary - with error handling
                    try:
                        audio_events = sum(1 for event in enhanced_events 
                                         if event and event.get("audio_anomaly"))
                        correlated_events = sum(1 for event in enhanced_events 
                                              if event and event.get("audio_visual_correlation", {}) and
                                              event.get("audio_visual_correlation", {}).get("is_correlated", False))
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🎯 Security Events", len(alerts))
                        with col2:
                            objects_detected = sum(len(e.get("detected_objects", [])) for e in enhanced_events if e)
                            st.metric("👁️ Objects Detected", objects_detected)
                        with col3:
                            st.metric("🎵 Audio Anomalies", audio_events)
                        with col4:
                            st.metric("🔗 Correlated Events", correlated_events)
                            
                        # Success message with gate jumping detection
                        if any("intrusion" in alert.get("alert", "").lower() for alert in alerts):
                            st.success("🚨 **CRITICAL SECURITY BREACH DETECTED!** Gate climbing/intrusion identified.")
                        elif any("potential" in alert.get("alert", "").lower() for alert in alerts):
                            st.warning("⚠️ **POTENTIAL SECURITY THREAT** detected in video analysis.")
                        else:
                            st.success(f"🎯 Analysis complete: {len(alerts)} security events generated.")
                            
                    except Exception as e:
                        st.error(f"Display error: {e}")
                        st.success(f"✅ Analysis complete! {len(alerts)} security events detected.")
                    
                else:
                    st.error("❌ Cannot analyze events without Gemini connection")
            else:
                st.info("ℹ️ No security events detected in this video")
            
            progress_bar.progress(100)
            
        except Exception as e:
            st.error(f"Error processing video: {e}")
            logger.error(f"Video processing error: {e}")
        
        finally:
            # Cleanup temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

def render_video_upload():
    """Render enhanced video upload interface."""
    st.header("📹 Enhanced Video Analysis")
    
    # Privacy settings
    with st.expander("🔒 Privacy Settings"):
        col1, col2 = st.columns(2)
        with col1:
            blur_all = st.checkbox(
                "Blur ALL faces (GDPR Compliance)", 
                value=st.session_state.blur_all_faces,
                help="Enable to blur all detected faces, including authorized personnel"
            )
            st.session_state.blur_all_faces = blur_all
        
        with col2:
            detection_sensitivity = st.slider(
                "Detection Sensitivity", 
                min_value=0.3, max_value=0.9, value=0.5, step=0.1,
                help="Higher values = more sensitive detection"
            )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Security Video")
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload security camera footage for comprehensive AI analysis"
        )
        
        if uploaded_file:
            # Show video info
            st.video(uploaded_file)
            
            st.markdown("### 🎯 Analysis Features")
            features = [
                "✅ **YOLOv8 Object Detection** - Vehicles, people, objects",
                "✅ **Face Recognition & Verification** - Authorized vs unauthorized",
                "✅ **Zone-Based Context Analysis** - Campus-specific rules",
                "✅ **Audio Anomaly Detection** - Sound pattern analysis",
                "✅ **Semi-Supervised Learning** - Automatic identity clustering",
                "✅ **Gemini AI Security Analysis** - Intelligent threat assessment"
            ]
            for feature in features:
                st.markdown(feature)
            
            if st.button("🔍 Analyze Video with Enhanced AI", type="primary", use_container_width=True):
                process_video_file(uploaded_file)
    
    with col2:
        st.markdown("### 🎲 Demo Options")
        
        # Enhanced sample events
        if st.button("🎯 Generate Enhanced Events", use_container_width=True):
            with st.spinner("Generating enhanced sample events..."):
                import random
                from datetime import datetime
                
                # Generate 3-5 sample events with enhanced features
                num_events = random.randint(3, 5)
                zones = ["main_gate", "construction_site", "library", "parking_lot", "dormitory"]
                
                for _ in range(num_events):
                    # Select random zone
                    zone = random.choice(zones)
                    location = zone.replace("_", " ").title()
                    
                    # Generate enhanced event
                    event = st.session_state.video_processor.generate_mock_detection(
                        timestamp=datetime.now().isoformat()
                    )
                    
                    # Add zone context
                    event["zone_context"] = {
                        "zone_name": location,
                        "zone_id": zone,
                        "is_high_risk_time": random.choice([True, False])
                    }
                    
                    # Add audio detection
                    audio_detection = st.session_state.audio_detector.simulate_audio_detection(
                        zone, event["event_type"], datetime.now()
                    )
                    
                    enhanced_event = st.session_state.audio_detector.generate_audio_enhanced_alert(
                        event, audio_detection
                    )
                    
                    # Try Gemini, but use fallback if it fails
                    try:
                        if setup_gemini():
                            alert = st.session_state.gemini_analyzer.analyze_security_event(enhanced_event)
                        else:
                            # Use fallback alert creation
                            alert = st.session_state.gemini_analyzer._create_fallback_alert(enhanced_event, "Demo mode - Gemini unavailable") if st.session_state.gemini_analyzer else {
                                'alert': f"🚨 {enhanced_event['event_type'].replace('_', ' ').title()} Detected",
                                'priority': random.randint(2, 4),
                                'recommended_action': f"Investigate {enhanced_event['event_type']} at {enhanced_event['location']}",
                                'explanation': enhanced_event['description'],
                                'source_event': enhanced_event,
                                'analysis_timestamp': datetime.now().isoformat(),
                                'analyzer': 'Demo-Fallback-System'
                            }
                        
                        alert_id = st.session_state.alert_manager.save_alert(alert)
                        st.session_state.alerts.append(alert)
                        
                    except Exception as e:
                        st.warning(f"Gemini unavailable (demo mode): {str(e)}")
                        # Create fallback alert
                        fallback_alert = {
                            'alert': f"🚨 {enhanced_event['event_type'].replace('_', ' ').title()} Detected",
                            'priority': random.randint(2, 4),
                            'recommended_action': f"Investigate {enhanced_event['event_type']} at {enhanced_event['location']}",
                            'explanation': enhanced_event['description'],
                            'source_event': enhanced_event,
                            'analysis_timestamp': datetime.now().isoformat(),
                            'analyzer': 'Demo-Fallback-System'
                        }
                        alert_id = st.session_state.alert_manager.save_alert(fallback_alert)
                        st.session_state.alerts.append(fallback_alert)
                
                st.success(f"🎉 Generated {num_events} enhanced security events with multi-modal analysis!")
                st.info("💡 Note: Using demo mode due to API limitations. Full Gemini analysis available with proper API setup.")
        
        # Zone information
        st.markdown("### 🏫 Campus Zones")
        zones_info = {
            "🚪 Main Gate": "Entry control, vehicle monitoring",
            "🏗️ Construction Site": "Safety compliance, restricted access",
            "📚 Library": "Quiet zones, study areas",
            "🚗 Parking Lot": "Vehicle security, parking violations",
            "🏠 Dormitory": "Resident access, visitor management"
        }
        
        for zone, description in zones_info.items():
            st.markdown(f"**{zone}**: {description}")
        
        # System status
        st.markdown("### 📊 System Status")
        st.info(f"🤖 **Object Detection**: YOLOv8 Ready")
        st.info(f"👁️ **Face Recognition**: {len(st.session_state.video_processor.face_verifier.authorized_encodings)} Authorized")
        st.info(f"🎵 **Audio Detection**: Mock System Active")
        st.info(f"🧠 **Gemini AI**: {'Connected' if st.session_state.gemini_analyzer else 'Disconnected'}")

def render_alert_card(alert_data):
    """Render individual alert card."""
    formatted = st.session_state.alert_manager.format_alert_for_display(alert_data)
    
    # Create alert card
    with st.container():
        # Priority indicator and title
        priority_color = formatted['priority_color']
        st.markdown(f"""
        <div class="alert-card priority-{formatted['priority']}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h4 style="margin: 0; color: {priority_color};">
                    {formatted['priority_emoji']} {formatted['title']}
                </h4>
                <span style="background: {priority_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">
                    {formatted['priority_label']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Use separate elements for content to avoid HTML issues
        st.markdown(f"**📍 Location:** {formatted['location']}")
        st.markdown(f"**⏰ Time:** {formatted['display_time']}")
        st.markdown(f"**🎯 Event:** {formatted['event_type'].replace('_', ' ').title()}")
        
        st.markdown("**📋 Recommended Action:**")
        st.write(formatted['recommended_action'])
        
        st.markdown("**💡 Explanation:**")
        st.write(formatted['description'])
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("✅ Resolve", key=f"resolve_{formatted['alert_id']}"):
                st.session_state.alert_manager.update_alert_status(
                    formatted['alert_id'], 'resolved', 'Resolved via dashboard'
                )
                st.success("Alert marked as resolved")
                st.rerun()
        
        with col2:
            if st.button("🔍 Investigate", key=f"investigate_{formatted['alert_id']}"):
                st.session_state.alert_manager.update_alert_status(
                    formatted['alert_id'], 'investigating', 'Under investigation'
                )
                st.info("Alert marked as under investigation")
                st.rerun()
        
        with col3:
            if st.button("❌ False Positive", key=f"false_{formatted['alert_id']}"):
                st.session_state.alert_manager.update_alert_status(
                    formatted['alert_id'], 'false_positive', 'Marked as false positive'
                )
                st.warning("Alert marked as false positive")
                st.rerun()
        
        with col4:
            with st.expander("📊 Details"):
                st.json(formatted['face_info'])
                st.text(f"Confidence: {formatted['confidence']:.2f}")
                st.text(f"Analyzer: {formatted['analyzer']}")

def render_alerts_dashboard():
    """Render the main alerts dashboard."""
    st.header("🚨 Security Alerts Dashboard")
    
    # Load recent alerts
    recent_alerts = st.session_state.alert_manager.load_alerts(limit=50)
    
    if not recent_alerts:
        st.info("No alerts found. Upload a video or generate sample events to get started.")
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_priority = st.selectbox(
            "Minimum Priority",
            options=[1, 2, 3, 4, 5],
            index=0,
            format_func=lambda x: f"{st.session_state.alert_manager.priority_emojis[x]} {st.session_state.alert_manager.priority_labels[x]}"
        )
    
    with col2:
        status_filter = st.selectbox(
            "Status Filter",
            options=['all', 'active', 'investigating', 'resolved', 'false_positive'],
            index=0
        )
    
    with col3:
        limit = st.number_input("Number of Alerts", min_value=5, max_value=100, value=20)
    
    # Apply filters
    filtered_alerts = []
    for alert in recent_alerts[:limit]:
        if alert.get('priority', 0) >= min_priority:
            if status_filter == 'all' or alert.get('status', 'active') == status_filter:
                filtered_alerts.append(alert)
    
    st.markdown(f"**Showing {len(filtered_alerts)} alerts**")
    
    # Display alerts
    for alert in filtered_alerts:
        render_alert_card(alert)

def render_analytics():
    """Render analytics and statistics."""
    st.header("📊 Security Analytics")
    
    # Get statistics
    stats_7d = st.session_state.alert_manager.get_alert_statistics(7)
    stats_30d = st.session_state.alert_manager.get_alert_statistics(30)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Alerts (7 days)", stats_7d.get('total_alerts', 0))
    with col2:
        st.metric("Alerts (30 days)", stats_30d.get('total_alerts', 0))
    with col3:
        avg_per_day = stats_7d.get('average_alerts_per_day', 0)
        st.metric("Avg/Day", f"{avg_per_day:.1f}")
    with col4:
        high_priority = sum(stats_7d.get('priority_distribution', {}).get(p, 0) for p in [4, 5])
        st.metric("High Priority", high_priority)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Priority distribution
        priority_dist = stats_7d.get('priority_distribution', {})
        if priority_dist:
            priority_labels = [st.session_state.alert_manager.priority_labels[p] for p in priority_dist.keys()]
            priority_colors = [st.session_state.alert_manager.priority_colors[p] for p in priority_dist.keys()]
            
            fig = px.pie(
                values=list(priority_dist.values()),
                names=priority_labels,
                title="Alert Priority Distribution (7 days)",
                color_discrete_sequence=priority_colors
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Event type distribution
        event_dist = stats_7d.get('event_type_distribution', {})
        if event_dist:
            fig = px.bar(
                x=list(event_dist.keys()),
                y=list(event_dist.values()),
                title="Event Types (7 days)",
                labels={'x': 'Event Type', 'y': 'Count'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Location analysis
    location_dist = stats_7d.get('location_distribution', {})
    if location_dist:
        st.subheader("📍 Location Analysis")
        fig = px.bar(
            x=list(location_dist.keys()),
            y=list(location_dist.values()),
            title="Alerts by Location (7 days)",
            labels={'x': 'Location', 'y': 'Alert Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

def render_chat_interface():
    """Render chat interface for follow-up queries."""
    st.header("💬 Ask VisionGuard AI")
    
    if not setup_gemini():
        st.error("Gemini connection required for chat functionality")
        return
    
    # Chat history
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about security events, generate reports, or get insights..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get recent alerts for context
                    recent_alerts = st.session_state.alert_manager.load_alerts(limit=10)
                    
                    # Create context-aware prompt
                    context_prompt = f"""User query: {prompt}
                    
Recent security alerts context:
{len(recent_alerts)} recent alerts available.

Please provide a helpful response about campus security. If asked for reports or summaries, use the alert context appropriately."""
                    
                    response = st.session_state.gemini_analyzer.model.generate_content(context_prompt)
                    response_text = response.text
                    
                    st.markdown(response_text)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {e}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

def render_face_management():
    """Render enhanced face management interface with semi-supervised learning insights."""
    st.header("👥 Identity Management & Semi-Supervised Learning")
    
    face_verifier = FaceVerificationSystem()
    
    # Create tabs for different aspects
    tab1, tab2, tab3 = st.tabs(["📁 Authorized Personnel", "🤖 Semi-Supervised Learning", "📊 Clustering Report"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📁 Current Database")
            st.info(f"**{len(face_verifier.authorized_encodings)}** authorized faces loaded")
            
            if face_verifier.authorized_names:
                st.write("**Authorized Personnel:**")
                for name in face_verifier.authorized_names:
                    st.write(f"• {name.replace('_', ' ')}")
            
            # Refresh button
            if st.button("🔄 Refresh Database"):
                st.session_state.video_processor = VideoProcessor()  # Reinitialize
                st.success("Database refreshed!")
                st.rerun()
        
        with col2:
            st.subheader("📋 Setup Instructions")
            st.markdown("""
            **To add authorized personnel:**
            
            1. 📁 Navigate to `data/authorized_faces/` folder
            2. 📸 Add clear, front-facing photos of authorized personnel
            3. 🏷️ Name files as: `FirstName_LastName.jpg`
            4. 🔄 Click "Refresh Database" to load new faces
            
            **File Requirements:**
            - **Formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`
            - **Quality:** Clear, well-lit front-facing photos
            - **Size:** Any reasonable size (will be auto-processed)
            
            **Example filenames:**
            - `John_Doe.jpg`
            - `Security_Manager.png`
            - `Jane_Smith_Admin.jpg`
            """)
            
            # Directory creation helper
            if st.button("📁 Create Faces Directory"):
                os.makedirs("data/authorized_faces/", exist_ok=True)
                st.success("Directory created: `data/authorized_faces/`")
    
    with tab2:
        st.subheader("🤖 Semi-Supervised Learning System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Learning Progress")
            st.metric("Unknown Identity Clusters", len(face_verifier.face_clusters))
            st.metric("Faces in Learning Buffer", len(face_verifier.new_identity_buffer))
            st.metric("Total Unknown Faces Processed", len(face_verifier.unknown_encodings))
            
            st.markdown("### ⚙️ System Configuration")
            st.write(f"**Clustering Threshold:** {face_verifier.clustering_threshold}")
            st.write(f"**Min Cluster Size:** {face_verifier.min_cluster_size}")
            st.write(f"**Similarity Threshold:** {face_verifier.similarity_threshold}")
        
        with col2:
            st.markdown("### 🎯 How It Works")
            st.markdown("""
            **Semi-Supervised Learning Process:**
            
            1. **🔍 Initial Detection**: System detects unknown faces in video
            2. **📊 Feature Extraction**: Generates face embeddings using deep learning
            3. **🎯 Clustering**: Groups similar faces using DBSCAN algorithm
            4. **🏷️ Identity Inference**: Identifies recurring unknown individuals
            5. **⚠️ Alert Generation**: Flags potential new identities for review
            
            **Benefits:**
            - **Reduced Labeling**: Minimal manual annotation required
            - **Adaptive Learning**: Automatically discovers new identities
            - **Pattern Recognition**: Identifies frequent unknown visitors
            - **Security Enhancement**: Flags suspicious recurring individuals
            """)
            
            if st.button("🔄 Force Cluster Update"):
                with st.spinner("Updating face clusters..."):
                    face_verifier._update_face_clusters()
                    st.success("Clusters updated successfully!")
                    st.rerun()
    
    with tab3:
        st.subheader("📊 Clustering Analysis Report")
        
        # Get clustering report
        report = face_verifier.get_clustering_report()
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Clusters", report['summary']['total_unknown_clusters'])
        with col2:
            st.metric("Faces Processed", report['summary']['total_unknown_faces_processed'])
        with col3:
            st.metric("Learning Buffer", report['summary']['faces_in_learning_buffer'])
        with col4:
            st.metric("Authorized IDs", report['summary']['total_authorized_identities'])
        
        # Cluster details
        if report['cluster_details']:
            st.subheader("🔍 Unknown Identity Clusters")
            
            for cluster_id, details in report['cluster_details'].items():
                with st.expander(f"Cluster: {cluster_id} ({details['face_count']} faces)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Total Appearances:** {details['face_count']}")
                        st.write(f"**First Detected:** {details['first_detected'][:19] if details['first_detected'] != 'Unknown' else 'Unknown'}")
                        st.write(f"**Last Detected:** {details['last_detected'][:19] if details['last_detected'] != 'Unknown' else 'Unknown'}")
                        st.write(f"**Frequency Score:** {details['frequency_score']:.2f}")
                    
                    with col2:
                        st.write("**Locations Seen:**")
                        for location, count in details['locations_seen'].items():
                            st.write(f"• {location}: {count} times")
        
        # Recommendations
        if report['recommendations']:
            st.subheader("💡 System Recommendations")
            
            for rec in report['recommendations']:
                if rec['priority'] == 'high':
                    st.error(f"🔥 **{rec['type'].replace('_', ' ').title()}**: {rec['message']}")
                elif rec['priority'] == 'medium':
                    st.warning(f"⚠️ **{rec['type'].replace('_', ' ').title()}**: {rec['message']}")
                else:
                    st.info(f"ℹ️ **{rec['type'].replace('_', ' ').title()}**: {rec['message']}")
        else:
            st.info("No specific recommendations at this time. System is learning from ongoing detections.")

def render_audio_analysis():
    """Render audio analysis dashboard."""
    st.header("🎵 Audio Anomaly Analysis")
    
    # Audio system overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Audio Detection System", "Active")
        st.metric("Anomaly Types", len(st.session_state.audio_detector.anomaly_types))
    
    with col2:
        st.metric("Campus Zones Monitored", len(st.session_state.audio_detector.zone_audio_patterns))
        st.metric("Time-Based Patterns", "3 periods")
    
    with col3:
        if st.button("🔄 Generate Audio Report"):
            st.rerun()
    
    # Zone-based audio analysis
    st.subheader("📊 Zone Audio Activity")
    
    selected_zone = st.selectbox(
        "Select Campus Zone",
        options=list(st.session_state.audio_detector.zone_audio_patterns.keys()),
        format_func=lambda x: x.replace("_", " ").title()
    )
    
    if selected_zone:
        # Get audio report for selected zone
        audio_report = st.session_state.audio_detector.get_zone_audio_report(selected_zone, hours=24)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Zone Activity Summary")
            st.metric("Total Audio Detections (24h)", audio_report["total_audio_detections"])
            st.metric("Most Common Anomaly", audio_report["most_common_anomaly"] or "None")
            st.metric("Average Severity", f"{audio_report['average_severity']:.1f}/5")
            
            # Severity distribution chart
            severity_data = audio_report["severity_distribution"]
            if any(severity_data.values()):
                fig_severity = px.bar(
                    x=list(severity_data.keys()),
                    y=list(severity_data.values()),
                    title="Audio Anomaly Severity Distribution",
                    labels={'x': 'Severity Level', 'y': 'Count'},
                    color=list(severity_data.values()),
                    color_continuous_scale="Reds"
                )
                st.plotly_chart(fig_severity, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Anomaly Types Detected")
            anomaly_data = audio_report["anomaly_type_distribution"]
            
            if anomaly_data:
                # Anomaly type distribution
                fig_anomaly = px.pie(
                    values=list(anomaly_data.values()),
                    names=[name.replace("_", " ").title() for name in anomaly_data.keys()],
                    title="Audio Anomaly Types (24h)"
                )
                st.plotly_chart(fig_anomaly, use_container_width=True)
                
                # Detailed breakdown
                st.markdown("**Detailed Breakdown:**")
                for anomaly_type, count in anomaly_data.items():
                    anomaly_info = st.session_state.audio_detector.anomaly_types[anomaly_type]
                    severity = anomaly_info["severity"]
                    description = anomaly_info["description"]
                    
                    severity_color = ["🟢", "🟡", "🟠", "🔴", "🚫"][severity-1]
                    st.markdown(f"**{severity_color} {anomaly_type.replace('_', ' ').title()}** ({count}x)")
                    st.caption(description)
            else:
                st.info("No audio anomalies detected in selected zone")
    
    # Audio anomaly types reference
    st.subheader("📚 Audio Anomaly Reference")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Anomaly Types", "🏫 Zone Patterns", "⏰ Time Analysis"])
    
    with tab1:
        st.markdown("### Audio Anomaly Classification")
        
        for anomaly_type, info in st.session_state.audio_detector.anomaly_types.items():
            with st.expander(f"{info['severity']}⭐ {anomaly_type.replace('_', ' ').title()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Campus Context:** {info['campus_context']}")
                
                with col2:
                    st.write(f"**Response Required:** {info['response_required']}")
                    st.write(f"**Frequency Range:** {info['frequency_range']}")
                    st.write(f"**Duration:** {info['duration']}")
    
    with tab2:
        st.markdown("### Zone-Specific Audio Patterns")
        
        for zone, patterns in st.session_state.audio_detector.zone_audio_patterns.items():
            st.markdown(f"**{zone.replace('_', ' ').title()}:**")
            pattern_display = ", ".join([p.replace("_", " ").title() for p in patterns])
            st.caption(f"Common anomalies: {pattern_display}")
    
    with tab3:
        st.markdown("### Time-Based Audio Analysis")
        
        time_periods = st.session_state.audio_detector.time_based_likelihood
        
        for period, anomalies in time_periods.items():
            st.markdown(f"### {period.title()} Period")
            
            if anomalies:
                # Create likelihood chart
                fig_time = px.bar(
                    x=list(anomalies.keys()),
                    y=list(anomalies.values()),
                    title=f"Audio Anomaly Likelihood - {period.title()}",
                    labels={'x': 'Anomaly Type', 'y': 'Likelihood'},
                    color=list(anomalies.values()),
                    color_continuous_scale="Blues"
                )
                fig_time.update_xaxes(tickangle=45)
                st.plotly_chart(fig_time, use_container_width=True)
    
    # Audio-Visual correlation insights
    st.subheader("🔗 Audio-Visual Correlation Insights")
    
    recent_alerts = st.session_state.alert_manager.load_alerts(limit=20)
    
    # Safe filtering to avoid None errors
    audio_correlated = []
    for alert in recent_alerts:
        try:
            if (alert and 
                isinstance(alert, dict) and 
                alert.get('source_event') and 
                isinstance(alert.get('source_event'), dict) and
                alert.get('source_event', {}).get('audio_visual_correlation') and
                alert.get('source_event', {}).get('audio_visual_correlation', {}).get('is_correlated', False)):
                audio_correlated.append(alert)
        except (AttributeError, TypeError):
            continue  # Skip invalid alerts
    
    if audio_correlated:
        st.success(f"Found {len(audio_correlated)} events with strong audio-visual correlation")
        
        for alert in audio_correlated[:3]:  # Show top 3
            correlation = alert['source_event']['audio_visual_correlation']
            audio_info = alert['source_event'].get('audio_anomaly', {})
            
            with st.expander(f"Correlated Event: {alert.get('alert', 'Security Alert')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Visual Event:**")
                    st.write(f"Type: {alert['source_event'].get('event_type', 'Unknown')}")
                    st.write(f"Location: {alert['source_event'].get('location', 'Unknown')}")
                
                with col2:
                    st.write("**Audio Detection:**")
                    st.write(f"Anomaly: {audio_info.get('anomaly_type', 'None')}")
                    st.write(f"Confidence: {audio_info.get('confidence', 0):.2f}")
                
                st.write(f"**Correlation Strength:** {correlation['correlation_strength']:.2f}")
                st.write(f"**Correlation Factors:** {', '.join(correlation['correlation_factors'])}")
    else:
        st.info("No recent events with strong audio-visual correlation found")

def main():
    """Main application function."""
    initialize_session_state()
    render_header()
    
    # Sidebar
    with st.sidebar:
        render_sidebar_stats()
        
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "📍 Navigate",
            ["🎬 Video Analysis", "🚨 Alerts Dashboard", "📊 Analytics", "💬 AI Chat", "👥 Face Management", "🎵 Audio Analysis"]
        )
    
    # Main content based on selected page
    if page == "🎬 Video Analysis":
        render_video_upload()
    elif page == "🚨 Alerts Dashboard":
        render_alerts_dashboard()
    elif page == "📊 Analytics":
        render_analytics()
    elif page == "💬 AI Chat":
        render_chat_interface()
    elif page == "👥 Face Management":
        render_face_management()
    elif page == "🎵 Audio Analysis":
        render_audio_analysis()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 20px;">
            🛡️ VisionGuard AI - Powered by Gemini 2.5 Pro & YOLOv8<br>
            Built for Smart Campus Security by KarmaVerse
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()