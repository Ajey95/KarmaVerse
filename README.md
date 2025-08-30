# 🛡️ VisionGuard AI - Smart Campus Security

An AI-powered video surveillance system that detects suspicious behavior, verifies authorized personnel, and generates intelligent security alerts using Gemini 2.5 Pro.

Note:- "The live demo includes our API key for seamless evaluation. The system gracefully handles rate limits and works fully even without API access via intelligent fallbacks."

## ✨ Features

- **🎬 Enhanced Video Analysis**: Upload and analyze security camera footage with YOLOv8 object detection
- **👤 Advanced Face Recognition**: Semi-supervised learning with automatic identity clustering
- **🔒 Privacy-First Design**: GDPR-compliant face blurring with toggle for all faces
- **🏫 Zone-Based Contextual Alerting**: Campus-specific rules for different areas (construction, library, dorms)
- **🎵 Audio Anomaly Detection**: Multi-modal surveillance with sound pattern analysis
- **🚨 Smart Alerts**: AI-powered threat detection with priority assessment using Gemini 2.5 Pro
- **📊 Real-time Analytics**: Comprehensive security metrics and correlation insights
- **💬 AI Assistant**: Natural language queries about security events and incidents
- **🤖 Semi-Supervised Learning**: Automatic unknown identity clustering with minimal labeling

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd visionguard-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.template .env
# Edit .env and add your Gemini API key
```

### 2. Get Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

### 3. Set Up Authorized Faces

```bash
# Create faces directory
mkdir -p data/authorized_faces

# Add authorized personnel photos
# Name files as: FirstName_LastName.jpg
# Example: John_Doe.jpg, Jane_Smith.jpg
```

### 4. Run the Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.


## Video link
**[https://drive.google.com/file/d/1FsKN8_sDNhvWexEEy8FXu037VjX3kPDS/view?usp=sharing]

**For Judges & Evaluators:**

### 🔑 API Key Information
- **Gemini API**: The live demo uses our API key for seamless evaluation
- **Rate Limits**: Free tier has 15 requests/minute, but system gracefully handles limits
- **Fallback System**: Full functionality available even without API access
- **Demo Mode**: Click "🎲 Generate Enhanced Events" for instant demonstration

### 🎬 Quick Evaluation Steps (2 minutes)
1. **Visit the live demo** (no setup required)
2. **Generate Sample Events**: Click "🎲 Generate Enhanced Events" 
3. **View Alerts**: Navigate to "🚨 Alerts Dashboard"
4. **Check Features**: Explore all 6 main sections
5. **Test Video Upload**: Upload any .mp4 file for real-time analysis

### 🎯 Key Demo Features to Evaluate
- ✅ **Gate Climbing Detection**: Upload video showing fence/gate climbing → System flags as Priority 4-5 Critical
- ✅ **Face Recognition**: Authorized vs unauthorized person handling with privacy blurring
- ✅ **Zone Intelligence**: Campus-specific rules (Main Gate, Construction, Library, Dorms)
- ✅ **Multi-Modal Analysis**: Audio-visual correlation for enhanced security
- ✅ **Semi-Supervised Learning**: Automatic identity clustering with minimal labeling
- ✅ **AI Chat**: Natural language security queries and incident reports

## 📁 Project Structure

```
visionguard-ai/
├── app.py                          # Main Streamlit application
├── config.py                      # Configuration settings
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables
├── README.md                     # This file
│
├── core/                         # Core functionality
│   ├── video_processor.py        # Video processing & frame extraction
│   ├── face_verification.py      # Face recognition & verification
│   ├── gemini_client.py          # Gemini AI integration
│   └── alert_manager.py          # Alert storage & management
│
├── data/                         # Data storage
│   ├── authorized_faces/         # Authorized personnel photos
│   ├── sample_videos/            # Sample video files
│   ├── alerts/                   # Generated security alerts
│   └── logs/                     # Application logs
│
├── models/                       # ML model weights
│   └── yolo_weights/             # YOLO model files
│
└── tests/                        # Test files
    ├── test_video_processor.py
    ├── test_face_verification.py
    └── test_gemini_client.py
```

## 🎯 Usage Guide

### Video Analysis

1. **Upload Video**: Go to "🎬 Enhanced Video Analysis" tab
2. **Privacy Settings**: Configure GDPR-compliant face blurring
3. **Detection Sensitivity**: Adjust AI detection thresholds
4. **Analyze**: Click "🔍 Analyze Video with Enhanced AI" to process
5. **View Results**: Review multi-modal detection results

### Face Management & Semi-Supervised Learning

1. **Add Personnel**: Place photos in `data/authorized_faces/` as `FirstName_LastName.jpg`
2. **Monitor Learning**: Check "🤖 Semi-Supervised Learning" tab for automatic clustering
3. **Review Clusters**: View unknown identity patterns in "📊 Clustering Report"
4. **System Recommendations**: Follow AI suggestions for security improvements

### Zone-Based Security

- **Campus Zones**: 5 pre-configured areas with intelligent rules
- **Contextual Alerts**: Location and time-aware threat assessment
- **Safety Compliance**: Construction zone equipment detection
- **Access Control**: Dormitory and restricted area monitoring

### Audio Analysis

1. **Zone Selection**: Choose campus area for audio monitoring
2. **Anomaly Detection**: Review 10+ audio threat types
3. **Pattern Analysis**: Understand time-based audio patterns
4. **Correlation**: View audio-visual event matching

### Alert Management

- **Priority System**: 5-level threat classification with colors
- **Status Tracking**: Resolve, investigate, or mark false positives
- **Export Capabilities**: Download alerts as CSV for compliance
- **Real-time Dashboard**: Live security monitoring interface

### AI Chat Assistant

Ask natural language questions like:
- "Summarize today's security events"
- "Why was this alert generated?"  
- "Show me high-priority incidents from this week"
- "Generate an incident report for management"
- "What are the patterns in construction zone alerts?"

## 🔧 Configuration

### Video Processing Settings

```python
# config.py
VIDEO_PROCESSING = {
    'frame_extraction_interval': 30,  # Every 30th frame
    'max_file_size_mb': 100,         # 100MB limit
    'resize_frame_width': 640,       # Resize for performance
}
```

### Face Recognition Settings

```python
FACE_RECOGNITION = {
    'similarity_threshold': 0.6,      # Matching sensitivity
    'blur_factor': 15,               # Blur intensity
    'face_detection_model': 'hog',   # 'hog' or 'cnn'
}
```

### Alert Priority Levels

- **🚫 Emergency (5)**: Immediate response required
- **🔥 Critical (4)**: Urgent security threat
- **🚨 High (3)**: Significant security concern
- **⚠️ Medium (2)**: Monitor situation
- **ℹ️ Low (1)**: Informational alert

## 🎮 Demo Mode

For testing without video files:

1. Go to "🎬 Video Analysis"
2. Click "🎲 Generate Sample Events"
3. View generated alerts in dashboard

## 🔍 Event Types Detected

### Visual Detection (YOLOv8)
- **Object Detection**: Person, vehicle, bicycle, backpack, suspicious objects
- **Zone-Aware Analysis**: Context-sensitive alerts based on campus areas
- **Face Verification**: Authorized vs unauthorized personnel identification

### Behavioral Analysis
- **Loitering**: Person stationary for extended period in sensitive areas
- **Intrusion**: Unauthorized entry or climbing over barriers
- **Abandoned Object**: Unattended items in restricted zones
- **Unauthorized Vehicle**: Restricted vehicles in prohibited areas
- **Construction Safety**: Safety equipment compliance in work zones
- **Crowd Formation**: Large group gatherings in restricted areas

### Audio Anomalies
- **Glass Breaking**: Potential vandalism or break-in attempts
- **Shouting/Yelling**: Possible altercations or emergencies
- **Vehicle Alarms**: Security incidents in parking areas
- **Machinery Noise**: Unauthorized construction activity
- **Emergency Sirens**: Active emergency response situations

## 📊 Analytics Features

- **Real-time Metrics**: Alert counts and trends
- **Priority Distribution**: Visual breakdown of alert severities
- **Location Analysis**: Security hotspot identification
- **Event Type Trends**: Pattern recognition over time
- **Export Capabilities**: CSV reports for external analysis

## 🛠️ Technical Details

### AI Models Used

- **Gemini 2.5 Pro**: Security event analysis and intelligent alert generation
- **YOLOv8**: Real-time object detection (person, vehicle, equipment)
- **Face Recognition (dlib)**: Face encoding and verification against authorized database
- **DBSCAN Clustering**: Semi-supervised learning for automatic identity discovery
- **FAISS**: Fast similarity search for face matching and clustering

### Campus Zone Intelligence

Each zone has specific rules and context:

**🚪 Main Gate**
- Vehicle type restrictions (no trucks/buses without permission)
- Visitor management and escort requirements
- After-hours access monitoring
- Loitering detection with 3-minute threshold

**🏗️ Construction Site**  
- Safety equipment compliance (helmet/vest detection ready)
- Restricted vehicle access (no motorcycles/bicycles)
- Worker authorization verification
- After-hours and lunch break monitoring

**📚 Library/Academic Buildings**
- Noise level monitoring and disturbance alerts
- Vehicle restrictions (no bikes/cars inside)
- Extended loitering allowance (2 hours for studying)
- Late-night access control

**🚗 Parking Lot**
- Vehicle break-in detection patterns
- Unauthorized parking identification
- 5-minute loitering threshold
- After-hours activity monitoring

**🏠 Student Dormitory**
- Strict visitor escort requirements
- No vehicle access near residence halls
- Very late night activity alerts (2-6 AM)
- Resident-only access enforcement

### Audio Anomaly Detection

**10+ Audio Threat Types:**
- Glass Breaking (vandalism/break-in)
- Shouting/Yelling (altercations/emergencies)  
- Vehicle Alarms (security incidents)
- Machinery Noise (unauthorized construction)
- Emergency Sirens (active response situations)
- Metal Clanging (fence climbing/equipment damage)
- Running Footsteps (chase/evacuation scenarios)
- Loud Music (noise violations)
- Door Slamming (forced entry/anger)
- Power Tools (unauthorized maintenance)

### Performance Optimization

- **Frame Sampling**: Process every 30th frame for speed
- **Face Caching**: FAISS indexing for fast face lookups
- **Batch Processing**: Multiple events analyzed together
- **Lazy Loading**: Components loaded on demand

### Security & Privacy

- **Face Blurring**: Automatic anonymization of unauthorized faces
- **Local Processing**: Face recognition runs locally
- **Audit Logging**: All security events are logged
- **Data Encryption**: Optional face embedding encryption

## 🚀 Deployment

### Development

```bash
# Run with auto-reload
streamlit run app.py --server.runOnSave true
```

### Production

```bash
# Set environment
export ENVIRONMENT=production

# Run with production settings
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

## 🔧 Troubleshooting

### Common Issues

**1. "No module named 'face_recognition'"**
```bash
# On macOS
brew install cmake
pip install dlib
pip install face_recognition

# On Ubuntu/Debian
sudo apt-get install cmake
pip install dlib
pip install face_recognition
```

**2. "Gemini API Error"**
- Check your API key in `.env` file
- Verify internet connection
- Ensure API quota is available

**3. "No faces detected"**
- Use clear, front-facing photos
- Ensure good lighting in images
- Check file formats (.jpg, .png supported)

**4. "Video processing failed"**
- Verify video file format
- Check file size (under 100MB default)
- Ensure video is not corrupted

### Performance Tips

- **Reduce frame interval** for faster processing
- **Lower video resolution** for better performance  
- **Limit authorized faces** to essential personnel only
- **Use SSD storage** for faster file operations

## 📈 Future Enhancements

- [ ] Real-time camera stream processing
- [ ] Mobile app for security notifications
- [ ] Integration with existing security systems
- [ ] Advanced behavior recognition models
- [ ] Multi-camera synchronization
- [ ] Cloud deployment options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For hackathon support or questions:
- Check the troubleshooting section above
- Review configuration in `config.py`
- Enable debug logging: `LOG_LEVEL=DEBUG` in `.env`

## 🎯 Hackathon Demo Script

### 🏆 For Judges (2-minute evaluation)

**Live Demo**: [Your Streamlit URL Here]

#### **Instant Demo (No Setup Required)**
1. **Visit live demo** → Click "🎲 Generate Enhanced Events"
2. **View Alerts** → Navigate to "🚨 Alerts Dashboard" 
3. **Check Priority System** → See 5-level threat classification
4. **Explore Features** → Visit all 6 main sections

#### **Advanced Demo (With Video Upload)**
1. **Upload Security Video** → Any .mp4 file 
2. **Watch Real-Time Processing** → YOLO + Face + Zone + Audio analysis
3. **Review Generated Alerts** → Intelligent threat assessment
4. **Test Gate Climbing Detection** → Upload fence/gate climbing video → Priority 4-5 Critical Alert

#### **Key Innovation Points**
- 🎯 **Semi-Supervised Learning**: Minimal labeling, maximum adaptability
- 🏫 **Zone-Based Intelligence**: Campus-specific contextual rules
- 🎵 **Multi-Modal Analysis**: Audio-visual correlation for enhanced accuracy
- 🔒 **Privacy-First**: GDPR-compliant face blurring
- 🤖 **Real AI Integration**: Actual Gemini 2.5 Pro analysis (not mocked)

### 📊 Technical Evaluation Checklist
- [ ] **Object Detection**: YOLOv8 real-time person/vehicle detection
- [ ] **Face Recognition**: Authorized vs unauthorized with clustering
- [ ] **Security Intelligence**: Context-aware threat assessment  
- [ ] **Privacy Compliance**: Automatic face anonymization
- [ ] **Scalability**: Production-ready architecture
- [ ] **Innovation**: Semi-supervised learning reduces operational overhead

### 🚨 Critical Security Detection Demo
**Upload a video showing gate/fence climbing** → System will:
1. Detect person with YOLOv8
2. Recognize unauthorized individual
3. Apply zone context (Main Gate security)
4. Generate **Priority 4-5 CRITICAL** alert: "Gate Climbing Intrusion Detected"
5. Recommend immediate security response

### Quick Demo Flow (5 minutes)

1. **Setup** (30s): Show project structure and configuration
2. **Face Management** (60s): Demonstrate authorized personnel setup
3. **Video Analysis** (120s): Upload sample video and show processing
4. **Alert Dashboard** (90s): Review generated alerts and actions
5. **AI Chat** (60s): Ask questions about security events
6. **Analytics** (30s): Show metrics and charts

### Sample Demo Data

```bash
# Generate sample events
python -c "
from core.video_processor import VideoProcessor
from core.gemini_client import GeminiSecurityAnalyzer
from core.alert_manager import AlertManager

vp = VideoProcessor()
for i in range(5):
    event = vp.generate_mock_detection()
    print(f'Generated event: {event[\"event_type\"]}')
"
```

---

**Built with ❤️ by KarmaVerse for Smart Campus Security**
