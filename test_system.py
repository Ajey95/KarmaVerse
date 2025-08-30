#!/usr/bin/env python3
"""
VisionGuard AI - System Test Script
Quick test to verify all components are working.
"""

import os
import sys
import traceback
from datetime import datetime

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("  ✓ Streamlit")
        
        import cv2
        print("  ✓ OpenCV")
        
        import face_recognition
        print("  ✓ Face Recognition")
        
        import google.generativeai as genai
        print("  ✓ Google Generative AI")
        
        import pandas as pd
        print("  ✓ Pandas")
        
        import numpy as np
        print("  ✓ NumPy")
        
        import plotly.express as px
        print("  ✓ Plotly")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_core_modules():
    """Test if our core modules can be imported and initialized."""
    print("\n🧩 Testing core modules...")
    
    try:
        # Add current directory to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from core.video_processor import VideoProcessor
        vp = VideoProcessor()
        print("  ✓ VideoProcessor")
        
        from core.face_verification import FaceVerificationSystem
        fv = FaceVerificationSystem()
        print("  ✓ FaceVerificationSystem")
        
        from core.detection_engine import ObjectDetectionEngine, ZoneContextManager
        ode = ObjectDetectionEngine()
        zcm = ZoneContextManager()
        print("  ✓ ObjectDetectionEngine & ZoneContextManager")
        
        from core.audio_anomaly_detector import AudioAnomalyDetector
        aad = AudioAnomalyDetector()
        print("  ✓ AudioAnomalyDetector")
        
        from core.alert_manager import AlertManager
        am = AlertManager()
        print("  ✓ AlertManager")
        
        # Test Gemini only if API key is available
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key and api_key != 'your_gemini_api_key_here':
            from core.gemini_client import GeminiSecurityAnalyzer
            ga = GeminiSecurityAnalyzer()
            print("  ✓ GeminiSecurityAnalyzer")
        else:
            print("  ⚠️  GeminiSecurityAnalyzer (API key not configured)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Module error: {e}")
        traceback.print_exc()
        return False

def test_directories():
    """Test if required directories exist."""
    print("\n📁 Testing directories...")
    
    required_dirs = [
        "data",
        "data/authorized_faces",
        "data/alerts", 
        "data/logs",
        "core"
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✓ {directory}")
        else:
            print(f"  ❌ {directory} - MISSING")
            all_exist = False
    
    return all_exist

def test_enhanced_detection():
    """Test enhanced detection with zones and audio."""
    print("\n🎯 Testing enhanced detection systems...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from core.video_processor import VideoProcessor
        from core.audio_anomaly_detector import AudioAnomalyDetector
        from core.detection_engine import ZoneContextManager
        
        vp = VideoProcessor()
        aad = AudioAnomalyDetector()
        zcm = ZoneContextManager()
        
        # Test zone configuration
        zones = zcm.zones
        print(f"  ✓ Campus zones configured: {len(zones)}")
        
        # Test audio anomaly types
        anomaly_types = aad.anomaly_types
        print(f"  ✓ Audio anomaly types: {len(anomaly_types)}")
        
        # Test enhanced mock detection
        enhanced_event = vp.generate_mock_detection()
        if 'zone_context' in enhanced_event or 'detected_objects' in enhanced_event:
            print("  ✓ Enhanced detection with zone context")
        else:
            print("  ⚠️  Basic detection (enhanced features may not be active)")
        
        # Test audio simulation
        audio_detection = aad.simulate_audio_detection("main_gate", "unauthorized_entry")
        if audio_detection:
            print(f"  ✓ Audio detection simulation: {audio_detection['anomaly_type']}")
        else:
            print("  ✓ Audio detection (no anomaly generated)")
        
        return True
    except Exception as e:
        print(f"  ❌ Enhanced detection error: {e}")
        return False

def test_alert_management():
    """Test alert saving and loading."""
    print("\n🚨 Testing alert management...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from core.alert_manager import AlertManager
        from core.video_processor import VideoProcessor
        
        am = AlertManager()
        vp = VideoProcessor()
        
        # Generate a test event
        test_event = vp.generate_mock_detection()
        
        # Create a test alert
        test_alert = {
            'alert': '🚨 Test Alert',
            'priority': 3,
            'recommended_action': 'Test action',
            'explanation': 'This is a test alert',
            'source_event': test_event,
            'analysis_timestamp': datetime.now().isoformat(),
            'analyzer': 'Test System'
        }
        
        # Save alert
        alert_id = am.save_alert(test_alert)
        if alert_id:
            print(f"  ✓ Alert saved with ID: {alert_id}")
        else:
            print("  ❌ Failed to save alert")
            return False
        
        # Load alerts
        alerts = am.load_alerts(limit=1)
        if alerts:
            print(f"  ✓ Alert loaded: {alerts[0]['alert']}")
        else:
            print("  ❌ Failed to load alerts")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Alert management error: {e}")
        return False

def test_gemini_connection():
    """Test Gemini API connection (if configured)."""
    print("\n🤖 Testing Gemini connection...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("  ⚠️  Skipping - API key not configured")
        print("     Add your Gemini API key to .env file to test")
        return True
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from core.gemini_client import GeminiSecurityAnalyzer
        
        ga = GeminiSecurityAnalyzer()
        health = ga.health_check()
        
        if health.get('status') == 'healthy':
            print("  ✓ Gemini API connection successful")
            return True
        else:
            print(f"  ❌ Gemini API error: {health.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"  ❌ Gemini connection error: {e}")
        return False

def run_full_test():
    """Run complete system test."""
    print("🛡️  VisionGuard AI - System Test")
    print("="*40)
    
    tests = [
        ("Imports", test_imports),
        ("Directories", test_directories), 
        ("Core Modules", test_core_modules),
        ("Enhanced Detection", test_enhanced_detection),
        ("Alert Management", test_alert_management),
        ("Gemini Connection", test_gemini_connection)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*40)
    print("📊 Test Results Summary")
    print("="*40)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to run.")
        print("   Run: streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("   Try running: python setup.py")
    
    return passed == total

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)