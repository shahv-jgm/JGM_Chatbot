#!/usr/bin/env python3
"""
Quick Test Script for JGM Insights Assistant
Run this to verify everything is working before deployment
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def check_environment():
    """Check environment configuration"""
    print_header("1. CHECKING ENVIRONMENT")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    checks = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "FLASK_PORT": os.getenv("FLASK_PORT", "5050"),
        "WORKSPACE": os.getenv("JGM_WORKSPACE", "jgm_workspace")
    }
    
    for key, value in checks.items():
        status = "✅" if value else "❌"
        display_value = value[:20] + "..." if (value and len(value) > 20) else value
        print(f"{status} {key}: {display_value}")
    
    return all(checks.values())

def check_dependencies():
    """Check required packages"""
    print_header("2. CHECKING DEPENDENCIES")
    
    required = [
        "flask",
        "pandas",
        "dotenv",
        "rapidfuzz",
        "sklearn",
        "matplotlib",
        "folium"
    ]
    
    all_installed = True
    for package in required:
        try:
            if package == "dotenv":
                __import__("dotenv")
            elif package == "sklearn":
                __import__("sklearn")
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_google_adk():
    """Check Google ADK"""
    print_header("3. CHECKING GOOGLE ADK")
    
    try:
        from google.genai import Client
        print("✅ Method 1: google.genai.Client")
        return True
    except ImportError:
        try:
            from google.adk.agents import Agent
            print("✅ Method 2: google.adk.agents.Agent")
            return True
        except ImportError:
            print("❌ Google ADK not installed")
            print("   Run: pip install google-adk")
            return False

def check_workspace():
    """Check workspace structure"""
    print_header("4. CHECKING WORKSPACE")
    
    workspace = Path(os.getenv("JGM_WORKSPACE", "jgm_workspace"))
    
    directories = {
        "Workspace": workspace,
        "Data": workspace / "data",
        "Graphs": workspace / "graphs",
        "Code": workspace / "code",
        "Transcripts": workspace / "transcripts"
    }
    
    all_exist = True
    for name, path in directories.items():
        if path.exists():
            file_count = len(list(path.glob("*"))) if path.is_dir() else 0
            print(f"✅ {name}: {path} ({file_count} files)")
        else:
            print(f"⚠️  {name}: {path} - WILL BE CREATED")
            all_exist = False
    
    return True  # Not critical if they don't exist yet

def test_agent():
    """Test agent initialization"""
    print_header("5. TESTING AGENT")
    
    try:
        from agent import initialize_agent, get_agent_status, BOT
        
        # Initialize
        success = initialize_agent()
        
        # Get status
        status = get_agent_status()
        
        print(f"\nAgent Status:")
        print(f"  Google ADK Available: {'✅' if status['google_adk_available'] else '❌'}")
        print(f"  Agent Initialized: {'✅' if status['agent_initialized'] else '❌'}")
        print(f"  Ollama Available: {'✅' if status['ollama_available'] else '❌'}")
        print(f"  Chatbot Ready: {'✅' if status['chatbot_ready'] else '❌'}")
        print(f"  Primary Engine: {status['primary_engine'].upper()}")
        
        # Test simple chat
        if status['chatbot_ready']:
            print(f"\n🧪 Testing conversation...")
            try:
                from agent import enhanced_chat
                response = enhanced_chat("hello")
                print(f"✅ Response received: {response['reply'][:50]}...")
                return True
            except Exception as e:
                print(f"⚠️  Conversation test failed: {e}")
                return False
        
        return success
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def test_flask_app():
    """Test Flask app can start"""
    print_header("6. TESTING FLASK APP")
    
    try:
        # Just import, don't run
        import app
        print("✅ Flask app imports successfully")
        print(f"   Configured for: {app.HOST}:{app.PORT}")
        return True
    except Exception as e:
        print(f"❌ Flask app failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "🤖" * 35)
    print("  JGM INSIGHTS ASSISTANT - SYSTEM CHECK")
    print("🤖" * 35)
    
    results = {
        "Environment": check_environment(),
        "Dependencies": check_dependencies(),
        "Google ADK": check_google_adk(),
        "Workspace": check_workspace(),
        "Agent": test_agent(),
        "Flask App": test_flask_app()
    }
    
    print_header("SUMMARY")
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "🎉" * 35)
        print("  ALL TESTS PASSED!")
        print("  You're ready to run: python app.py")
        print("🎉" * 35)
        return 0
    else:
        print("\n" + "⚠️ " * 35)
        print("  SOME TESTS FAILED")
        print("  Check the errors above and fix them")
        print("  See DEPLOYMENT_GUIDE.md for help")
        print("⚠️ " * 35)
        return 1

if __name__ == "__main__":
    sys.exit(main())