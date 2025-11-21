#!/usr/bin/env python3
"""
JGM Insights Assistant - Quick Start Script
Run this to start the application with automatic checks
"""

import os
import sys
from pathlib import Path

def check_env():
    """Check if .env exists"""
    if not Path(".env").exists():
        print("⚠️  .env file not found!")
        print("   Creating from .env.example...")
        try:
            if Path(".env.example").exists():
                import shutil
                shutil.copy(".env.example", ".env")
                print("✅ Created .env file")
                print("⚠️  IMPORTANT: Edit .env and add your GOOGLE_API_KEY!")
                print("   Current key is already included, but verify it's correct.")
                return False
            else:
                print("❌ .env.example not found!")
                return False
        except Exception as e:
            print(f"❌ Could not create .env: {e}")
            return False
    return True

def check_workspace():
    """Ensure workspace exists"""
    from dotenv import load_dotenv
    load_dotenv()
    
    workspace = Path(os.getenv("JGM_WORKSPACE", "jgm_workspace"))
    directories = ["data", "graphs", "code", "transcripts"]
    
    for dir_name in directories:
        dir_path = workspace / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Workspace ready: {workspace}")

def check_dependencies():
    """Check critical dependencies"""
    try:
        import flask
        import pandas
        import dotenv
        print("✅ Core dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def main():
    print("\n" + "🤖" * 35)
    print("  JGM INSIGHTS ASSISTANT - STARTING")
    print("🤖" * 35 + "\n")
    
    # Checks
    if not check_env():
        print("\n❌ Please configure .env file first!")
        print("   1. Open .env file")
        print("   2. Verify GOOGLE_API_KEY is set")
        print("   3. Run this script again")
        return 1
    
    if not check_dependencies():
        print("\n❌ Please install dependencies first!")
        return 1
    
    check_workspace()
    
    print("\n" + "🚀" * 35)
    print("  STARTING APPLICATION...")
    print("🚀" * 35 + "\n")
    
    # Import and run app
    try:
        import app
        
        print(f"✅ Application starting on http://{app.HOST}:{app.PORT}")
        print(f"✅ Workspace: {app.WORKSPACE}")
        print(f"✅ Mode: {'PRODUCTION' if app.PRODUCTION_MODE else 'DEVELOPMENT'}")
        print("\n📌 Press Ctrl+C to stop\n")
        
        app.app.run(
            host=app.HOST,
            port=app.PORT,
            debug=app.DEBUG,
            threaded=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Application failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check .env configuration")
        print("  2. Run: python test_system.py")
        print("  3. See DEPLOYMENT_GUIDE.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
