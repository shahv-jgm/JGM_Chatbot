#!/usr/bin/env python3
"""
Diagnostic script to check Google API key status
Run this before starting the app
"""

import os
from pathlib import Path
from dotenv import load_dotenv

print("=" * 70)
print("🔍 API KEY DIAGNOSTIC TOOL")
print("=" * 70)

# Check current directory
current_dir = Path.cwd()
print(f"\n1. Current directory: {current_dir}")

# Check if .env exists
env_file = current_dir / ".env"
print(f"\n2. .env file exists: {env_file.exists()}")
if env_file.exists():
    print(f"   Location: {env_file}")
else:
    print("   ❌ ERROR: .env file not found!")
    print("   Create it with: echo 'GOOGLE_API_KEY=your_key_here' > .env")
    exit(1)

# Load .env
print(f"\n3. Loading .env file...")
load_dotenv()

# Check if key is loaded
api_key = os.getenv("GOOGLE_API_KEY")
print(f"\n4. GOOGLE_API_KEY loaded: {bool(api_key)}")

if not api_key:
    print("   ❌ ERROR: GOOGLE_API_KEY not found in environment!")
    print("   Add it to .env file: GOOGLE_API_KEY=your_key_here")
    exit(1)

print(f"   Key preview: {api_key[:20]}...")
print(f"   Key length: {len(api_key)} characters")
print(f"   Starts with 'AIzaSy': {api_key.startswith('AIzaSy')}")

if not api_key.startswith("AIzaSy"):
    print("   ⚠️  WARNING: Key doesn't look like a valid Google API key!")
    print("   Valid keys start with 'AIzaSy'")

# Test the key with Google
print(f"\n5. Testing API key with Google Gemini...")
try:
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # Try multiple models in order of preference
    models_to_try = ['gemini-pro', 'gemini-1.5-pro', 'models/gemini-pro']
    
    model = None
    model_name_used = None
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            model_name_used = model_name
            print(f"   Using model: {model_name}")
            break
        except Exception:
            continue
    
    if not model:
        print("   ❌ ERROR: No compatible models found!")
        print("   Listing available models...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"     - {m.name}")
        exit(1)
    
    print("   Sending test request...")
    response = model.generate_content("Say 'test successful' if you can read this")
    
    print("   ✅ SUCCESS! Your API key works!")
    print(f"   Model used: {model_name_used}")
    print(f"   Response: {response.text[:100]}")
    print("\n" + "=" * 70)
    print("✅ Your API key is VALID and WORKING!")
    print("=" * 70)
    
except Exception as e:
    error_msg = str(e)
    print(f"   ❌ FAILED: {error_msg}")
    
    if "expired" in error_msg.lower() or "invalid" in error_msg.lower():
        print("\n" + "=" * 70)
        print("❌ YOUR API KEY IS EXPIRED OR INVALID!")
        print("=" * 70)
        print("\n🔧 HOW TO FIX:")
        print("1. Go to: https://aistudio.google.com/app/apikey")
        print("2. DELETE all old keys")
        print("3. Click 'Create API Key'")
        print("4. Copy the NEW key")
        print("5. Update .env file:")
        print(f"   nano {env_file}")
        print("   Replace with: GOOGLE_API_KEY=your_new_key_here")
        print("6. Run this diagnostic again")
        print("=" * 70)
    elif "not found" in error_msg.lower():
        print("\n" + "=" * 70)
        print("⚠️  MODEL NOT AVAILABLE")
        print("=" * 70)
        print("\nListing available models...")
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            print("\nAvailable models:")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f"  ✅ {m.name}")
        except Exception as list_error:
            print(f"Could not list models: {list_error}")
        print("=" * 70)
    else:
        print(f"\n⚠️  Error: {error_msg}")
    
    exit(1)