"""
JGM Insights Assistant - Google Gemini Integration (FIXED)
Direct google.generativeai integration with proper model selection
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ===== GOOGLE GEMINI SETUP =====
GOOGLE_AVAILABLE = False

try:
    import google.generativeai as genai
    
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        GOOGLE_AVAILABLE = True
        print("✅ Google Gemini configured successfully")
        print(f"   API Key: {GOOGLE_API_KEY[:20]}...")
    else:
        print("⚠️  No GOOGLE_API_KEY found in environment")
        
except ImportError as e:
    print(f"⚠️  google-generativeai not installed: {e}")
    print("   Install with: pip install google-generativeai")
    GOOGLE_AVAILABLE = False
except Exception as e:
    print(f"⚠️  Error configuring Google Gemini: {e}")
    GOOGLE_AVAILABLE = False

# Import chatbot
from jgm_rag_chatbot import JGMRAG

# Configuration
WORKSPACE_PATH = Path(os.getenv("JGM_WORKSPACE", "./jgm_workspace"))

# Initialize chatbot (always available as fallback)
BOT = None
try:
    BOT = JGMRAG(WORKSPACE_PATH)
    BOT.build_index()
    print(f"✅ JGM Chatbot initialized: {WORKSPACE_PATH}")
except Exception as e:
    print(f"❌ Error initializing chatbot: {e}")
    BOT = JGMRAG(WORKSPACE_PATH)

# ===== GEMINI MODEL SELECTION =====

def get_best_model():
    """Try models in order of preference"""
    if not GOOGLE_AVAILABLE:
        return None
    
    models_to_try = [
        'gemini-pro',
        'gemini-1.5-pro', 
        'models/gemini-pro',
        'gemini-1.0-pro'
    ]
    
    import google.generativeai as genai
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            print(f"✅ Using Gemini model: {model_name}")
            return model
        except Exception:
            continue
    
    # If none work, list available models
    print("⚠️  Could not find compatible model. Available models:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"   - {m.name}")
    except Exception as e:
        print(f"   Could not list models: {e}")
    
    return None

# ===== GEMINI HELPER =====

GEMINI_MODEL = None

def call_gemini(prompt: str) -> Optional[str]:
    """
    Call Google Gemini with error handling
    Returns None if fails
    """
    global GEMINI_MODEL
    
    if not GOOGLE_AVAILABLE or not GOOGLE_API_KEY:
        return None
    
    try:
        import google.generativeai as genai
        
        # Initialize model if not already done
        if GEMINI_MODEL is None:
            GEMINI_MODEL = get_best_model()
        
        if GEMINI_MODEL is None:
            print("⚠️  No Gemini model available")
            return None
        
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        print(f"Gemini error: {error_msg[:200]}")
        
        # Log specific error types
        if "expired" in error_msg.lower():
            print("❌ API KEY EXPIRED - Get new key at: https://aistudio.google.com/app/apikey")
        elif "invalid" in error_msg.lower():
            print("❌ API KEY INVALID - Check your .env file")
        elif "quota" in error_msg.lower():
            print("⚠️  API QUOTA EXCEEDED - Wait or upgrade plan")
        elif "not found" in error_msg.lower():
            print("⚠️  MODEL NOT FOUND - Trying to reinitialize...")
            GEMINI_MODEL = get_best_model()
        
        return None

# ===== INITIALIZE AGENT =====

def initialize_agent() -> bool:
    """Initialize Google agent"""
    global GOOGLE_AVAILABLE, GEMINI_MODEL
    
    if not GOOGLE_API_KEY:
        print("⚠️  No GOOGLE_API_KEY - running in fallback mode")
        return False
    
    if not GOOGLE_AVAILABLE:
        print("⚠️  Google Gemini not available")
        return False
    
    try:
        GEMINI_MODEL = get_best_model()
        
        if GEMINI_MODEL:
            # Test the connection
            test_response = call_gemini("Say 'ready'")
            if test_response:
                print("✅ Google Gemini initialized and tested")
                return True
            else:
                print("⚠️  Gemini test failed - using fallback mode")
                return False
        else:
            print("⚠️  No compatible Gemini model found")
            return False
            
    except Exception as e:
        print(f"❌ Failed to initialize Gemini: {e}")
        GOOGLE_AVAILABLE = False
        return False

# ===== PUBLIC INTERFACE =====

def enhanced_chat(message: str) -> Dict[str, Any]:
    """
    Process message through Gemini (primary) or fallback to chatbot
    """
    # Try Gemini first if available
    if GOOGLE_AVAILABLE and GOOGLE_API_KEY and GEMINI_MODEL:
        try:
            # For data queries, always use chatbot (it has the actual data)
            data_keywords = ['dropout', 'rate', 'data', 'applicant', 'region', 
                           'map', 'chart', 'simulate', 'faculty', 'department',
                           'province', 'statistics', 'compare', 'show']
            
            if any(kw in message.lower() for kw in data_keywords):
                # Use chatbot for data queries
                result = BOT.chat(message)
                result["source"] = "chatbot"
                return result
            
            # For casual conversation, use Gemini
            response = call_gemini(f"""You are a helpful Peru education data assistant.
User said: {message}

Give a brief, friendly response. If they're asking about data, acknowledge it and suggest they ask specific questions.""")
            
            if response:
                return {
                    "reply": response,
                    "source": "gemini"
                }
        
        except Exception as e:
            print(f"⚠️  Gemini error, using fallback: {e}")
    
    # Fallback to direct chatbot
    try:
        result = BOT.chat(message)
        result["source"] = "chatbot"
        return result
    except Exception as e:
        return {
            "reply": f"Error processing request: {str(e)}",
            "source": "error"
        }

def greet_user() -> str:
    """Get greeting message"""
    greeting = BOT.greet_and_collect()
    
    # Try to enhance with Gemini if available
    if GOOGLE_AVAILABLE and GEMINI_MODEL:
        try:
            enhanced = call_gemini("Give a very brief, friendly greeting for a Peru education data assistant. One sentence only.")
            if enhanced:
                return enhanced + "\n\n" + greeting
        except Exception:
            pass
    
    return greeting

def set_user_profile(first_name="", last_name="", role="", contact="") -> str:
    """Set user profile"""
    return BOT.set_profile(
        first_name=first_name or None,
        last_name=last_name or None,
        role=role or None,
        contact=contact or None
    )

def get_agent_status() -> Dict[str, Any]:
    """Get current agent status for monitoring"""
    return {
        "google_api_key_set": bool(GOOGLE_API_KEY),
        "google_available": GOOGLE_AVAILABLE,
        "gemini_model_ready": GEMINI_MODEL is not None,
        "agent_initialized": GOOGLE_AVAILABLE and GEMINI_MODEL is not None,
        "ollama_available": BOT.llm_available if BOT else False,
        "chatbot_ready": BOT is not None,
        "primary_engine": "gemini" if (GOOGLE_AVAILABLE and GEMINI_MODEL) else ("ollama" if (BOT and BOT.llm_available) else "direct")
    }

# ===== STARTUP TEST =====

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🤖 JGM INSIGHTS ASSISTANT - GOOGLE GEMINI TEST")
    print("=" * 70)
    
    # Test initialization
    success = initialize_agent()
    
    status = get_agent_status()
    print("\n📊 STATUS:")
    print(f"   Google API Key Set: {'✅' if status['google_api_key_set'] else '❌'}")
    print(f"   Google Available: {'✅' if status['google_available'] else '❌'}")
    print(f"   Gemini Model Ready: {'✅' if status['gemini_model_ready'] else '❌'}")
    print(f"   Agent Initialized: {'✅' if status['agent_initialized'] else '❌'}")
    print(f"   Ollama Available: {'✅' if status['ollama_available'] else '❌'}")
    print(f"   Chatbot Ready: {'✅' if status['chatbot_ready'] else '❌'}")
    print(f"   Primary Engine: {status['primary_engine'].upper()}")
    
    if success:
        print("\n🧪 TESTING AGENT...")
        try:
            test_response = enhanced_chat("hello")
            print(f"✅ Test passed!")
            print(f"   Response: {test_response['reply'][:100]}...")
            print(f"   Source: {test_response.get('source', 'unknown')}")
        except Exception as e:
            print(f"⚠️  Test failed: {e}")
    else:
        print("\n⚠️  Agent initialization failed - will use fallback mode")
    
    print("\n" + "=" * 70)
    print("✅ Ready for Flask integration!")
    print("=" * 70)