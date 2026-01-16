"""
JGM Insights Assistant - Google Gemini Integration
UPDATED: Now allows general knowledge questions via Gemini
Data queries still go through the chatbot for accurate Peru education data
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
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'gemini-pro',
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

# ===== QUERY CLASSIFICATION =====

def _is_peru_education_query(message: str) -> bool:
    """
    Determine if the query is about Peru education data
    Returns True if it should be handled by the chatbot (BOT)
    Returns False if it should go to Gemini for general knowledge
    """
    msg_lower = message.lower()
    
    # Peru education specific keywords
    peru_education_keywords = [
        # Data queries
        'dropout', 'desercion', 'tasa', 'rate',
        'applicant', 'aplicante', 'undergraduate', 'postulante',
        'faculty', 'facultad', 'program', 'programa',
        'department', 'departamento', 'region', 'province', 'provincia',
        
        # Peru locations (when combined with education context)
        'lima dropout', 'cusco school', 'arequipa education',
        
        # Education terms
        'primaria', 'primary school', 'secundaria', 'secondary school',
        'escuela', 'colegio', 'universidad',
        
        # Visualization/analysis commands
        'map', 'chart', 'graph', 'plot', 'visualize',
        'compare dropout', 'comparison rate', 'statistics dropout',
        
        # What-If simulator
        'simulate', 'simulation', 'what if', 'what-if', 'scenario',
        'roi', 'intervention', 'policy impact',
        'meal program', 'scholarship impact', 'mentorship program', 
        'teacher training impact', 'infrastructure improvement', 'class size',
        
        # Data year combined with education
        '2025 dropout', '2025 education', '2025 school'
    ]
    
    # Check for education-specific keyword combinations
    if any(kw in msg_lower for kw in peru_education_keywords):
        return True
    
    # Check for commands that should go to chatbot
    chatbot_commands = ['summary', 'summarize', 'simulate menu', 'show map', 'create map', 'build map']
    if any(cmd in msg_lower for cmd in chatbot_commands):
        return True
    
    # Check if asking about Peru AND education together
    peru_terms = ['peru', 'peruvian', 'lima', 'cusco', 'arequipa', 'piura', 'puno']
    education_terms = ['school', 'education', 'student', 'dropout', 'university', 'college']
    
    has_peru = any(p in msg_lower for p in peru_terms)
    has_education = any(e in msg_lower for e in education_terms)
    
    if has_peru and has_education:
        return True
    
    return False

# ===== PUBLIC INTERFACE =====

def enhanced_chat(message: str) -> Dict[str, Any]:
    """
    Process message through appropriate system:
    - Peru education queries → BOT (chatbot with actual data)
    - General knowledge → Gemini AI
    - Fallback → Basic response
    """
    
    # Check if this is a Peru education query
    if _is_peru_education_query(message):
        # Use chatbot for Peru education data (it has the actual data)
        try:
            result = BOT.chat(message)
            if result:  # Chatbot returned a valid response
                result["source"] = "chatbot"
                return result
        except Exception as e:
            print(f"Chatbot error: {e}")
    
    # For general knowledge questions OR if chatbot returned None, use Gemini
    if GOOGLE_AVAILABLE and GOOGLE_API_KEY:
        try:
            # Ensure model is initialized
            global GEMINI_MODEL
            if GEMINI_MODEL is None:
                GEMINI_MODEL = get_best_model()
            
            if GEMINI_MODEL:
                response = call_gemini(f"""You are a helpful AI assistant for JGM Organization's education analytics platform.

You can answer ANY question - general knowledge, history, science, technology, current events, people, companies, etc.

You also specialize in Peru education data analysis. If someone asks about Peru education specifically, 
you can mention they can try queries like:
- "Show dropout rates by region"
- "Create a map of education metrics"
- "Simulate meal program impact"

User's question: {message}

Provide a helpful, accurate, and informative response. Be conversational and friendly.
If asked about a person, company, or topic, give comprehensive information.""")
                
                if response:
                    return {
                        "reply": response,
                        "source": "gemini"
                    }
        
        except Exception as e:
            print(f"⚠️  Gemini error: {e}")
    
    # Fallback response if nothing else works
    return {
        "reply": (
            "I'm sorry, I couldn't process that request. I can help you with:\n\n"
            "📊 **Peru Education Data:**\n"
            "  • Dropout rates by region\n"
            "  • Undergraduate applicant statistics\n"
            "  • Interactive maps and charts\n"
            "  • What-If policy simulations\n\n"
            "🌍 **General Questions:**\n"
            "  • Ask me about any topic!\n\n"
            "Please try rephrasing your question."
        ),
        "source": "fallback"
    }

def greet_user() -> str:
    """Get greeting message"""
    return BOT.greet_and_collect()

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
        "primary_engine": "gemini" if (GOOGLE_AVAILABLE and GEMINI_MODEL) else ("ollama" if (BOT and BOT.llm_available) else "direct"),
        "general_knowledge_enabled": GOOGLE_AVAILABLE and GEMINI_MODEL is not None
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
    print(f"   General Knowledge: {'✅ ENABLED' if status['general_knowledge_enabled'] else '❌ DISABLED'}")
    print(f"   Chatbot Ready: {'✅' if status['chatbot_ready'] else '❌'}")
    print(f"   Primary Engine: {status['primary_engine'].upper()}")
    
    if success:
        print("\n🧪 TESTING...")
        
        # Test general knowledge
        print("\n🌍 Testing: 'Who is George Washington?'")
        try:
            result = enhanced_chat("Who is George Washington?")
            print(f"   Source: {result.get('source', 'unknown')}")
            print(f"   Response: {result['reply'][:150]}...")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test education query
        print("\n📚 Testing: 'What is the dropout rate?'")
        try:
            result = enhanced_chat("What is the dropout rate?")
            print(f"   Source: {result.get('source', 'unknown')}")
            print(f"   Response: {result['reply'][:150]}...")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Agent ready!")
    print("=" * 70)
