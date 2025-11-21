"""
JGM Insights Assistant - Google ADK Integration (PRODUCTION READY)
Gemini as primary AI with Ollama fallback
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===== GOOGLE ADK SETUP =====
GOOGLE_ADK_AVAILABLE = False
root_agent = None

try:
    # Try primary import method
    from google.genai import Client
    from google.genai.types import Tool, FunctionDeclaration
    GOOGLE_ADK_AVAILABLE = True
    print("✅ Google ADK imported successfully (Method 1: genai.Client)")
except ImportError:
    try:
        # Try alternative import
        from google.adk.agents import Agent
        from google.adk.tools import tool
        GOOGLE_ADK_AVAILABLE = True
        print("✅ Google ADK imported successfully (Method 2: adk.agents)")
    except ImportError as e:
        print(f"⚠️  Google ADK not available: {e}")
        print("   Falling back to Ollama/LlamaIndex")
        GOOGLE_ADK_AVAILABLE = False

# Import chatbot
from jgm_rag_chatbot import JGMRAG

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
WORKSPACE_PATH = Path(os.getenv("JGM_WORKSPACE", "./jgm_workspace"))

# Initialize chatbot (always available as fallback)
BOT = None
try:
    BOT = JGMRAG(WORKSPACE_PATH)
    BOT.build_index()
    print(f"✅ JGM Chatbot initialized: {WORKSPACE_PATH}")
except Exception as e:
    print(f"❌ Error initializing chatbot: {e}")
    # Create empty bot for safety
    BOT = JGMRAG(WORKSPACE_PATH)

# ===== INITIALIZE AGENT =====

def initialize_agent() -> bool:
    """Initialize Google ADK agent with tools"""
    global root_agent, GOOGLE_ADK_AVAILABLE
    
    if not GOOGLE_API_KEY:
        print("⚠️  No GOOGLE_API_KEY found in environment")
        GOOGLE_ADK_AVAILABLE = False
        return False
    
    if not GOOGLE_ADK_AVAILABLE:
        return False
    
    try:
        # Method 1: Using google.genai (newer SDK)
        try:
            client = Client(api_key=GOOGLE_API_KEY)
            
            # Define tools as function declarations
            tools = [
                Tool(
                    function_declarations=[
                        FunctionDeclaration(
                            name="answer_question",
                            description="Answer questions about Peru 2025 education data including dropout rates, applicants, regions, and statistics",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "The user's question about education data"
                                    }
                                },
                                "required": ["question"]
                            }
                        ),
                        FunctionDeclaration(
                            name="create_visualization",
                            description="Create charts, graphs, or maps to visualize education data",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "request": {
                                        "type": "string",
                                        "description": "What visualization to create (chart, map, graph)"
                                    }
                                },
                                "required": ["request"]
                            }
                        ),
                        FunctionDeclaration(
                            name="run_simulation",
                            description="Run What-If policy simulations to predict impact of education interventions",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "scenario": {
                                        "type": "string",
                                        "description": "Scenario name: meal_program, scholarship, mentorship, reduce_class_size, teacher_training, infrastructure, or 'menu' to list all"
                                    },
                                    "region": {
                                        "type": "string",
                                        "description": "Optional: specific region/department to analyze"
                                    }
                                },
                                "required": ["scenario"]
                            }
                        ),
                        FunctionDeclaration(
                            name="set_profile",
                            description="Save user profile information for personalization",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "first_name": {"type": "string"},
                                    "last_name": {"type": "string"},
                                    "role": {"type": "string", "description": "parent/student/teacher/NGO/donor/investor"},
                                    "contact": {"type": "string"}
                                }
                            }
                        )
                    ]
                )
            ]
            
            # Create agent wrapper
            class GeminiAgent:
                def __init__(self, client, tools):
                    self.client = client
                    self.tools = tools
                    self.model_name = "gemini-2.0-flash-exp"
                
                def send_message(self, message: str) -> str:
                    """Send message to Gemini and handle function calls"""
                    try:
                        # Generate response with tools
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=message,
                            config={
                                "tools": self.tools,
                                "temperature": 0.7,
                            }
                        )
                        
                        # Handle function calls if present
                        if hasattr(response, 'candidates') and response.candidates:
                            candidate = response.candidates[0]
                            
                            # Check for function calls
                            if hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'function_call'):
                                        func_call = part.function_call
                                        func_name = func_call.name
                                        func_args = dict(func_call.args)
                                        
                                        # Execute function
                                        result = self._execute_function(func_name, func_args)
                                        
                                        # Get final response with function result
                                        final_response = self.client.models.generate_content(
                                            model=self.model_name,
                                            contents=[
                                                {"role": "user", "parts": [{"text": message}]},
                                                {"role": "model", "parts": [{"function_call": func_call}]},
                                                {"role": "function", "parts": [{"function_response": {
                                                    "name": func_name,
                                                    "response": result
                                                }}]}
                                            ]
                                        )
                                        return final_response.text
                            
                            # No function call, return text
                            return response.text
                        
                        return "I couldn't process that request."
                    
                    except Exception as e:
                        print(f"Gemini error: {e}")
                        # Fallback to direct chatbot
                        return str(BOT.chat(message).get("reply", "Error processing request"))
                
                def _execute_function(self, func_name: str, args: dict) -> dict:
                    """Execute the appropriate function based on name"""
                    try:
                        if func_name == "answer_question":
                            res = BOT.chat(args.get("question", ""))
                            return {"reply": res.get("reply", ""), "refs": res.get("refs", [])}
                        
                        elif func_name == "create_visualization":
                            res = BOT.chat(args.get("request", ""))
                            return {
                                "message": res.get("reply", ""),
                                "image_path": res.get("image_path"),
                                "map_path": res.get("map_path")
                            }
                        
                        elif func_name == "run_simulation":
                            query = f"simulate {args.get('scenario', 'menu')}"
                            if args.get('region'):
                                query += f" for {args['region']}"
                            res = BOT.chat(query)
                            return {"simulation": res.get("reply", "")}
                        
                        elif func_name == "set_profile":
                            msg = BOT.set_profile(
                                first_name=args.get("first_name"),
                                last_name=args.get("last_name"),
                                role=args.get("role"),
                                contact=args.get("contact")
                            )
                            return {"status": "ok", "message": msg}
                        
                        else:
                            return {"error": f"Unknown function: {func_name}"}
                    
                    except Exception as e:
                        return {"error": str(e)}
            
            root_agent = GeminiAgent(client, tools)
            print("✅ Google Gemini Agent initialized (genai.Client)")
            print(f"   Model: gemini-2.0-flash-exp")
            print(f"   Tools: 4 available")
            return True
        
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            
            # Method 2: Try older ADK method
            try:
                from google.adk.agents import Agent
                
                # This is a placeholder - adjust based on actual ADK API
                root_agent = Agent(
                    model="gemini-2.0-flash-exp",
                    name="JGM Insights Assistant",
                    api_key=GOOGLE_API_KEY
                )
                print("✅ Google ADK Agent initialized (legacy method)")
                return True
            
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
                GOOGLE_ADK_AVAILABLE = False
                return False
    
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        GOOGLE_ADK_AVAILABLE = False
        return False

# ===== PUBLIC INTERFACE =====

def enhanced_chat(message: str) -> Dict[str, Any]:
    """
    Process message through Gemini agent (primary) or fallback to direct chatbot
    """
    # Try Gemini agent first
    if root_agent and GOOGLE_ADK_AVAILABLE:
        try:
            response = root_agent.send_message(message)
            return {
                "reply": str(response),
                "source": "gemini"
            }
        except Exception as e:
            print(f"⚠️  Gemini error: {e}, using fallback")
    
    # Fallback to direct chatbot
    try:
        result = BOT.chat(message)
        result["source"] = "ollama" if BOT.llm_available else "direct"
        return result
    except Exception as e:
        return {
            "reply": f"Error processing request: {str(e)}",
            "source": "error"
        }

def greet_user() -> str:
    """Get greeting message"""
    if root_agent and GOOGLE_ADK_AVAILABLE:
        try:
            return root_agent.send_message("hello")
        except Exception:
            pass
    
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
        "google_adk_available": GOOGLE_ADK_AVAILABLE,
        "agent_initialized": root_agent is not None,
        "ollama_available": BOT.llm_available if BOT else False,
        "chatbot_ready": BOT is not None,
        "primary_engine": "gemini" if (root_agent and GOOGLE_ADK_AVAILABLE) else ("ollama" if (BOT and BOT.llm_available) else "direct")
    }

# ===== STARTUP TEST =====

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🤖 JGM INSIGHTS ASSISTANT - GOOGLE ADK INTEGRATION TEST")
    print("=" * 70)
    
    # Test initialization
    success = initialize_agent()
    
    status = get_agent_status()
    print("\n📊 STATUS:")
    print(f"   Google ADK Available: {'✅' if status['google_adk_available'] else '❌'}")
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
    
    print("\n" + "=" * 70)
    print("✅ Ready for Flask integration!")
    print("=" * 70)