"""
JGM Insights Assistant - Flask App
Fixed session management - profiles stored PER SESSION, not globally
Each user gets their own session via X-Session-ID header
"""

import os
import uuid
import datetime
import json
from pathlib import Path
from flask import (
    Flask, request, jsonify, send_from_directory,
    render_template_string, make_response, Response
)
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import agent system
try:
    from agent import (
        enhanced_chat, get_agent_status, initialize_agent,
        BOT, GOOGLE_ADK_AVAILABLE
    )
    AGENT_SYSTEM_AVAILABLE = True
    print("✅ Agent system loaded successfully")
except Exception as e:
    AGENT_SYSTEM_AVAILABLE = False
    print(f"❌ Agent system failed: {e}")
    print("   Using direct chatbot fallback")
    try:
        from jgm_rag_chatbot import JGMRAG
    except Exception as e2:
        print(f"❌ Fallback also failed: {e2}")
        JGMRAG = None

# ===== CONFIGURATION =====
HOST = os.getenv("FLASK_HOST", "0.0.0.0")
PORT = int(os.getenv("FLASK_PORT", "5050"))
DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())
PRODUCTION_MODE = os.getenv("PRODUCTION_MODE", "False").lower() == "true"

# Paths
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = BASE_DIR / os.getenv("JGM_WORKSPACE", "jgm_workspace")
DATA_DIR = WORKSPACE / "data"
GRAPHS_DIR = WORKSPACE / "graphs"
CODE_DIR = WORKSPACE / "code"
TRANS_DIR = WORKSPACE / "transcripts"
LOGS_DIR = BASE_DIR / "logs"

for folder in [WORKSPACE, DATA_DIR, GRAPHS_DIR, CODE_DIR, TRANS_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ===== FLASK APP =====
app = Flask(__name__)
app.secret_key = SECRET_KEY

# ===== CORS CONFIGURATION =====
# Allow requests from Vercel frontend and localhost for development
CORS(app, 
    origins=[
        "https://jgm-ds-website.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ], 
    supports_credentials=True,
    allow_headers=["Content-Type", "X-Session-ID"],
    expose_headers=["X-Session-ID"]
)

# ===== INITIALIZE SYSTEM =====
bot = None
if AGENT_SYSTEM_AVAILABLE:
    agent_ready = initialize_agent()
    if agent_ready:
        print("✅ Google Gemini agent active")
    else:
        print("⚠️  Using fallback chatbot")
else:
    if JGMRAG:
        bot = JGMRAG(WORKSPACE)
        bot.build_index()
        print("✅ Direct chatbot initialized")
    else:
        print("⚠️  No chatbot available - basic responses only")

# ===== SESSION STORAGE =====
# Each session: {"messages": [], "profile": {}}
# Profile is stored PER SESSION - not globally!
SESS = {}

def _get_sid():
    """Get session ID from header, body, cookie, or generate new"""
    sid = None
    
    # 1. Check X-Session-ID header first (frontend sends this)
    sid = request.headers.get("X-Session-ID")
    
    # 2. Check request body
    if not sid:
        try:
            if request.is_json and request.json:
                sid = request.json.get("session_id")
        except Exception:
            pass
    
    # 3. Fallback to cookie
    if not sid:
        sid = request.cookies.get("session_id")
    
    # 4. Generate new if none found
    if not sid:
        sid = str(uuid.uuid4())
        print(f"🆕 New session created: {sid[:8]}...")
    
    # Initialize session storage
    if sid not in SESS:
        SESS[sid] = {"messages": [], "profile": {}}
        tfile = TRANS_DIR / f"{sid}.json"
        if tfile.exists():
            try:
                data = json.loads(tfile.read_text(encoding="utf-8"))
                # Handle old format (list) vs new format (dict)
                if isinstance(data, list):
                    SESS[sid] = {"messages": data, "profile": {}}
                else:
                    SESS[sid] = data
            except Exception:
                SESS[sid] = {"messages": [], "profile": {}}
    
    return sid

def _save_session(sid):
    """Save session to disk"""
    try:
        (TRANS_DIR / f"{sid}.json").write_text(
            json.dumps(SESS[sid], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        print(f"Error saving session: {e}")

def _record(sid, role, text, attachments=None):
    """Record conversation to session"""
    SESS[sid]["messages"].append({
        "role": role,
        "text": text,
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "attachments": attachments or []
    })
    _save_session(sid)

def _get_profile(sid):
    """Get profile for a session"""
    return SESS.get(sid, {}).get("profile", {})

def _set_profile(sid, profile_data):
    """Set profile for a session"""
    if sid not in SESS:
        SESS[sid] = {"messages": [], "profile": {}}
    SESS[sid]["profile"] = profile_data
    _save_session(sid)

def _smooth(text: str) -> str:
    """Add punctuation if missing"""
    if not text:
        return text
    text = text.strip()
    if not text.endswith((".", "!", "?")):
        text += "."
    return text

# ===== UI HTML =====
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JGM Insights Assistant 🤖 Powered by Google Gemini</title>
  <style>
    * { box-sizing: border-box; }
    body { 
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; 
      margin: 0; 
      background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
      color: #e0e0e0; 
      min-height: 100vh;
    }
    
    header { 
      padding: 16px 24px; 
      background: rgba(30, 30, 50, 0.95);
      border-bottom: 2px solid #3a3a5a;
      backdrop-filter: blur(10px);
      box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .header-title {
      font-size: 24px;
      font-weight: 700;
      color: #7dcfff;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    
    .header-title::before {
      content: "🤖";
      font-size: 28px;
    }
    
    .ai-badge {
      font-size: 12px;
      padding: 4px 12px;
      background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
      color: white;
      border-radius: 12px;
      font-weight: 700;
      letter-spacing: 0.5px;
      animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.8; }
    }
    
    .subtitle {
      font-size: 12px;
      color: #a0a0c0;
      margin-bottom: 12px;
    }
    
    .controls {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    
    main { 
      padding: 20px; 
      max-width: 1100px; 
      margin: auto; 
    }
    
    #log { 
      background: rgba(15, 15, 25, 0.8);
      border: 1px solid #3a3a5a;
      border-radius: 12px;
      padding: 20px; 
      height: 58vh; 
      overflow-y: auto;
      margin-bottom: 20px;
      box-shadow: inset 0 2px 8px rgba(0,0,0,0.5);
    }
    
    #log::-webkit-scrollbar { width: 8px; }
    #log::-webkit-scrollbar-track { background: rgba(30, 30, 50, 0.5); border-radius: 4px; }
    #log::-webkit-scrollbar-thumb { background: #7dcfff; border-radius: 4px; }
    
    input, button, select { 
      padding: 12px 16px; 
      font-size: 14px;
      border-radius: 8px;
      border: 1px solid #3a3a5a;
      background: rgba(30, 30, 50, 0.8);
      color: #e0e0e0;
      transition: all 0.3s ease;
    }
    
    input:focus, select:focus {
      outline: none;
      border-color: #7dcfff;
      box-shadow: 0 0 0 3px rgba(125, 207, 255, 0.1);
    }
    
    button {
      cursor: pointer;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-size: 12px;
    }
    
    button:hover {
      background: rgba(50, 50, 70, 0.9);
      border-color: #7dcfff;
      transform: translateY(-1px);
      box-shadow: 0 4px 8px rgba(125, 207, 255, 0.2);
    }
    
    .btn-primary {
      background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
      color: white;
      border: none;
      font-weight: 700;
    }
    
    .btn-primary:hover {
      background: linear-gradient(135deg, #5a95ff 0%, #46ba64 100%);
      box-shadow: 0 4px 12px rgba(66, 133, 244, 0.4);
    }
    
    .btn-danger {
      background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
      color: white;
      border: none;
    }
    
    .btn-simulator {
      background: linear-gradient(135deg, #bb9af7 0%, #9d7cd8 100%);
      color: white;
      border: none;
      font-weight: 700;
    }
    
    .btn-simulator::before { content: "🔮"; margin-right: 6px; }
    
    #msg { flex: 1; min-width: 300px; }
    
    .sys { 
      color: #7dcfff; 
      background: rgba(125, 207, 255, 0.1);
      padding: 12px 16px;
      border-radius: 8px;
      border-left: 4px solid #7dcfff;
      margin-bottom: 12px;
      font-size: 13px;
    }
    
    .message-container {
      margin: 12px 0;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    
    .me { 
      color: #dcd7ba;
      background: rgba(220, 215, 186, 0.1);
      padding: 12px 16px;
      border-radius: 8px;
      border-left: 4px solid #dcd7ba;
      font-weight: 500;
    }
    
    .bot { 
      color: #a7c080;
      background: rgba(167, 192, 128, 0.1);
      padding: 14px 18px;
      border-radius: 8px;
      border-left: 4px solid #a7c080;
      white-space: pre-wrap;
      line-height: 1.6;
    }
    
    .attachment {
      margin-top: 8px;
      padding: 10px;
      background: rgba(50, 50, 70, 0.5);
      border-radius: 6px;
      border: 1px solid #3a3a5a;
    }
    
    .attachment a {
      color: #7dcfff;
      text-decoration: none;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    
    .attachment a::before { content: "📎"; }
    
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    
    .tip-box {
      background: linear-gradient(135deg, rgba(66, 133, 244, 0.1) 0%, rgba(52, 168, 83, 0.1) 100%);
      border: 1px solid rgba(66, 133, 244, 0.3);
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 16px;
      font-size: 13px;
    }
    
    .tip-box strong { color: #4285f4; display: block; margin-bottom: 6px; }
    
    .loading { display: none; color: #4285f4; font-style: italic; padding: 8px; }
    .loading.active { display: block; }
    
    @media (max-width: 768px) {
      .controls { flex-direction: column; align-items: stretch; }
      #msg { min-width: 100%; }
    }
  </style>
</head>
<body>
<header>
  <div class="header-title">
    JGM Insights Assistant
    <span class="ai-badge">🤖 GOOGLE GEMINI</span>
  </div>
  <div class="subtitle">Powered by Google Gemini AI • Advanced Education Analytics</div>
  <div class="controls">
    <form id="uploadForm" class="row" enctype="multipart/form-data" onsubmit="return uploadFile(event)">
      <input type="file" name="file" required />
      <select name="target">
        <option value="data">📊 Data</option>
        <option value="graphs">📈 Graphs</option>
        <option value="code">💻 Code</option>
      </select>
      <button type="submit">Upload</button>
    </form>
    <button onclick="reindex()">🔄 Reindex</button>
    <button onclick="simulator()" class="btn-simulator">What-If</button>
    <button onclick="endAndDownload()" class="btn-danger">⬇️ Download</button>
  </div>
</header>

<main>
  <div class="tip-box">
    <strong>🚀 Now Powered by Google Gemini!</strong>
    Advanced AI • What-If Simulator • Real-time Analysis • Production Ready
  </div>
  
  <div id="log"></div>
  <div class="loading" id="loading">🤖 Processing with AI...</div>
  
  <div class="row" style="margin-top:10px;">
    <input id="msg" placeholder="Ask anything about Peru 2025 education data..." onkeypress="if(event.key==='Enter') send()"/>
    <button onclick="send()" class="btn-primary">Send</button>
    <button onclick="profile()">👤 Profile</button>
  </div>
</main>

<script>
const log = document.getElementById("log");
const loading = document.getElementById("loading");

// Session management
function getSessionId() {
  return localStorage.getItem("jgm_session_id");
}

function setSessionId(sid) {
  if (sid) {
    localStorage.setItem("jgm_session_id", sid);
  }
}

function getHeaders() {
  const headers = { "Content-Type": "application/json" };
  const sid = getSessionId();
  if (sid) {
    headers["X-Session-ID"] = sid;
  }
  return headers;
}

function append(who, text, refs = []){
  const container = document.createElement("div");
  container.className = "message-container";
  
  const div = document.createElement("div");
  div.className = who;
  
  const prefix = who === "me" ? "You: " : (who === "bot" ? "Assistant: " : "System: ");
  div.textContent = prefix + text;
  
  container.appendChild(div);
  
  if (refs && refs.length > 0) {
    refs.forEach(ref => {
      const attDiv = document.createElement("div");
      attDiv.className = "attachment";
      const link = document.createElement("a");
      link.href = ref;
      link.target = "_blank";
      
      if (ref.includes(".html") || ref.includes("map")) {
        link.textContent = "🗺️ Open Interactive Map";
      } else if (ref.includes(".png") || ref.includes(".jpg") || ref.includes("chart")) {
        link.textContent = "📊 View Chart";
      } else {
        link.textContent = "📎 View File";
      }
      
      attDiv.appendChild(link);
      container.appendChild(attDiv);
    });
  }
  
  log.appendChild(container);
  log.scrollTop = log.scrollHeight;
}

async function greet(){
  try {
    const res = await fetch("/api/greet", { headers: getHeaders() });
    const j = await res.json();
    if (j.session_id) setSessionId(j.session_id);
    append("bot", j.message || "(no message)");
  } catch (e) {
    append("sys", "Error connecting");
  }
}

async function profile(){
  const first_name = prompt("First name?") || "";
  const last_name  = prompt("Last name?") || "";
  const role       = prompt("Role (parent/student/teacher/NGO/donor/investor)?") || "";
  const contact    = prompt("Contact (email/phone)?") || "";
  
  try {
    loading.classList.add("active");
    const res = await fetch("/api/set_profile", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({first_name, last_name, role, contact})
    });
    const j = await res.json();
    if (j.session_id) setSessionId(j.session_id);
    append("bot", j.message || "(ok)");
  } catch (e) {
    append("sys", "Error saving profile");
  } finally {
    loading.classList.remove("active");
  }
}

async function simulator(){
  const inp = document.getElementById("msg");
  inp.value = "simulate menu";
  send();
}

async function send(){
  const inp = document.getElementById("msg");
  const text = inp.value.trim();
  if (!text) return;
  
  append("me", text);
  inp.value = "";
  
  try {
    loading.classList.add("active");
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({message: text})
    });
    const j = await res.json();
    
    // Save session ID from response
    if (j.session_id) setSessionId(j.session_id);
    
    const refs = j.refs || [];
    append("bot", j.reply || "(no reply)", refs);
  } catch (e) {
    console.error("Error:", e);
    append("sys", "Error sending message");
  } finally {
    loading.classList.remove("active");
  }
}

async function reindex(){
  try {
    loading.classList.add("active");
    const res = await fetch("/api/reindex", { method: "POST", headers: getHeaders() });
    const j = await res.json();
    if (j.session_id) setSessionId(j.session_id);
    append("bot", `✅ Reindexed! Found ${j.items} items.`);
  } catch (e) {
    append("sys", "Error reindexing");
  } finally {
    loading.classList.remove("active");
  }
}

async function uploadFile(ev){
  ev.preventDefault();
  const form = document.getElementById("uploadForm");
  const fd = new FormData(form);
  
  // Add session ID to form data
  const sid = getSessionId();
  if (sid) fd.append("session_id", sid);
  
  try {
    loading.classList.add("active");
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const j = await res.json();
    if (j.session_id) setSessionId(j.session_id);
    append("bot", j.message || "Uploaded.");
    setTimeout(reindex, 300);
  } catch (e) {
    append("sys", "Error uploading");
  } finally {
    loading.classList.remove("active");
  }
  
  return false;
}

async function endAndDownload(){
  if (!confirm("Download conversation?")) return;
  
  try {
    loading.classList.add("active");
    const sid = getSessionId();
    const url = "/api/download?format=html" + (sid ? "&session_id=" + sid : "");
    const res = await fetch(url);
    
    if (res.ok) {
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      const timestamp = new Date().toISOString().slice(0,19).replace(/:/g,"-");
      a.download = `JGM_Conversation_${timestamp}.html`;
      a.click();
      append("bot", "✅ Downloaded!");
    } else {
      append("sys", "Could not create transcript");
    }
  } catch (e) {
    append("sys", "Error downloading");
  } finally {
    loading.classList.remove("active");
  }
}

window.onload = greet;
</script>
</body>
</html>
"""

# ===== ROUTES =====

@app.get("/")
def index():
    sid = _get_sid()
    resp = make_response(render_template_string(INDEX_HTML))
    resp.set_cookie("session_id", sid, httponly=True, samesite="Lax")
    return resp

@app.get("/favicon.ico")
def favicon():
    return Response(status=204)

@app.post("/api/reindex")
def reindex():
    sid = _get_sid()
    try:
        if AGENT_SYSTEM_AVAILABLE and BOT:
            df = BOT.build_index()
            items = 0 if df is None else len(df)
        elif bot:
            df = bot.build_index()
            items = 0 if df is None else len(df)
        else:
            items = 0
        
        _record(sid, "system", "Reindexed workspace")
        return jsonify({"status": "ok", "items": items, "session_id": sid})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e), "session_id": sid}), 500

@app.get("/api/greet")
def greet():
    """
    Greet the user - uses SESSION-BASED profile only, not global state!
    """
    sid = _get_sid()
    try:
        # Get profile from THIS SESSION ONLY
        profile = _get_profile(sid)
        name = profile.get("first_name", "")
        
        # Generic greeting - DO NOT use agent's greet_user() which has global state
        if name:
            # Returning user with profile set
            msg = f"""Welcome back, {name}! 👋

How can I help you with Peru's 2025 education data today?

💡 Quick commands:
  • 'summary' - Get conversation overview
  • 'map' - Generate geographic visualization
  • 🔮 'simulate menu' - See What-If scenarios
  • Or just ask any question!"""
        else:
            # New user - no profile yet
            msg = """👋 Hello! I'm the JGM Insights Assistant.

I can help you explore Peru education data for 2025:
  • Undergraduate applicant statistics
  • Primary/secondary school dropout rates
  • Interactive maps and charts
  • 🔮 What-If Simulator - Predict policy impacts!

Please click '👤 Profile' to share your name and role.

💡 Quick commands:
  • 'summary' - Get conversation overview
  • 'map' - Generate geographic visualization
  • 🔮 'simulate menu' - See What-If scenarios
  • Or just ask any question!"""
        
        _record(sid, "bot", msg)
        return jsonify({"message": msg, "session_id": sid})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}", "session_id": sid}), 500

@app.post("/api/set_profile")
def set_profile():
    """
    Set user profile - stores in SESSION ONLY, not globally!
    """
    sid = _get_sid()
    data = request.json or {}
    
    try:
        # Store profile ONLY in this session - not globally!
        profile_data = {
            "first_name": data.get("first_name", ""),
            "last_name": data.get("last_name", ""),
            "role": data.get("role", ""),
            "contact": data.get("contact", "")
        }
        _set_profile(sid, profile_data)
        
        # Build personalized response - DO NOT use agent's set_user_profile()
        name = profile_data["first_name"] or "there"
        role = profile_data["role"] or "user"
        
        # Customize message based on role
        role_tips = {
            "parent": "I can help you understand your child's educational opportunities and regional school performance.",
            "student": "I can help you explore undergraduate programs and admission statistics across Peru.",
            "teacher": "I can provide insights on dropout rates, regional challenges, and educational metrics.",
            "ngo": "I can help analyze regional disparities and identify areas needing intervention.",
            "donor": "I can show you impact metrics and regional needs to help guide your contributions.",
            "investor": "I can provide data on educational infrastructure and growth opportunities.",
        }
        
        role_tip = role_tips.get(role.lower(), "I'm here to help you explore Peru's education data.")
        
        msg = f"""Nice to meet you, {name}! 👋

I've noted that you're a {role}. {role_tip}

How can I help you today? Try:
  • Ask about dropout rates by region
  • Request a map visualization
  • 🔮 Type 'simulate menu' for What-If scenarios"""
        
        msg = _smooth(msg)
        _record(sid, "user", f"(set_profile: {name}, {role})")
        _record(sid, "bot", msg)
        
        print(f"✅ Profile set for session {sid[:8]}...: {name} ({role})")
        
        return jsonify({"message": msg, "session_id": sid})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}", "session_id": sid}), 500

@app.post("/api/chat")
def chat():
    """Main chat endpoint - handles messages from frontend"""
    sid = _get_sid()
    data = request.json or {}
    q = (data.get("message") or "").strip()
    
    if not q:
        return jsonify({"reply": "Please ask a question!", "refs": [], "session_id": sid})
    
    _record(sid, "user", q)
    
    # Get profile for this session to personalize responses
    profile = _get_profile(sid)
    user_name = profile.get("first_name", "")
    user_role = profile.get("role", "")
    
    try:
        if AGENT_SYSTEM_AVAILABLE:
            res = enhanced_chat(q)
        elif bot:
            res = bot.chat(q)
        else:
            # Basic fallback response
            res = {"reply": "I'm sorry, the AI system is not fully initialized. Please try again later."}
        
        reply = _smooth(res.get("reply", ""))
        
        # Collect all references (maps, charts, images)
        refs = []
        
        # Handle map_path
        if res.get("map_path"):
            map_file = Path(res["map_path"]).name
            refs.append(f"/files/{map_file}")
            print(f"✅ Map created: {res['map_path']}")
        
        # Handle image_path
        if res.get("image_path"):
            img_file = Path(res["image_path"]).name
            refs.append(f"/files/{img_file}")
        
        # Handle images array
        if res.get("images"):
            for img in res["images"]:
                if img not in refs:
                    refs.append(img)
        
        _record(sid, "bot", reply, attachments=refs)
        
        print(f"📊 Session {sid[:8]}... ({user_name or 'anonymous'}) | Q: {q[:30]}... | Refs: {refs}")
        
        return jsonify({
            "reply": reply,
            "refs": refs,
            "session_id": sid
        })
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        _record(sid, "bot", error_msg)
        return jsonify({"reply": error_msg, "refs": [], "session_id": sid}), 500

@app.get("/files/<path:filename>")
def files(filename):
    """Serve files from workspace"""
    print(f"📁 Serving file: {filename}")
    try:
        return send_from_directory(WORKSPACE, filename, as_attachment=False)
    except Exception as e:
        print(f"❌ File serve error: {e}")
        return Response(f"File not found: {filename}", status=404)

@app.get("/api/download")
def download_transcript():
    fmt = (request.args.get("format") or "html").lower()
    
    # Get session ID from query param, header, or cookie
    sid = request.args.get("session_id") or request.headers.get("X-Session-ID") or request.cookies.get("session_id")
    
    if not sid or sid not in SESS:
        return jsonify({"error": "No conversation found", "session_id": sid}), 400
    
    convo = SESS.get(sid, {}).get("messages", [])
    profile = _get_profile(sid)
    
    if not convo:
        return jsonify({"error": "No conversation", "session_id": sid}), 400

    if fmt == "json":
        content = json.dumps(SESS[sid], ensure_ascii=False, indent=2).encode("utf-8")
        resp = make_response(content)
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        resp.headers["Content-Disposition"] = "attachment; filename=JGM_Conversation.json"
        return resp

    # Get user name for the transcript
    user_name = profile.get("first_name", "User")
    
    html = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>JGM Conversation - {user_name}</title>",
        "<style>",
        "body { font-family: 'Segoe UI', sans-serif; margin: 0; padding: 24px; background: #f5f5f5; }",
        ".container { max-width: 900px; margin: auto; background: white; padding: 32px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }",
        "h1 { color: #1a1a2e; }",
        ".badge { background: linear-gradient(135deg, #4285f4 0%, #34a853 100%); color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 700; }",
        ".profile { background: #e3f2fd; padding: 12px; border-radius: 8px; margin-bottom: 20px; }",
        ".message { margin: 20px 0; padding: 16px; border-radius: 8px; }",
        ".user { background: #e3f2fd; border-left: 4px solid #2196f3; }",
        ".bot { background: #e8f5e9; border-left: 4px solid #4caf50; }",
        ".label { font-weight: 700; color: #333; margin-bottom: 8px; }",
        ".content { white-space: pre-wrap; line-height: 1.6; color: #333; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='container'>",
        "<h1>🤖 JGM Insights Assistant <span class='badge'>GOOGLE GEMINI</span></h1>",
    ]
    
    # Add profile info if available
    if profile.get("first_name"):
        html.append("<div class='profile'>")
        html.append(f"<strong>User:</strong> {profile.get('first_name', '')} {profile.get('last_name', '')}<br>")
        if profile.get('role'):
            html.append(f"<strong>Role:</strong> {profile.get('role', '')}")
        html.append("</div>")

    for m in convo:
        role = m.get("role", "user")
        who = user_name if role == "user" else "Assistant"
        css_class = role
        
        safe_text = (m.get("text", "")
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))

        html.append(f"<div class='message {css_class}'>")
        html.append(f"<div class='label'>{who}</div>")
        html.append(f"<div class='content'>{safe_text}</div>")
        html.append("</div>")

    html.append("</div></body></html>")

    content = "\n".join(html).encode("utf-8")
    resp = make_response(content)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    resp.headers["Content-Disposition"] = f"attachment; filename=JGM_Conversation_{user_name}.html"
    return resp

@app.post("/api/upload")
def upload():
    sid = _get_sid()
    f = request.files.get("file")
    target = (request.form.get("target") or "data").strip().lower()
    
    if not f:
        return jsonify({"status": "error", "message": "No file", "session_id": sid}), 400
    
    if target not in ("data", "graphs", "code"):
        target = "data"
    
    dest_dir = {"data": DATA_DIR, "graphs": GRAPHS_DIR, "code": CODE_DIR}[target]
    save_path = dest_dir / Path(f.filename).name
    
    try:
        f.save(save_path)
        
        if AGENT_SYSTEM_AVAILABLE and BOT:
            BOT.build_index()
        elif bot:
            bot.build_index()
        
        _record(sid, "system", f"Uploaded {save_path.name}")
        return jsonify({"status": "ok", "message": f"✅ Uploaded: {save_path.name}", "session_id": sid})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e), "session_id": sid}), 500

# ===== HEALTH CHECK =====
@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        status = get_agent_status() if AGENT_SYSTEM_AVAILABLE else {
            "google_adk_available": False,
            "agent_initialized": False,
            "ollama_available": False,
            "chatbot_ready": bot is not None,
            "primary_engine": "fallback" if bot else "none"
        }
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "workspace": str(WORKSPACE),
            "production": PRODUCTION_MODE,
            "agent_system": AGENT_SYSTEM_AVAILABLE,
            "active_sessions": len(SESS),
            **status
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.get("/api/status")
def api_status():
    """Detailed status"""
    try:
        status = get_agent_status() if AGENT_SYSTEM_AVAILABLE else {}
        
        return jsonify({
            "agent_system_available": AGENT_SYSTEM_AVAILABLE,
            "google_api_key_set": bool(os.getenv("GOOGLE_API_KEY")),
            "workspace_exists": WORKSPACE.exists(),
            "data_files": len(list(DATA_DIR.glob("*"))) if DATA_DIR.exists() else 0,
            "active_sessions": len(SESS),
            **status
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== STARTUP =====
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 JGM INSIGHTS ASSISTANT - PER-SESSION PROFILES")
    print("=" * 70)
    
    if AGENT_SYSTEM_AVAILABLE:
        status = get_agent_status()
        print(f"\n✅ Agent System: ACTIVE")
        print(f"   Primary Engine: {status.get('primary_engine', 'unknown').upper()}")
    elif bot:
        print(f"\n⚠️  Agent System: FALLBACK MODE (Direct Chatbot)")
    else:
        print(f"\n❌ No AI system available")
    
    print(f"\n🌐 CORS Enabled for:")
    print(f"   - https://jgm-ds-website.vercel.app")
    print(f"   - http://localhost:3000")
    print(f"   - http://localhost:5173")
    
    print(f"\n🔑 Session Management:")
    print(f"   - X-Session-ID header support")
    print(f"   - Per-session profile storage (NO GLOBAL STATE)")
    print(f"   - Session ID returned in all responses")
    
    print("=" * 70)
    print(f"📍 URL: http://localhost:{PORT}")
    print(f"🔮 Features: What-If, Maps, Charts, Conversations")
    print("=" * 70)
    
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True, use_reloader=False)
