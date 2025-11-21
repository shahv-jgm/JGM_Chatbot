"""
JGM Insights Assistant - Flask App (MAP FIX)
Fixed map link display issue
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import agent system
try:
    from agent import (
        enhanced_chat, greet_user, set_user_profile, 
        get_agent_status, initialize_agent,
        BOT, GOOGLE_ADK_AVAILABLE
    )
    AGENT_SYSTEM_AVAILABLE = True
    print("✅ Agent system loaded successfully")
except Exception as e:
    AGENT_SYSTEM_AVAILABLE = False
    print(f"❌ Agent system failed: {e}")
    print("   Using direct chatbot fallback")
    from jgm_rag_chatbot import JGMRAG

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

# ===== INITIALIZE SYSTEM =====
if AGENT_SYSTEM_AVAILABLE:
    agent_ready = initialize_agent()
    if agent_ready:
        print("✅ Google Gemini agent active")
    else:
        print("⚠️  Using fallback chatbot")
else:
    bot = JGMRAG(WORKSPACE)
    bot.build_index()
    print("✅ Direct chatbot initialized")

# ===== SESSION STORAGE =====
SESS = {}

def _get_sid():
    """Get or create session ID"""
    sid = request.cookies.get("session_id")
    if not sid:
        sid = str(uuid.uuid4())
    if sid not in SESS:
        SESS[sid] = []
        tfile = TRANS_DIR / f"{sid}.json"
        if tfile.exists():
            try:
                SESS[sid] = json.loads(tfile.read_text(encoding="utf-8"))
            except Exception:
                SESS[sid] = []
    return sid

def _record(sid, role, text, attachments=None):
    """Record conversation to session"""
    SESS[sid].append({
        "role": role,
        "text": text,
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "attachments": attachments or []
    })
    try:
        (TRANS_DIR / f"{sid}.json").write_text(
            json.dumps(SESS[sid], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        print(f"Error saving transcript: {e}")

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

function append(who, text, attachments = []){
  const container = document.createElement("div");
  container.className = "message-container";
  
  const div = document.createElement("div");
  div.className = who;
  
  const prefix = who === "me" ? "You: " : (who === "bot" ? "Assistant: " : "System: ");
  div.textContent = prefix + text;
  
  container.appendChild(div);
  
  // FIXED: Better attachment handling
  if (attachments && attachments.length > 0) {
    attachments.forEach(att => {
      const attDiv = document.createElement("div");
      attDiv.className = "attachment";
      const link = document.createElement("a");
      link.href = att;
      link.target = "_blank";
      
      // Better link text based on file type
      if (att.includes(".html") || att.includes("map")) {
        link.textContent = "🗺️ Open Interactive Map";
      } else if (att.includes(".png") || att.includes(".jpg") || att.includes("chart")) {
        link.textContent = "📊 View Chart";
      } else {
        link.textContent = "📎 View File";
      }
      
      attDiv.appendChild(link);
      container.appendChild(attDiv);
      
      console.log("Added attachment:", att);  // Debug
    });
  }
  
  log.appendChild(container);
  log.scrollTop = log.scrollHeight;
}

async function greet(){
  try {
    const res = await fetch("/api/greet");
    const j = await res.json();
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
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({first_name, last_name, role, contact})
    });
    const j = await res.json();
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
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: text})
    });
    const j = await res.json();
    
    console.log("Response:", j);  // Debug
    
    // FIXED: Collect ALL attachments properly
    const attachments = [];
    
    // Check for map_path (primary)
    if (j.map_path) {
      const mapFile = j.map_path.split('/').pop();
      attachments.push(`/files/${mapFile}`);
      console.log("Found map_path:", j.map_path);
    }
    
    // Check for map (alternative)
    if (j.map && !j.map_path) {
      attachments.push(j.map);
      console.log("Found map:", j.map);
    }
    
    // Check for image_path
    if (j.image_path) {
      const imgFile = j.image_path.split('/').pop();
      attachments.push(`/files/${imgFile}`);
      console.log("Found image_path:", j.image_path);
    }
    
    // Check for images array
    if (j.images && j.images.length > 0) {
      attachments.push(...j.images);
      console.log("Found images:", j.images);
    }
    
    console.log("Final attachments:", attachments);  // Debug
    
    append("bot", j.reply || "(no reply)", attachments);
  } catch (e) {
    console.error("Error:", e);  // Debug
    append("sys", "Error sending message");
  } finally {
    loading.classList.remove("active");
  }
}

async function reindex(){
  try {
    loading.classList.add("active");
    const res = await fetch("/api/reindex", {method: "POST"});
    const j = await res.json();
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
  
  try {
    loading.classList.add("active");
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const j = await res.json();
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
    const res = await fetch("/api/download?format=html");
    
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
        else:
            items = 0
        
        _record(sid, "system", "Reindexed workspace")
        return jsonify({"status":"ok", "items": items})
    except Exception as e:
        return jsonify({"status":"error", "message": str(e)}), 500

@app.get("/api/greet")
def greet():
    sid = _get_sid()
    try:
        if AGENT_SYSTEM_AVAILABLE:
            msg = greet_user()
        else:
            msg = bot.greet_and_collect()
        
        _record(sid, "bot", msg)
        return jsonify({"message": msg})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.post("/api/set_profile")
def set_profile():
    sid = _get_sid()
    data = request.json or {}
    
    try:
        if AGENT_SYSTEM_AVAILABLE:
            msg = set_user_profile(
                first_name=data.get("first_name", ""),
                last_name=data.get("last_name", ""),
                role=data.get("role", ""),
                contact=data.get("contact", "")
            )
        else:
            msg = bot.set_profile(
                first_name=data.get("first_name"),
                last_name=data.get("last_name"),
                role=data.get("role"),
                contact=data.get("contact"),
            )
        
        msg = _smooth(msg)
        _record(sid, "user", "(set_profile)")
        _record(sid, "bot", msg)
        
        return jsonify({"message": msg})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.post("/api/chat")
def chat():
    sid = _get_sid()
    data = request.json or {}
    q = (data.get("message") or "").strip()
    
    if not q:
        return jsonify({"reply": "Please ask a question!"})
    
    _record(sid, "user", q)
    
    try:
        if AGENT_SYSTEM_AVAILABLE:
            res = enhanced_chat(q)
        else:
            res = bot.chat(q)
        
        reply = _smooth(res.get("reply", ""))
        
        # FIXED: Better attachment handling
        attachments = []
        
        # Handle map_path
        if res.get("map_path"):
            map_file = Path(res["map_path"]).name
            attachments.append(f"/files/{map_file}")
            print(f"✅ Map created: {res['map_path']}")  # Debug
        
        # Handle image_path
        if res.get("image_path"):
            img_file = Path(res["image_path"]).name
            attachments.append(f"/files/{img_file}")
            # Also add to images array for consistency
            if "images" not in res:
                res["images"] = []
            res["images"].append(f"/files/{img_file}")
        
        # Handle existing images array
        if res.get("images"):
            for img in res["images"]:
                if img not in attachments:
                    attachments.append(img)
        
        _record(sid, "bot", reply, attachments=attachments)
        res["reply"] = reply
        
        print(f"📊 Response keys: {res.keys()}")  # Debug
        print(f"📎 Attachments: {attachments}")  # Debug
        
        return jsonify(res)
        
    except Exception as e:
        print(f"❌ Chat error: {e}")  # Debug
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        _record(sid, "bot", error_msg)
        return jsonify({"reply": error_msg}), 500

@app.get("/files/<path:filename>")
def files(filename):
    """Serve files from workspace"""
    print(f"📁 Serving file: {filename}")  # Debug
    try:
        return send_from_directory(WORKSPACE, filename, as_attachment=False)
    except Exception as e:
        print(f"❌ File serve error: {e}")  # Debug
        return Response(f"File not found: {filename}", status=404)

@app.get("/api/download")
def download_transcript():
    fmt = (request.args.get("format") or "html").lower()
    sid = _get_sid()
    convo = SESS.get(sid, [])
    
    if not convo:
        return jsonify({"error":"no conversation"}), 400

    if fmt == "json":
        content = (TRANS_DIR / f"{sid}.json").read_bytes()
        resp = make_response(content)
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        resp.headers["Content-Disposition"] = "attachment; filename=JGM_Conversation.json"
        return resp

    html = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>JGM Conversation - Google Gemini Powered</title>",
        "<style>",
        "body { font-family: 'Segoe UI', sans-serif; margin: 0; padding: 24px; background: #f5f5f5; }",
        ".container { max-width: 900px; margin: auto; background: white; padding: 32px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }",
        "h1 { color: #1a1a2e; }",
        ".badge { background: linear-gradient(135deg, #4285f4 0%, #34a853 100%); color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 700; }",
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

    for m in convo:
        role = m.get("role", "user")
        who = "You" if role == "user" else "Assistant"
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
    resp.headers["Content-Disposition"] = "attachment; filename=JGM_Conversation.html"
    return resp

@app.post("/api/upload")
def upload():
    sid = _get_sid()
    f = request.files.get("file")
    target = (request.form.get("target") or "data").strip().lower()
    
    if not f:
        return jsonify({"status":"error","message":"No file"}), 400
    
    if target not in ("data","graphs","code"):
        target = "data"
    
    dest_dir = {"data": DATA_DIR, "graphs": GRAPHS_DIR, "code": CODE_DIR}[target]
    save_path = dest_dir / Path(f.filename).name
    
    try:
        f.save(save_path)
        
        if AGENT_SYSTEM_AVAILABLE and BOT:
            BOT.build_index()
        
        _record(sid, "system", f"Uploaded {save_path.name}")
        return jsonify({"status":"ok","message":f"✅ Uploaded: {save_path.name}"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

# ===== HEALTH CHECK =====
@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        status = get_agent_status() if AGENT_SYSTEM_AVAILABLE else {
            "google_adk_available": False,
            "agent_initialized": False,
            "ollama_available": False,
            "chatbot_ready": False,
            "primary_engine": "none"
        }
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "workspace": str(WORKSPACE),
            "production": PRODUCTION_MODE,
            "agent_system": AGENT_SYSTEM_AVAILABLE,
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
    print("🚀 JGM INSIGHTS ASSISTANT - MAP FIX VERSION")
    print("=" * 70)
    
    if AGENT_SYSTEM_AVAILABLE:
        status = get_agent_status()
        print(f"\n✅ Agent System: ACTIVE")
        print(f"   Primary Engine: {status.get('primary_engine', 'unknown').upper()}")
    else:
        print(f"\n⚠️  Agent System: FALLBACK MODE")
    
    print("=" * 70)
    print(f"📍 URL: http://localhost:{PORT}")
    print(f"🔮 Features: What-If, Maps, Charts, Conversations")
    print(f"🐛 Debug: Console logging enabled for map issues")
    print("=" * 70)
    
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True, use_reloader=False)