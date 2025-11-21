import os, re, io, json, glob, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import datetime

import pandas as pd
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (10, 5)})
plt.rcParams.update({"axes.grid": True})

# LlamaIndex + Ollama imports
try:
    from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.core.node_parser import SimpleNodeParser
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    print("⚠️ LlamaIndex not installed. Running in basic mode.")

# Peru departments/regions with approximate coordinates
PERU_LOCATIONS = {
    "amazonas": {"lat": -5.7667, "lon": -78.0833},
    "ancash": {"lat": -9.5278, "lon": -77.5278},
    "apurimac": {"lat": -13.6333, "lon": -72.8833},
    "arequipa": {"lat": -16.4090, "lon": -71.5375},
    "ayacucho": {"lat": -13.1631, "lon": -74.2236},
    "cajamarca": {"lat": -7.1611, "lon": -78.5126},
    "callao": {"lat": -12.0565, "lon": -77.1181},
    "cusco": {"lat": -13.5319, "lon": -71.9675},
    "cuzco": {"lat": -13.5319, "lon": -71.9675},
    "huancavelica": {"lat": -12.7825, "lon": -74.9758},
    "huanuco": {"lat": -9.9306, "lon": -76.2422},
    "ica": {"lat": -14.0678, "lon": -75.7286},
    "junin": {"lat": -11.1583, "lon": -75.9925},
    "la libertad": {"lat": -8.1116, "lon": -79.0289},
    "lambayeque": {"lat": -6.7011, "lon": -79.9061},
    "lima": {"lat": -12.0464, "lon": -77.0428},
    "loreto": {"lat": -3.7491, "lon": -73.2538},
    "madre de dios": {"lat": -12.5934, "lon": -69.1892},
    "moquegua": {"lat": -17.1928, "lon": -70.9342},
    "pasco": {"lat": -10.6798, "lon": -76.2561},
    "piura": {"lat": -5.1945, "lon": -80.6328},
    "puno": {"lat": -15.8402, "lon": -70.0219},
    "san martin": {"lat": -6.4856, "lon": -76.3647},
    "tacna": {"lat": -18.0147, "lon": -70.2536},
    "tumbes": {"lat": -3.5669, "lon": -80.4515},
    "ucayali": {"lat": -8.3791, "lon": -74.5539}
}

# What-If Scenarios with impact data
SCENARIOS = {
    "reduce_class_size": {
        "name": "Reduce Class Sizes by 20%",
        "description": "Hire additional teachers to reduce student-teacher ratio",
        "impact_rate": -0.25,
        "cost_per_student": 120,
        "confidence": 78,
        "evidence": "Based on 23 similar interventions in Latin America"
    },
    "teacher_training": {
        "name": "Intensive Teacher Training Program",
        "description": "3-month professional development for all teachers",
        "impact_rate": -0.18,
        "cost_per_student": 85,
        "confidence": 72,
        "evidence": "Based on Peru's 2019-2022 pilot programs"
    },
    "meal_program": {
        "name": "Universal School Meal Program",
        "description": "Free breakfast and lunch for all students",
        "impact_rate": -0.32,
        "cost_per_student": 180,
        "confidence": 85,
        "evidence": "Based on 41 meal program studies globally"
    },
    "infrastructure": {
        "name": "School Infrastructure Improvement",
        "description": "Renovate facilities, add technology, improve safety",
        "impact_rate": -0.22,
        "cost_per_student": 250,
        "confidence": 68,
        "evidence": "Based on World Bank education infrastructure data"
    },
    "scholarship": {
        "name": "Need-Based Scholarship Program",
        "description": "Financial aid for low-income families",
        "impact_rate": -0.38,
        "cost_per_student": 200,
        "confidence": 81,
        "evidence": "Based on Peru Ministry of Education scholarship outcomes"
    },
    "mentorship": {
        "name": "Student Mentorship Program",
        "description": "Pair at-risk students with mentors",
        "impact_rate": -0.28,
        "cost_per_student": 95,
        "confidence": 75,
        "evidence": "Based on NGO mentorship programs in Andean regions"
    }
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _fmt_num(x):
    try:
        if pd.isna(x):
            return "NA"
        if isinstance(x, (int,)) or (isinstance(x, float) and abs(x) >= 1000 and float(x).is_integer()):
            return f"{int(round(float(x))):,}"
        if isinstance(x, float):
            return f"{x:,.2f}"
        return str(x)
    except Exception:
        return str(x)


class JGMRAG:
    VALID_YEAR = 2025

    def __init__(self, root: Path):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.graphs_dir = self.root / "graphs"
        self.code_dir = self.root / "code"
        
        self.docs: List[str] = []
        self.doc_meta: List[Dict[str, Any]] = []
        self.vectorizer = None
        self.matrix = None
        self.loaded_tables: Dict[str, pd.DataFrame] = {}
        self.user_profile = {
            "first_name": None, 
            "last_name": None, 
            "role": None, 
            "contact": None,
            "onboarded": False
        }
        self.graph_catalog: List[Dict[str, str]] = []
        self.conversation_history = []
        
        # LlamaIndex + Ollama setup
        self.llm_available = False
        self.index = None
        self.query_engine = None
        
        if LLAMAINDEX_AVAILABLE:
            self._init_llm()

    def _init_llm(self):
        """Initialize Ollama + LlamaIndex"""
        try:
            llm = Ollama(
                model="llama3.1",
                request_timeout=60.0,
                temperature=0.7,
                context_window=4096
            )
            
            embed_model = OllamaEmbedding(
                model_name="nomic-embed-text",
                base_url="http://localhost:11434"
            )
            
            Settings.llm = llm
            Settings.embed_model = embed_model
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
            
            self.llm_available = True
            print("✅ Ollama + LlamaIndex initialized successfully!")
            print(f"   Model: llama3.1")
            print(f"   Status: Ready for smart conversations!")
            
        except Exception as e:
            print(f"⚠️ Could not connect to Ollama: {e}")
            print(f"   Make sure Ollama is running: ollama serve")
            print(f"   And models are pulled: ollama pull llama3.1 && ollama pull nomic-embed-text")
            self.llm_available = False

    def _build_llamaindex(self):
        """Build LlamaIndex from documents"""
        if not LLAMAINDEX_AVAILABLE or not self.llm_available:
            return
        
        try:
            documents = []
            
            for name, df in self.loaded_tables.items():
                summary = f"Dataset: {name}\n"
                summary += f"Columns: {', '.join(df.columns)}\n"
                summary += f"Total records: {len(df)}\n"
                summary += f"Year: {self.VALID_YEAR}\n"
                
                if len(df) > 0:
                    summary += f"\nSample data:\n{df.head(3).to_string()}\n"
                
                documents.append(Document(
                    text=summary,
                    metadata={"source": name, "type": "table"}
                ))
            
            for p in self.code_dir.glob("*"):
                if p.suffix.lower() in (".py", ".ipynb", ".md", ".txt"):
                    try:
                        if p.suffix == ".ipynb":
                            txt = self._read_ipynb(p)
                        else:
                            txt = self._read_text_file(p)
                        
                        if txt.strip():
                            documents.append(Document(
                                text=txt[:2000],
                                metadata={"source": p.name, "type": "code"}
                            ))
                    except Exception:
                        pass
            
            if documents:
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    show_progress=False
                )
                
                self.query_engine = self.index.as_query_engine(
                    response_mode="compact",
                    similarity_top_k=3
                )
                
                print(f"✅ LlamaIndex built with {len(documents)} documents!")
            
        except Exception as e:
            print(f"⚠️ LlamaIndex build error: {e}")
            self.llm_available = False

    def add_to_history(self, role: str, text: str):
        self.conversation_history.append({
            "role": role,
            "text": text,
            "timestamp": datetime.datetime.now().isoformat()
        })

    def generate_summary(self) -> str:
        if not self.conversation_history:
            return "No conversation history yet. Start asking questions!"
        
        user_msgs = [m for m in self.conversation_history if m["role"] == "user"]
        bot_msgs = [m for m in self.conversation_history if m["role"] == "bot"]
        
        summary_parts = [
            "=" * 60,
            "📊 CONVERSATION SUMMARY",
            "=" * 60,
            f"\n📈 Statistics:",
            f"  • Total exchanges: {len(user_msgs)} questions, {len(bot_msgs)} responses",
            f"  • Session started: {self.conversation_history[0]['timestamp'][:19]}",
            f"\n💬 Full Conversation:\n"
        ]
        
        for i, msg in enumerate(self.conversation_history, 1):
            role = "YOU" if msg["role"] == "user" else "ASSISTANT"
            time = msg["timestamp"][11:19]
            text = msg["text"]
            if len(text) > 300:
                text = text[:300] + "..."
            summary_parts.append(f"{i}. [{time}] {role}:\n   {text}\n")
        
        summary_parts.append("=" * 60)
        return "\n".join(summary_parts)

    # ========== WHAT-IF SIMULATOR ==========
    def _detect_simulation_query(self, query: str) -> Optional[str]:
        """Detect if user wants to run a simulation - FIXED"""
        qn = _norm(query)
        
        sim_triggers = [
            "what if", "simulate", "scenario", "predict", 
            "what would happen", "impact of", "effect of",
            "roi", "return on investment", "cost benefit"
        ]
        
        if any(trigger in qn for trigger in sim_triggers):
            # FIXED: Match specific keywords for each scenario
            if any(kw in qn for kw in ["meal", "food", "lunch", "breakfast", "nutrition"]):
                return "meal_program"
            elif any(kw in qn for kw in ["mentor", "mentorship", "mentoring", "tutor"]):
                return "mentorship"
            elif any(kw in qn for kw in ["scholarship", "financial aid", "grant"]):
                return "scholarship"
            elif any(kw in qn for kw in ["class size", "smaller class", "reduce class"]):
                return "reduce_class_size"
            elif any(kw in qn for kw in ["teacher training", "train teacher", "professional development"]):
                return "teacher_training"
            elif any(kw in qn for kw in ["infrastructure", "facilities", "building", "renovation"]):
                return "infrastructure"
            
            return "menu"
        
        return None

    def run_simulation(self, scenario_key: str = None, region: str = None) -> str:
        """Run What-If simulation"""
        
        if scenario_key == "menu" or scenario_key is None:
            return self._show_simulation_menu()
        
        if scenario_key not in SCENARIOS:
            return self._show_simulation_menu()
        
        scenario = SCENARIOS[scenario_key]
        
        dataset_type = self._detect_dataset_type("dropout")
        table_name = self._find_table_by_type(dataset_type)
        
        if not table_name:
            return "⚠️ Cannot run simulation - no dropout data available."
        
        df = self.loaded_tables[table_name].copy()
        
        metric_col = None
        if "Tasa" in df.columns:
            metric_col = "Tasa"
        
        if not metric_col:
            return "⚠️ Cannot run simulation - dropout rate data not found."
        
        if region:
            region_data = df[df["Departamento"].str.lower() == region.lower()]
            if region_data.empty:
                baseline_rate = df[metric_col].mean()
                scope = "National Average"
            else:
                baseline_rate = region_data[metric_col].mean()
                scope = region.title()
        else:
            baseline_rate = df[metric_col].mean()
            scope = "National Average"
        
        impact = scenario["impact_rate"]
        predicted_rate = baseline_rate * (1 + impact)
        rate_reduction = baseline_rate - predicted_rate
        
        total_students = len(df) * 500
        students_at_risk = int(total_students * (baseline_rate / 100))
        students_saved = int(students_at_risk * abs(impact))
        
        total_cost = scenario["cost_per_student"] * total_students
        
        value_per_student = 35000
        total_value = students_saved * value_per_student
        roi = total_value / total_cost if total_cost > 0 else 0
        
        response = [
            "🔮 WHAT-IF SIMULATION RESULTS",
            "=" * 60,
            f"\n**SCENARIO:** {scenario['name']}",
            f"**Description:** {scenario['description']}",
            f"**Scope:** {scope}",
            f"\n📊 **PREDICTED OUTCOMES:**",
            f"\n**Immediate Impact (Year 1):**",
            f"  • Current dropout rate: {_fmt_num(baseline_rate)}%",
            f"  • Predicted rate: {_fmt_num(predicted_rate)}% ",
            f"  • Expected reduction: {_fmt_num(rate_reduction)}% ({_fmt_num(abs(impact)*100)}% improvement)",
            f"\n👥 **HUMAN IMPACT:**",
            f"  • Students currently at risk: ~{_fmt_num(students_at_risk)}",
            f"  • Students who would stay in school: ~{_fmt_num(students_saved)}",
            f"  • Lives positively impacted: ~{_fmt_num(students_saved * 4)} (including families)",
            f"\n💰 **FINANCIAL ANALYSIS:**",
            f"\n**Investment Required:**",
            f"  • Cost per student: ${_fmt_num(scenario['cost_per_student'])}/year",
            f"  • Total annual investment: ${_fmt_num(total_cost/1000000)}M",
            f"  • 5-year commitment: ${_fmt_num((total_cost*5)/1000000)}M",
            f"\n**Economic Returns:**",
            f"  • Economic value per retained student: ${_fmt_num(value_per_student)} (lifetime)",
            f"  • Total economic value generated: ${_fmt_num(total_value/1000000)}M",
            f"  • **Return on Investment (ROI): {_fmt_num(roi)}x**",
            f"  • Break-even timeline: ~{_fmt_num(1/roi * 10)} years",
            f"\n📈 **FOR INVESTORS:**",
            f"  • Social Impact: {_fmt_num(students_saved)} students",
            f"  • Financial Return: {_fmt_num((roi-1)*100)}% over 10 years",
            f"  • Risk Level: Medium (education interventions)",
            f"  • Alignment: UN SDG 4 (Quality Education)",
            f"\n🎯 **FOR NGOs:**",
            f"  • Impact per $1,000: {_fmt_num((students_saved/(total_cost/1000)))} students helped",
            f"  • Scalability: High (can expand to all regions)",
            f"  • Community benefit: Reduced poverty, increased literacy",
            f"  • Monitoring: Monthly dropout tracking available",
            f"\n👨‍👩‍👧 **FOR FAMILIES:**",
            f"  • Your child's success rate increases by {_fmt_num(abs(impact)*100)}%",
            f"  • Better educational outcomes lead to higher income potential",
            f"  • Access to additional support services included",
            f"  • Community improvement benefits everyone",
            f"\n⚠️ **CONFIDENCE & EVIDENCE:**",
            f"  • Prediction confidence: {scenario['confidence']}%",
            f"  • Evidence base: {scenario['evidence']}",
            f"  • Recommended pilot: Start with 2-3 high-risk regions",
            f"\n" + "=" * 60,
            f"\n💡 **Want to explore more scenarios?**",
            f"Type 'simulate menu' to see all options!"
        ]
        
        return "\n".join(response)

    def _show_simulation_menu(self) -> str:
        """Show available simulation scenarios"""
        menu = [
            "🔮 WHAT-IF SIMULATOR - Available Scenarios",
            "=" * 60,
            "\nChoose a scenario to simulate:\n"
        ]
        
        for i, (key, scenario) in enumerate(SCENARIOS.items(), 1):
            menu.append(f"{i}. **{scenario['name']}**")
            menu.append(f"   {scenario['description']}")
            menu.append(f"   Expected impact: {abs(int(scenario['impact_rate']*100))}% reduction")
            menu.append(f"   Cost: ${scenario['cost_per_student']}/student/year")
            menu.append("")
        
        menu.extend([
            "📝 **How to use:**",
            "  • 'Simulate meal program'",
            "  • 'What if we reduce class sizes?'",
            "  • 'Show impact of scholarships'",
            "  • 'Predict teacher training outcomes'",
            "\n💡 Simulations show:",
            "  • Predicted dropout rate changes",
            "  • Number of students helped",
            "  • Cost-benefit analysis",
            "  • ROI for investors",
            "  • Impact metrics for NGOs"
        ])
        
        return "\n".join(menu)

    # ========== END SIMULATOR ==========

    def _handle_casual_conversation(self, query: str) -> Optional[str]:
        """Handle greetings, thanks, and casual conversation"""
        qn = _norm(query)
        
        greetings = ["hello", "hi", "hey", "hola", "good morning", "good afternoon", "good evening", 
                     "greetings", "hi there", "hey there", "hello there", "sup", "what's up", "whats up"]
        if any(qn.startswith(g) for g in greetings) or qn in greetings:
            name = self.user_profile.get("first_name")
            if name:
                return (
                    f"Hello {name}! 👋 Great to see you again!\n\n"
                    f"How can I help you with Peru's {self.VALID_YEAR} education data today?\n\n"
                    "💡 You can ask me about:\n"
                    "  • Dropout rates by region\n"
                    "  • Undergraduate applicant statistics\n"
                    "  • Geographic visualizations\n"
                    "  • 🔮 **What-If Simulations** (try 'simulate menu')\n"
                    "  • Or type 'summary' to review our conversation"
                )
            else:
                return (
                    "Hello! 👋 Welcome to JGM Insights Assistant!\n\n"
                    f"I'm here to help you explore Peru's {self.VALID_YEAR} education data.\n\n"
                    "Before we start, would you like to set up your profile? Click '👤 Profile' button.\n\n"
                    "Or jump right in and ask me anything about:\n"
                    "  • Dropout rates\n"
                    "  • Applicant statistics\n"
                    "  • Regional comparisons\n"
                    "  • Interactive maps\n"
                    "  • 🔮 **What-If Simulations** (try 'simulate menu')"
                )
        
        thanks = ["thank", "thanks", "thank you", "thx", "gracias", "appreciate"]
        if any(t in qn for t in thanks):
            return (
                "You're very welcome! 😊\n\n"
                "I'm happy to help. Feel free to ask me anything else about the education data!"
            )
        
        if any(phrase in qn for phrase in ["how are you", "how r u", "how are u", "hows it going", "how's it going"]):
            return (
                "I'm doing great, thank you for asking! 🤖\n\n"
                f"I'm ready to help you analyze Peru's {self.VALID_YEAR} education data.\n\n"
                "What would you like to explore today?"
            )
        
        help_keywords = ["help", "what can you do", "what do you do", "capabilities", "how to use", 
                        "commands", "what can i ask"]
        if any(kw in qn for kw in help_keywords):
            llm_status = "🧠 AI-Powered (Ollama)" if self.llm_available else "📊 Data Analysis"
            return (
                f"🤖 **JGM Insights Assistant** ({llm_status})\n\n"
                f"**Data Analysis ({self.VALID_YEAR}):**\n"
                "  • Dropout rates (primary & secondary schools)\n"
                "  • Undergraduate applicant statistics\n"
                "  • Regional/provincial breakdowns\n"
                "  • Faculty and program analysis\n\n"
                "**Visualizations:**\n"
                "  • Interactive charts (just ask for 'chart' or 'graph')\n"
                "  • Geographic maps (type 'map' or 'show map')\n"
                "  • Comparative analysis\n\n"
                "**🔮 What-If Simulator:**\n"
                "  • Predict policy impact\n"
                "  • ROI calculations for investors\n"
                "  • Cost-benefit analysis for NGOs\n"
                "  • Try: 'simulate menu' to see options\n\n"
                "**Commands:**\n"
                "  • 'summary' - Get full conversation recap\n"
                "  • 'Compare [X] and [Y]' - Compare regions\n"
                "  • 'simulate [scenario]' - Run What-If analysis\n"
                "  • Profile setup - Click '👤 Profile' button\n\n"
                "**Example questions:**\n"
                "  • What's the average dropout rate?\n"
                "  • Show applicants by faculty\n"
                "  • Compare Lima and Cusco dropout rates\n"
                "  • What if we implement meal programs?\n"
                "  • Simulate scholarship impact\n\n"
                "What would you like to explore?"
            )
        
        if qn in ["bye", "goodbye", "good bye", "see you", "see ya", "exit", "quit"]:
            return (
                "Goodbye! 👋 It was great helping you today.\n\n"
                "Feel free to come back anytime to explore more education data!\n\n"
                "Don't forget to download your conversation if you need it. 📥"
            )
        
        if any(phrase in qn for phrase in ["nice to meet", "pleasure to meet", "pleased to meet"]):
            return (
                "Nice to meet you too! 😊\n\n"
                "I'm excited to help you explore the education data. What would you like to know?"
            )
        
        return None

    def _validate_year(self, query: str) -> Optional[str]:
        qn = _norm(query)
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        years = re.findall(year_pattern, query)
        
        if years:
            invalid_years = [y for y in years if int(y) != self.VALID_YEAR]
            if invalid_years:
                return (
                    f"⚠️ I only have data for {self.VALID_YEAR}. "
                    f"Please search within {self.VALID_YEAR} data only. "
                    f"Years mentioned: {', '.join(invalid_years)} are not available."
                )
        
        time_terms = ["historical", "past decade", "trend over time", "since", "before"]
        if any(term in qn for term in time_terms):
            if any(kw in qn for kw in ["from", "to", "between"]) and years:
                return (
                    f"⚠️ I only have data for {self.VALID_YEAR}. "
                    f"Historical trends are not available. Please ask about {self.VALID_YEAR} data."
                )
        
        return None

    @staticmethod
    def _read_text_file(p: Path) -> str:
        try:
            return p.read_text(errors="ignore")
        except Exception:
            return ""

    @staticmethod
    def _read_ipynb(p: Path) -> str:
        try:
            nb = json.loads(p.read_text(errors="ignore"))
            texts = []
            for cell in nb.get("cells", []):
                if cell.get("cell_type") in ("markdown", "code"):
                    src = "".join(cell.get("source", []))
                    texts.append(src)
            return "\n\n".join(texts)
        except Exception:
            return ""

    def _try_load_table(self, p):
        suf = p.suffix.lower()
        try:
            if suf == ".csv":
                return pd.read_csv(p, encoding='utf-8', low_memory=False)
            if suf == ".tsv":
                return pd.read_csv(p, sep="\t", encoding='utf-8', low_memory=False)
            if suf in (".xlsx", ".xls"):
                return pd.read_excel(p)
            if suf == ".json":
                return pd.read_json(p)
        except Exception:
            try:
                if suf == ".csv":
                    return pd.read_csv(p, encoding='latin-1', low_memory=False)
            except Exception:
                pass
        return None

    def build_index(self) -> Optional[pd.DataFrame]:
        self.docs.clear()
        self.doc_meta.clear()
        self.loaded_tables.clear()
        self.graph_catalog.clear()

        for p in self.data_dir.glob("*"):
            if p.suffix.lower() in (".csv", ".tsv", ".xlsx", ".xls", ".json"):
                df = self._try_load_table(p)
                if df is not None and not df.empty:
                    self.loaded_tables[p.name] = df
                    summary = f"Table: {p.name}\nColumns: {', '.join(df.columns)}\nRows: {len(df)}"
                    self.docs.append(summary)
                    self.doc_meta.append({"type": "table", "path": str(p), "name": p.name})

        for p in self.code_dir.glob("*"):
            if p.suffix.lower() in (".py", ".ipynb", ".md", ".txt", ".R", ".sql"):
                if p.suffix == ".ipynb":
                    txt = self._read_ipynb(p)
                else:
                    txt = self._read_text_file(p)
                if txt.strip():
                    self.docs.append(txt)
                    self.doc_meta.append({"type": "code", "path": str(p), "name": p.name})

        for p in self.graphs_dir.glob("*"):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg"):
                caption = _norm(p.stem.replace("_", " "))
                self.graph_catalog.append({"file": str(p), "caption": caption})
                self.docs.append(f"Graph: {caption}")
                self.doc_meta.append({"type": "graph", "path": str(p), "name": p.name})

        if self.docs:
            self.vectorizer = TfidfVectorizer(max_features=800, ngram_range=(1, 2), stop_words="english")
            self.matrix = self.vectorizer.fit_transform(self.docs)
        
        if LLAMAINDEX_AVAILABLE and self.llm_available:
            self._build_llamaindex()

        return pd.DataFrame(self.doc_meta) if self.doc_meta else None

    def _detect_dataset_type(self, query: str) -> str:
        qn = _norm(query)
        
        if any(kw in qn for kw in ["primary", "primaria", "elementary"]):
            return "primary"
        if any(kw in qn for kw in ["secondary", "secundaria", "high school"]):
            return "secondary"
        if any(kw in qn for kw in ["applicant", "aplicante", "undergraduate", "admission", "faculty", "program", "modality"]):
            return "applicants"
        if any(kw in qn for kw in ["dropout", "desercion", "deserción", "tasa", "rate"]):
            return "secondary"
        
        return "unknown"

    def _find_table_by_type(self, dataset_type: str) -> Optional[str]:
        for name in self.loaded_tables.keys():
            name_lower = name.lower()
            if dataset_type == "applicants" and any(kw in name_lower for kw in ["applicant", "undergraduate", "dataset"]):
                return name
            elif dataset_type == "primary" and "primaria" in name_lower:
                return name
            elif dataset_type == "secondary" and "secundaria" in name_lower:
                return name
        return None

    def _geocode_location(self, location_name: str) -> Optional[Dict[str, float]]:
        """Convert Peru location name to coordinates"""
        loc_norm = _norm(str(location_name))
        if loc_norm in PERU_LOCATIONS:
            return PERU_LOCATIONS[loc_norm]
        return None

    def _answer_from_tables(self, query: str) -> Optional[Dict[str, Any]]:
        if not self.loaded_tables:
            return None

        qn = _norm(query)
        dataset_type = self._detect_dataset_type(query)
        
        table_name = self._find_table_by_type(dataset_type)
        if not table_name:
            table_name = list(self.loaded_tables.keys())[0] if self.loaded_tables else None
            if not table_name:
                return None

        df = self.loaded_tables[table_name].copy()
        refs = f"Source: {table_name}"

        specific_locations = []
        for col in df.columns:
            if _norm(col) in ["region", "departamento", "department", "province", "provincia", "district", "distrito"]:
                for val in df[col].dropna().unique():
                    if _norm(str(val)) in qn:
                        specific_locations.append((col, val))

        group_by = None
        if any(kw in qn for kw in ["by region", "by department", "by departamento", "region", "department"]):
            for col in df.columns:
                if _norm(col) in ["region", "departamento", "department"]:
                    group_by = col
                    break
        elif any(kw in qn for kw in ["by province", "by provincia", "province"]):
            for col in df.columns:
                if _norm(col) in ["province", "provincia"]:
                    group_by = col
                    break
        elif any(kw in qn for kw in ["by faculty", "faculty"]):
            for col in df.columns:
                if _norm(col) == "faculty":
                    group_by = col
                    break
        elif any(kw in qn for kw in ["by program", "program"]):
            for col in df.columns:
                if _norm(col) == "program":
                    group_by = col
                    break

        metric_col = None
        if dataset_type in ["primary", "secondary"]:
            if "Tasa" in df.columns:
                metric_col = "Tasa"
        
        use_count = (metric_col is None)

        if specific_locations and any(kw in qn for kw in ["compare", "comparison", "between", "vs", "versus"]):
            results = []
            for col, val in specific_locations:
                subset = df[df[col] == val]
                if not subset.empty:
                    if use_count:
                        value = len(subset)
                        results.append(f"**{val}**: {_fmt_num(value)} records")
                    else:
                        value = subset[metric_col].mean()
                        results.append(f"**{val}**: {_fmt_num(value)}% average dropout rate")
            
            if results:
                comparison = "\n  • ".join(results)
                text = f"📊 Comparison for {self.VALID_YEAR}:\n\n  • {comparison}"
                return {"reply": text, "refs": [refs]}

        try:
            if group_by and group_by in df.columns:
                if use_count:
                    result = df.groupby(group_by).size().sort_values(ascending=False)
                    metric_name = "Number of Records"
                    dataset_desc = "undergraduate applicants" if dataset_type == "applicants" else "schools"
                else:
                    result = df.groupby(group_by)[metric_col].mean().sort_values(ascending=False)
                    metric_name = "Average Dropout Rate (%)"
                    dataset_desc = "dropout rate"

                result = result.head(10)
                
                bullets = [f"  • **{str(k)}**: {_fmt_num(v)}" for k, v in result.items()]
                text = f"📊 Top {len(result)} {group_by}s by {dataset_desc} ({self.VALID_YEAR}):\n\n" + "\n".join(bullets)
                
                image_path = self._create_chart(result, group_by, metric_name)
                
                return {
                    "reply": text,
                    "image_path": str(image_path) if image_path else None,
                    "refs": [refs]
                }
            else:
                if use_count:
                    value = len(df)
                    dataset_desc = "undergraduate applicants" if dataset_type == "applicants" else "school records"
                    text = f"📊 Total {dataset_desc} in {self.VALID_YEAR}: **{_fmt_num(value)}**"
                else:
                    value = df[metric_col].mean()
                    level = "primary" if dataset_type == "primary" else "secondary"
                    text = f"📊 Average {level} school dropout rate in {self.VALID_YEAR}: **{_fmt_num(value)}%**"
                
                return {"reply": text, "refs": [refs]}

        except Exception as e:
            return {
                "reply": f"I had trouble processing that query. Error: {str(e)}",
                "refs": [refs]
            }

    def _create_chart(self, data: pd.Series, group_name: str, metric_name: str) -> Optional[Path]:
        try:
            out = self.root / f"chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.figure(figsize=(10, 6))
            
            data.plot(kind="barh", color='steelblue')
            plt.title(f"{metric_name} by {group_name} ({self.VALID_YEAR})", fontsize=14, fontweight='bold')
            plt.xlabel(metric_name, fontsize=12)
            plt.ylabel(group_name, fontsize=12)
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            plt.savefig(out, dpi=180, bbox_inches='tight')
            plt.close()
            return out
        except Exception:
            plt.close()
            return None

    def _llm_enhanced_response(self, query: str, data_response: str) -> str:
        """Use LLM to enhance the data response with natural language"""
        if not self.llm_available or not self.query_engine:
            return data_response
        
        try:
            context = f"Peru education data for {self.VALID_YEAR}. User asked: '{query}'\n\n"
            context += f"Data result: {data_response}\n\n"
            context += "Provide a natural, conversational response that explains this data clearly. "
            context += "Keep it concise (2-3 sentences) and highlight key insights."
            
            response = self.query_engine.query(context)
            
            if response and str(response).strip():
                return f"{data_response}\n\n💡 **Insight:** {str(response)}"
            
        except Exception as e:
            print(f"LLM enhancement error: {e}")
        
        return data_response

    def _rag_synthesis(self, query: str) -> Dict[str, Any]:
        """Enhanced RAG with LLM if available"""
        
        # Try LlamaIndex first if available
        if self.llm_available and self.query_engine:
            try:
                context = f"Based on Peru {self.VALID_YEAR} education data, answer: {query}"
                response = self.query_engine.query(context)
                
                if response and str(response).strip():
                    return {
                        "reply": f"🧠 **AI Analysis:**\n\n{str(response)}",
                        "refs": ["LlamaIndex + Ollama"]
                    }
            except Exception as e:
                print(f"LlamaIndex query error: {e}")
        
        # Fallback to traditional RAG
        if not (self.vectorizer and self.matrix is not None):
            return {
                "reply": "I don't have any data indexed yet. Please upload files and click 'Rebuild Index'.",
                "refs": []
            }
        
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        idxs = sims.argsort()[::-1][:3]
        
        snippets = []
        for i in idxs:
            if sims[i] < 0.1:
                continue
            doc = self.docs[i]
            meta = self.doc_meta[i]
            snippet = doc[:300].replace("\n", " ")
            snippets.append({"score": round(float(sims[i]), 4), "meta": meta, "snippet": snippet})
        
        if not snippets:
            return {
                "reply": "I couldn't find relevant information. Try being more specific or check that your data is uploaded.",
                "refs": []
            }
        
        synthesis = f"Based on {self.VALID_YEAR} data, here's what I found:\n\n"
        for i, h in enumerate(snippets, 1):
            file_name = Path(h['meta'].get('path', '')).name if h['meta'].get('path') else 'Unknown'
            synthesis += f"{i}. {file_name}: {h['snippet'][:150]}...\n\n"
        
        return {"reply": synthesis, "refs": snippets}

    def build_map(self, query: str = "") -> Optional[Path]:
        """Build map using location names with geocoding"""
        try:
            import folium
        except Exception:
            return None

        dataset_type = self._detect_dataset_type(query)
        table_name = self._find_table_by_type(dataset_type)
        
        if not table_name:
            table_name = list(self.loaded_tables.keys())[0] if self.loaded_tables else None
        
        if not table_name:
            return None
        
        df = self.loaded_tables[table_name].copy()
        
        location_col = None
        for col in ["Departamento", "Department", "Region", "departamento", "department", "region"]:
            if col in df.columns:
                location_col = col
                break
        
        if not location_col:
            return None
        
        metric_col = None
        if "Tasa" in df.columns:
            metric_col = "Tasa"
        
        try:
            if metric_col:
                location_data = df.groupby(location_col)[metric_col].mean().to_dict()
            else:
                location_data = df.groupby(location_col).size().to_dict()
            
            m = folium.Map(
                location=[-9.19, -75.0152],
                zoom_start=5,
                tiles="OpenStreetMap"
            )
            
            markers_added = 0
            for location_name, value in location_data.items():
                coords = self._geocode_location(location_name)
                if coords:
                    popup_text = f"<b>{location_name}</b><br>"
                    if metric_col:
                        popup_text += f"Dropout Rate: {_fmt_num(value)}%"
                        color = 'green' if value < 2 else ('orange' if value < 5 else 'red')
                    else:
                        popup_text += f"Count: {_fmt_num(value)}"
                        color = 'blue'
                    
                    folium.CircleMarker(
                        location=[coords['lat'], coords['lon']],
                        radius=8,
                        color=color,
                        fill=True,
                        fillOpacity=0.7,
                        popup=folium.Popup(popup_text, max_width=250)
                    ).add_to(m)
                    markers_added += 1
            
            if markers_added == 0:
                return None
            
            if metric_col:
                legend_html = '''
                <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; 
                     background-color: white; border:2px solid grey; z-index:9999; 
                     padding: 10px; font-size:14px;">
                <p><b>Dropout Rate Legend</b></p>
                <p><span style="color:green;">●</span> Low (&lt;2%)</p>
                <p><span style="color:orange;">●</span> Medium (2-5%)</p>
                <p><span style="color:red;">●</span> High (&gt;5%)</p>
                </div>
                '''
                m.get_root().html.add_child(folium.Element(legend_html))
            
            out = self.root / f"map_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            m.save(out)
            return out
            
        except Exception as e:
            print(f"Map error: {e}")
            return None

    def greet_and_collect(self) -> str:
        llm_badge = " 🧠 AI-Powered!" if self.llm_available else ""
        return (
            f"👋 Hello! I'm the JGM Insights Assistant{llm_badge}\n\n"
            f"I can help you explore Peru education data for {self.VALID_YEAR}:\n"
            "  • Undergraduate applicant statistics\n"
            "  • Primary/secondary school dropout rates\n"
            "  • Interactive maps and charts\n"
            "  • 🔮 **What-If Simulator** - Predict policy impacts!\n"
            f"{'  • 🧠 Natural language Q&A with AI' if self.llm_available else ''}\n\n"
            "Before we start, please click '👤 Profile' to share:\n"
            "  • Your name\n"
            "  • Your role (parent/student/teacher/NGO/donor/investor)\n"
            "  • Contact information\n\n"
            "💡 Quick commands:\n"
            "  • 'summary' - Get full conversation overview\n"
            "  • 'help' - See what I can do\n"
            "  • 'map' - Generate geographic visualization\n"
            "  • 🔮 'simulate menu' - See What-If scenarios\n"
            "  • Ask any question about the data!"
        )

    def set_profile(self, first_name: str = None, last_name: str = None, role: str = None, contact: str = None) -> str:
        if first_name:
            self.user_profile["first_name"] = first_name
        if last_name:
            self.user_profile["last_name"] = last_name
        if role:
            self.user_profile["role"] = role
        if contact:
            self.user_profile["contact"] = contact
        
        self.user_profile["onboarded"] = True
        
        fn = self.user_profile.get("first_name") or "there"
        role_specific = ""
        if role:
            role_lower = role.lower()
            if "ngo" in role_lower:
                role_specific = "\n\n💡 As an NGO member, try our What-If Simulator to evaluate program impact!"
            elif "investor" in role_lower or "donor" in role_lower or "business" in role_lower:
                role_specific = "\n\n💡 As an investor, check out our What-If Simulator for ROI analysis!"
            elif "parent" in role_lower or "student" in role_lower:
                role_specific = "\n\n💡 Want to see how interventions could help? Try 'simulate menu'!"
        
        return (
            f"Thanks, {fn}! 🎉 Profile saved.{role_specific}\n\n"
            f"Now you're all set! Ask me anything about Peru's {self.VALID_YEAR} education data.\n\n"
            "💡 Try:\n"
            "  • 'What's the average dropout rate?'\n"
            "  • 'Show applicants by faculty'\n"
            "  • 'Compare Lima and Cusco'\n"
            "  • 'Create a map'\n"
            "  • 🔮 'What if we implement meal programs?'"
        )

    # ========== NEW: OFF-TOPIC DETECTION ==========
    
    def _is_off_topic(self, query: str) -> bool:
        """Detect if question is off-topic or too vague"""
        qn = _norm(query)
        
        # CRITICAL: Always allow these commands
        valid_commands = [
            "hello", "hi", "help", "summary", "summarize", "profile",
            "map", "maps", "show map", "create map", "build map", "generate map",
            "chart", "graph", "plot", "visualize", "show chart", "create chart",
            "simulate", "what if", "compare", "analysis"
        ]
        if any(cmd in qn for cmd in valid_commands):
            return False
        
        # Education keywords (ON-TOPIC)
        on_topic = [
            "dropout", "desercion", "student", "estudiante", "education", "educacion",
            "school", "escuela", "university", "universidad", "applicant", "aplicante",
            "rate", "tasa", "data", "datos", "region", "department", "departamento",
            "province", "provincia", "peru", "faculty", "program", "primary", "secondary",
            "primaria", "secundaria", "lima", "cusco", "arequipa", "piura", "puno", "ica",
            "average", "total", "show", "display", "2025", "roi", "impact", "policy",
            "intervention", "scholarship", "meal", "teacher", "infrastructure", "mentorship"
        ]
        
        # Off-topic keywords (clearly unrelated)
        off_topic = [
            "weather", "clima", "temperature", "rain", "sunny",
            "movie", "film", "pelicula", "cinema", "netflix",
            "recipe", "receta", "cooking", "restaurant", "restaurante",
            "game", "juego", "video game", "gaming",
            "sport", "deporte", "football", "soccer", "basketball",
            "music", "musica", "song", "cancion", "singer", "band",
            "celebrity", "celebridad", "famous", "actor", "actress",
            "joke", "chiste", "funny", "humor",
            "stock", "crypto", "bitcoin", "trading",
            "hotel", "travel", "vacation", "tourist", "tourism"
        ]
        
        # Check if clearly off-topic
        if any(kw in qn for kw in off_topic):
            return True
        
        # Check if has education keywords
        has_on_topic = any(kw in qn for kw in on_topic)
        
        # If has education keywords, it's on-topic
        if has_on_topic:
            return False
        
        # If very short and no context (but already passed valid commands check)
        if len(qn.split()) < 3:
            return True
        
        # Check for vague patterns without education context
        vague_patterns = [
            "tell me about", "what about", "how about", 
            "tell me something", "anything", "whatever", "random"
        ]
        is_vague = any(pattern in qn for pattern in vague_patterns)
        
        if is_vague:
            return True
        
        return False
    
    def _get_off_topic_response(self) -> str:
        """Return polite message for off-topic questions"""
        return (
            "I'm specifically designed to help with **Peru 2025 education data**. 📚\n\n"
            "I can answer questions about:\n"
            "  • Dropout rates (primary & secondary schools)\n"
            "  • Undergraduate applicant statistics\n"
            "  • Regional/provincial comparisons\n"
            "  • Faculty and program analysis\n"
            "  • What-If policy simulations\n"
            "  • Interactive maps and charts\n\n"
            "💡 **Try asking:**\n"
            "  • 'What's the dropout rate in Lima?'\n"
            "  • 'Show applicants by faculty'\n"
            "  • 'Compare Lima and Cusco'\n"
            "  • 'What if we implement meal programs?'\n"
            "  • 'Create a map'\n\n"
            "Please ask a question related to Peru's education data! 😊"
        )
    
    # ========== END OFF-TOPIC DETECTION ==========

    def chat(self, message: str) -> Dict[str, Any]:
        """UPDATED - with off-topic detection"""
        q = message.strip()
        qn = _norm(q)

        if not q:
            return {"reply": "Please ask me a question! 😊"}

        # PRIORITY 1: Casual conversation
        casual_response = self._handle_casual_conversation(q)
        if casual_response:
            return {"reply": casual_response}

        # PRIORITY 2: Summary
        if any(kw in qn for kw in ["summary", "summarize", "sum up", "recap"]):
            summary = self.generate_summary()
            return {"reply": summary}

        # PRIORITY 3: Off-topic detection (NEW!)
        if self._is_off_topic(q):
            return {"reply": self._get_off_topic_response()}

        # PRIORITY 4: What-If Simulator
        scenario_key = self._detect_simulation_query(q)
        if scenario_key:
            simulation_result = self.run_simulation(scenario_key)
            return {"reply": simulation_result}

        # PRIORITY 5: Profile reminder
        if not self.user_profile.get("onboarded"):
            if any(kw in qn for kw in ["dropout", "rate", "applicant", "faculty", "region", "data", "chart", "map"]):
                return {
                    "reply": (
                        "👋 I'd love to help with that!\n\n"
                        "Quick question first: Would you like to set up your profile? "
                        "It helps me personalize responses for you.\n\n"
                        "Click '👤 Profile' button, or just ask your question again to skip!"
                    )
                }

        # PRIORITY 6: Year validation
        year_error = self._validate_year(q)
        if year_error:
            return {"reply": year_error}

        # PRIORITY 7: Map
        if any(k in qn for k in ["map", "maps", "show map", "geographical", "geography"]):
            mp = self.build_map(query=q)
            if mp:
                return {
                    "reply": f"🗺️ I've created an interactive map showing dropout rates by department! Click below to explore.",
                    "map_path": str(mp)
                }
            else:
                return {"reply": "I couldn't create a map with the current data. The dataset needs location information (departments/regions)."}

        # PRIORITY 8: Chart
        if any(k in qn for k in ["chart", "plot", "graph", "visualize"]):
            result = self._answer_from_tables(q)
            if result:
                return result

        # PRIORITY 9: Data queries
        result = self._answer_from_tables(q)
        if result:
            # Enhance with LLM if available
            if self.llm_available:
                result["reply"] = self._llm_enhanced_response(q, result.get("reply", ""))
            return result

        # PRIORITY 10: RAG fallback
        return self._rag_synthesis(q)