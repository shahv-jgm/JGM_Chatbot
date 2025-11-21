# Test Script for New Chatbot Version

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from jgm_rag_chatbot import JGMRAG

# Initialize
workspace = Path("./jgm_workspace")
bot = JGMRAG(workspace)

# Build index
print("=" * 60)
print("Building index...")
result = bot.build_index()
if result is not None:
    print(f"✅ Index built! Found {len(result)} items")
else:
    print("❌ No data found. Check jgm_workspace/data/ folder")
    sys.exit(1)

print("\n" + "=" * 60)
print("RUNNING TESTS")
print("=" * 60)

# Test queries
test_queries = [
    "What's the average secondary dropout rate?",
    "Show dropout rates by department",
    "How many applicants by faculty?",
    "What's the primary dropout rate?",
]

for i, query in enumerate(test_queries, 1):
    print(f"\n📝 Test {i}: {query}")
    print("-" * 60)
    
    try:
        response = bot.chat(query)
        reply = response.get("reply", "No reply")
        
        # Truncate long replies
        if len(reply) > 300:
            reply = reply[:300] + "..."
        
        print(f"✅ Response: {reply}")
        
        if response.get("image_path"):
            print(f"📊 Chart created: {response['image_path']}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print("\nIf all tests passed, the chatbot is ready!")
print("Run: python app.py")