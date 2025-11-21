#!/usr/bin/env python3
"""
Quick Map Test - Verify map functionality
"""

import sys
from pathlib import Path

def test_map():
    print("🗺️  TESTING MAP FUNCTIONALITY\n")
    print("=" * 60)
    
    # Test 1: Check folium installation
    print("\n1. Testing folium installation...")
    try:
        import folium
        print("   ✅ folium is installed")
        print(f"   Version: {folium.__version__}")
    except ImportError:
        print("   ❌ folium NOT installed")
        print("   Fix: pip install folium")
        return False
    
    # Test 2: Check chatbot
    print("\n2. Testing chatbot initialization...")
    try:
        from jgm_rag_chatbot import JGMRAG
        from pathlib import Path
        
        workspace = Path("jgm_workspace")
        bot = JGMRAG(workspace)
        bot.build_index()
        print(f"   ✅ Chatbot initialized")
        print(f"   Data files loaded: {len(bot.loaded_tables)}")
    except Exception as e:
        print(f"   ❌ Chatbot failed: {e}")
        return False
    
    # Test 3: Test map queries
    print("\n3. Testing map queries...")
    
    test_queries = [
        "show map",
        "create map",
        "map",
        "show me a map"
    ]
    
    for query in test_queries:
        try:
            response = bot.chat(query)
            reply = response.get("reply", "")
            map_path = response.get("map_path")
            
            if map_path:
                print(f"   ✅ '{query}' → Map created: {Path(map_path).name}")
            elif "can't create a map" in reply.lower() or "couldn't create" in reply.lower():
                print(f"   ⚠️  '{query}' → No location data in dataset")
            elif "education data" in reply.lower() and "can answer" in reply.lower():
                print(f"   ❌ '{query}' → Blocked by off-topic filter!")
            else:
                print(f"   ⚠️  '{query}' → {reply[:50]}...")
        except Exception as e:
            print(f"   ❌ '{query}' → Error: {e}")
    
    # Test 4: Try building map directly
    print("\n4. Testing direct map building...")
    try:
        map_path = bot.build_map()
        if map_path and Path(map_path).exists():
            print(f"   ✅ Map created: {map_path}")
        else:
            print(f"   ⚠️  Map not created - check if data has location columns")
            print(f"   Available tables: {list(bot.loaded_tables.keys())}")
            if bot.loaded_tables:
                first_table = list(bot.loaded_tables.keys())[0]
                df = bot.loaded_tables[first_table]
                print(f"   Columns in {first_table}: {list(df.columns)}")
    except Exception as e:
        print(f"   ❌ Direct map build failed: {e}")
    
    print("\n" + "=" * 60)
    return True

if __name__ == "__main__":
    print("\n🤖 JGM MAP FUNCTIONALITY TEST\n")
    success = test_map()
    
    if success:
        print("\n✅ Map test completed!")
        print("\nIf maps aren't working, check:")
        print("  1. folium is installed: pip install folium")
        print("  2. Your data has location columns (Departamento, Department, Region)")
        print("  3. Location names match Peru regions (Lima, Cusco, etc.)")
    else:
        print("\n❌ Map test failed - fix errors above")
    
    sys.exit(0 if success else 1)