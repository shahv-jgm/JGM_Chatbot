#!/usr/bin/env python3
"""
Flask Secret Key Setup Script (Python Version)
Works on Windows, Mac, and Linux
"""

import os
import secrets
import sys

def generate_secret_key():
    """Generate a secure secret key"""
    return secrets.token_hex(32)

def check_file_exists(filename):
    """Check if a file exists"""
    return os.path.exists(filename)

def read_env_file():
    """Read .env file and return as dict"""
    env_vars = {}
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    return env_vars

def write_env_file(env_vars):
    """Write dict to .env file"""
    with open('.env', 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")

def main():
    print("🔐 Flask Secret Key Setup Script")
    print("=" * 50)
    print()

    # Step 1: Check for .env file
    print("Step 1: Checking for .env file...")
    if check_file_exists('.env'):
        print("✅ .env file exists")
        env_vars = read_env_file()
    else:
        print("⚠️  .env file not found - creating one...")
        env_vars = {}
        with open('.env', 'w') as f:
            f.write("# JGM Chatbot Environment Variables\n")
        print("✅ Created .env file")
    print()

    # Step 2: Check for FLASK_SECRET_KEY
    print("Step 2: Checking for FLASK_SECRET_KEY...")
    if 'FLASK_SECRET_KEY' in env_vars and env_vars['FLASK_SECRET_KEY']:
        print("✅ FLASK_SECRET_KEY found in .env")
        print(f"   Current key (first 20 chars): {env_vars['FLASK_SECRET_KEY'][:20]}...")
        print()
        response = input("Do you want to generate a new key? (y/N): ").strip().lower()
        generate_new = response in ['y', 'yes']
    else:
        print("⚠️  FLASK_SECRET_KEY not found in .env")
        generate_new = True
    print()

    # Step 3: Generate key if needed
    if generate_new:
        print("Step 3: Generating new secret key...")
        new_key = generate_secret_key()
        print(f"✅ Generated new key: {new_key[:20]}...")
        env_vars['FLASK_SECRET_KEY'] = new_key
        write_env_file(env_vars)
        print("✅ Added to .env file")
        print()
        print("📋 Your new secret key:")
        print(f"   {new_key}")
        print()
        print("⚠️  Save this key! You'll need it for Railway deployment.")
    else:
        print("Step 3: Keeping existing key")
    print()

    # Step 4: Check .gitignore
    print("Step 4: Checking .gitignore...")
    if check_file_exists('.gitignore'):
        with open('.gitignore', 'r') as f:
            gitignore_content = f.read()
        
        if '.env' in gitignore_content:
            print("✅ .env is in .gitignore")
        else:
            print("⚠️  Adding .env to .gitignore...")
            with open('.gitignore', 'a') as f:
                f.write("\n# Environment variables\n")
                f.write(".env\n")
                f.write("*.env\n")
                f.write(".env.*\n")
            print("✅ Added .env to .gitignore")
    else:
        print("⚠️  .gitignore not found - creating one...")
        with open('.gitignore', 'w') as f:
            f.write("# Environment variables\n")
            f.write(".env\n")
            f.write("*.env\n")
            f.write(".env.*\n")
            f.write("\n# Python\n")
            f.write("__pycache__/\n")
            f.write("*.py[cod]\n")
            f.write("*$py.class\n")
            f.write("*.so\n")
            f.write(".Python\n")
            f.write("venv/\n")
            f.write("env/\n")
            f.write("*.log\n")
            f.write(".DS_Store\n")
        print("✅ Created .gitignore")
    print()

    # Step 5: Create .env.example
    print("Step 5: Creating .env.example template...")
    if check_file_exists('.env.example'):
        print("   .env.example already exists")
    else:
        with open('.env.example', 'w') as f:
            f.write("# Environment Variables Template\n")
            f.write("# Copy this to .env and fill in your actual values\n\n")
            f.write("GOOGLE_API_KEY=your_google_api_key_here\n")
            f.write("FLASK_SECRET_KEY=your_flask_secret_key_here\n")
            f.write("FLASK_HOST=0.0.0.0\n")
            f.write("FLASK_PORT=5050\n")
            f.write("PRODUCTION_MODE=True\n")
            f.write("JGM_WORKSPACE=jgm_workspace\n")
        print("✅ Created .env.example")
    print()

    # Step 6: Check GOOGLE_API_KEY
    print("Step 6: Checking for GOOGLE_API_KEY...")
    if 'GOOGLE_API_KEY' in env_vars and env_vars['GOOGLE_API_KEY'] and not env_vars['GOOGLE_API_KEY'].startswith('your_'):
        print("✅ GOOGLE_API_KEY is set")
    else:
        print("⚠️  GOOGLE_API_KEY not set properly")
        print("   Please add your Google API key to .env:")
        print("   GOOGLE_API_KEY=AIzaSy...")
    print()

    # Step 7: Summary
    print("Step 7: Summary of .env file...")
    print("=" * 50)
    if check_file_exists('.env'):
        print("Current .env contents (keys hidden):")
        env_vars = read_env_file()
        for key, value in env_vars.items():
            if len(value) > 20:
                print(f"  ✅ {key} = {value[:20]}...")
            elif not value or value.startswith('your_'):
                print(f"  ⚠️  {key} = (not set)")
            else:
                print(f"  ✅ {key} = {value}")
    else:
        print("⚠️  .env file not found")
    print()

    # Final message
    print("=" * 50)
    print("✅ Setup Complete!")
    print()
    print("📋 Next Steps:")
    print("1. Verify your .env file has all required keys")
    print("2. Test locally: python app.py")
    print("3. For Railway deployment, run:")
    print("   railway variables set FLASK_SECRET_KEY=your_key_here")
    print()
    print("🔒 Security Reminders:")
    print("- .env is in .gitignore ✓")
    print("- Never commit .env to GitHub")
    print("- Keep your keys secret")
    print()
    print("Need help? Check FLASK_SECRET_KEY_SETUP.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)