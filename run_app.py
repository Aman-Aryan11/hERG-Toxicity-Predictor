#!/usr/bin/env python3
"""
Startup script for hERG Toxicity Predictor Streamlit App
"""

import subprocess
import sys
import os
import argparse

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'torch', 'torch_geometric', 'rdkit', 
        'scikit-learn', 'pandas', 'numpy', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements_deploy.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_model_file():
    """Check if the model file exists"""
    model_file = "gnn_best_model.pt"
    if not os.path.exists(model_file):
        print(f"❌ Model file '{model_file}' not found")
        print("   Make sure the trained model file is in the current directory")
        return False
    
    print(f"✅ Model file '{model_file}' found")
    return True

def run_streamlit(port=8501, host="localhost"):
    """Run the Streamlit application"""
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true"
    ]
    
    print(f"🚀 Starting hERG Toxicity Predictor on http://{host}:{port}")
    print("   Press Ctrl+C to stop the application")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="hERG Toxicity Predictor")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the app on (default: 8501)")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency and model file checks")
    
    args = parser.parse_args()
    
    print("🧬 hERG Toxicity Predictor - Startup")
    print("=" * 50)
    
    if not args.skip_checks:
        print("\n🔍 Running pre-flight checks...")
        
        if not check_dependencies():
            sys.exit(1)
        
        if not check_model_file():
            sys.exit(1)
        
        print("\n✅ All checks passed!")
    
    print("\n" + "=" * 50)
    run_streamlit(args.port, args.host)

if __name__ == "__main__":
    main() 