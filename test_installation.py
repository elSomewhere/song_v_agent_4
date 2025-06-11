#!/usr/bin/env python3
"""Test script to verify VC-RAG-SBG installation."""

import sys
import os
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import openai
        print("✓ OpenAI library")
    except ImportError:
        print("✗ OpenAI library not found - run: pip install openai")
        return False
    
    try:
        import lancedb
        print("✓ LanceDB")
    except ImportError:
        print("✗ LanceDB not found - run: pip install lancedb")
        return False
    
    try:
        import langgraph
        print("✓ LangGraph")
    except ImportError:
        print("✗ LangGraph not found - run: pip install langgraph")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL/Pillow")
    except ImportError:
        print("✗ Pillow not found - run: pip install pillow")
        return False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError:
        print("✗ PyYAML not found - run: pip install pyyaml")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✓ python-dotenv")
    except ImportError:
        print("✗ python-dotenv not found - run: pip install python-dotenv")
        return False
    
    # Test local imports
    try:
        from src.loader import Loader
        from src.models import WorkflowState
        from src.memory import MemoryService
        print("✓ Local modules")
    except ImportError as e:
        print(f"✗ Error importing local modules: {e}")
        return False
    
    return True


def test_environment():
    """Test environment setup."""
    print("\nTesting environment...")
    
    # Check for .env file
    if Path(".env").exists():
        print("✓ .env file found")
        
        # Load and check API key
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key.startswith("sk-"):
            print("✓ OpenAI API key configured")
        else:
            print("✗ OpenAI API key not found or invalid in .env")
            return False
    else:
        print("✗ .env file not found - copy example.env to .env and add your API key")
        return False
    
    return True


def test_data_files():
    """Test that example data files exist."""
    print("\nTesting data files...")
    
    required_files = [
        ("data/script.md", "Script file"),
        ("data/style.md", "Style guide"),
        ("data/entities.md", "Entities file"),
        ("config.yaml", "Configuration file")
    ]
    
    all_found = True
    for filepath, description in required_files:
        if Path(filepath).exists():
            print(f"✓ {description} found")
        else:
            print(f"✗ {description} not found at {filepath}")
            all_found = False
    
    # Check for reference images
    refs_dir = Path("data/refs")
    if refs_dir.exists() and refs_dir.is_dir():
        image_count = len(list(refs_dir.glob("**/*.png")) + 
                         list(refs_dir.glob("**/*.jpg")) + 
                         list(refs_dir.glob("**/*.jpeg")))
        print(f"✓ Reference directory found with {image_count} images")
    else:
        print("! Reference directory not found (optional)")
    
    return all_found


def main():
    """Run all tests."""
    print("VC-RAG-SBG Installation Test")
    print("=" * 40)
    
    tests_passed = []
    
    # Run tests
    tests_passed.append(test_imports())
    tests_passed.append(test_environment())
    tests_passed.append(test_data_files())
    
    # Summary
    print("\n" + "=" * 40)
    if all(tests_passed):
        print("✓ All tests passed! You're ready to run: python run.py")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 