#!/usr/bin/env python3
"""
Enhanced Context Examples for VC-RAG-SBG
Demonstrates how to configure and use the enhanced context system for long storyboards.
"""

import yaml
from pathlib import Path

def create_basic_config():
    """Basic configuration for short storyboards."""
    return {
        'ctx_images': 3,
        'ctx_window': 4,
        'context_mode': 'basic',
        'global_context_enabled': False,
        'budget_usd': 25
    }

def create_enhanced_config():
    """Enhanced configuration for medium-length storyboards."""
    return {
        'ctx_images': 6,
        'ctx_window': 8,
        'context_mode': 'enhanced',
        'global_context_enabled': True,
        'global_ctx_window': 15,
        'global_ctx_weight': 0.3,
        'character_consistency_window': 30,
        'environment_consistency_window': 20,
        'consistency_tracking': {
            'characters': True,
            'environments': True,
            'style_elements': True,
            'narrative_flow': True
        },
        'budget_usd': 40
    }

def create_global_config():
    """Global configuration for long storyboards with maximum consistency."""
    return {
        'ctx_images': 10,
        'ctx_window': 12,
        'context_mode': 'enhanced',
        'global_context_enabled': True,
        'global_ctx_window': 50,
        'global_ctx_weight': 0.5,
        'character_consistency_window': 100,
        'environment_consistency_window': 75,
        'consistency_tracking': {
            'characters': True,
            'environments': True,
            'style_elements': True,
            'narrative_flow': True
        },
        'token_cap': 6000,  # Higher token limit for global context
        'budget_usd': 75    # Higher budget for longer processing
    }

def run_with_basic_context():
    """Run storyboard generation with basic context (fast, less consistent)."""
    print("🚀 Running with BASIC context...")
    print("   - Fast generation")
    print("   - Lower token usage")
    print("   - Good for short scenes or prototyping")
    print("   - May have consistency issues in long storyboards")
    
    # Save basic config
    config = create_basic_config()
    with open('config_basic.yaml', 'w') as f:
        yaml.dump(config, f, indent=2)
    
    # Example command
    print("\n💻 Command:")
    print("python run.py --config config_basic.yaml")

def run_with_enhanced_context():
    """Run storyboard generation with enhanced context (balanced)."""
    print("⚡ Running with ENHANCED context...")
    print("   - Balanced speed and consistency")
    print("   - Character/environment tracking")
    print("   - Good for medium-length storyboards")
    print("   - Reasonable token usage")
    
    # Save enhanced config
    config = create_enhanced_config()
    with open('config_enhanced.yaml', 'w') as f:
        yaml.dump(config, f, indent=2)
    
    # Example command
    print("\n💻 Command:")
    print("python run.py --config config_enhanced.yaml")

def run_with_global_context():
    """Run storyboard generation with global context (slow, maximum consistency)."""
    print("🌟 Running with GLOBAL context...")
    print("   - Maximum consistency across long storyboards")
    print("   - Extensive character/environment tracking")
    print("   - Higher token usage and cost")
    print("   - Best for final production storyboards")
    
    # Save global config
    config = create_global_config()
    with open('config_global.yaml', 'w') as f:
        yaml.dump(config, f, indent=2)
    
    # Example command
    print("\n💻 Command:")
    print("python run.py --config config_global.yaml")

def demonstrate_config_differences():
    """Show the differences between configuration levels."""
    print("📊 CONFIGURATION COMPARISON\n")
    
    configs = {
        'Basic': create_basic_config(),
        'Enhanced': create_enhanced_config(),
        'Global': create_global_config()
    }
    
    print("| Feature                    | Basic | Enhanced | Global |")
    print("|----------------------------|-------|----------|--------|")
    print(f"| Context Images             | {configs['Basic']['ctx_images']:5} | {configs['Enhanced']['ctx_images']:8} | {configs['Global']['ctx_images']:6} |")
    print(f"| Context Window             | {configs['Basic']['ctx_window']:5} | {configs['Enhanced']['ctx_window']:8} | {configs['Global']['ctx_window']:6} |")
    print(f"| Global Context             | {'No':5} | {'Yes':8} | {'Yes':6} |")
    print(f"| Character Tracking         | {'No':5} | {configs['Enhanced']['character_consistency_window']:8} | {configs['Global']['character_consistency_window']:6} |")
    print(f"| Environment Tracking       | {'No':5} | {configs['Enhanced']['environment_consistency_window']:8} | {configs['Global']['environment_consistency_window']:6} |")
    print(f"| Budget (USD)               | ${configs['Basic']['budget_usd']:4} | ${configs['Enhanced']['budget_usd']:7} | ${configs['Global']['budget_usd']:5} |")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("• Basic: Quick prototypes, short scenes (1-10 shots)")
    print("• Enhanced: Most storyboards, balanced approach (10-50 shots)")  
    print("• Global: Long narratives, maximum consistency (50+ shots)")

if __name__ == "__main__":
    print("🎬 VC-RAG-SBG Enhanced Context System\n")
    
    demonstrate_config_differences()
    print("\n" + "="*60 + "\n")
    
    run_with_basic_context()
    print("\n" + "-"*40 + "\n")
    
    run_with_enhanced_context()
    print("\n" + "-"*40 + "\n")
    
    run_with_global_context()
    
    print("\n" + "="*60)
    print("✨ Configuration files created:")
    print("   - config_basic.yaml")
    print("   - config_enhanced.yaml") 
    print("   - config_global.yaml")
    print("\nTo use a specific configuration:")
    print("   python run.py --config config_enhanced.yaml") 