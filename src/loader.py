"""Loader module for validating inputs and initializing the workflow state."""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.models import WorkflowState
from src.utils import ensure_directory, format_timestamp, log_entry


class Loader:
    """Validates inputs and initializes workflow state."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize loader with configuration."""
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_inputs(self, script_path: str, style_path: str, 
                       entities_path: str, refs_dir: Optional[str] = None) -> None:
        """Validate that all required input files exist."""
        # Check required files
        for path, name in [(script_path, "script"), 
                          (style_path, "style"), 
                          (entities_path, "entities")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} file not found: {path}")
            
            # Check file extension
            if not path.endswith('.md'):
                raise ValueError(f"{name} file must be markdown: {path}")
        
        # Check optional refs directory
        if refs_dir:
            refs_path = Path(refs_dir)
            if not refs_path.exists():
                raise FileNotFoundError(f"refs directory not found: {refs_dir}")
            
            # Check for valid image files
            valid_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
            image_files = [f for f in refs_path.iterdir() 
                          if f.suffix.lower() in valid_extensions]
            
            if not image_files:
                print(f"Warning: No valid image files found in {refs_dir}")
    
    def create_output_directory(self, base_output_dir: str = "output") -> str:
        """Create output directory with timestamp."""
        timestamp = format_timestamp()
        run_dir = f"run_{timestamp}"
        output_dir = Path(base_output_dir) / run_dir
        
        # Create directory structure
        ensure_directory(str(output_dir))
        ensure_directory(str(output_dir / "frames"))
        ensure_directory(str(output_dir / "variations"))
        ensure_directory(str(output_dir / "memory"))
        
        # Create cache directory for thumbnails
        cache_dir = Path(".cache") / "thumbs"
        ensure_directory(str(cache_dir))
        
        return str(output_dir)
    
    def load_input_files(self, script_path: str, style_path: str, 
                        entities_path: str) -> Dict[str, Any]:
        """Load content from input files."""
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        with open(style_path, 'r', encoding='utf-8') as f:
            style_content = f.read()
        
        with open(entities_path, 'r', encoding='utf-8') as f:
            entities_content = f.read()
        
        # Parse entities as JSON if possible
        entities_dict = {}
        try:
            # Try to extract JSON from markdown
            json_start = entities_content.find('{')
            json_end = entities_content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                entities_json = entities_content[json_start:json_end]
                entities_dict = json.loads(entities_json)
        except:
            # If JSON parsing fails, keep as plain text
            pass
        
        return {
            "script": script_content,
            "style": style_content,
            "entities": entities_content,
            "entities_dict": entities_dict
        }
    
    def initialize_state(self, script_path: str, style_path: str,
                        entities_path: str, refs_dir: Optional[str] = None,
                        config_overrides: Optional[Dict[str, Any]] = None) -> WorkflowState:
        """Initialize workflow state with inputs and configuration."""
        # Validate inputs
        self.validate_inputs(script_path, style_path, entities_path, refs_dir)
        
        # Create output directory
        output_dir = self.create_output_directory()
        
        # Load input files
        inputs = self.load_input_files(script_path, style_path, entities_path)
        
        # Merge config with overrides
        config = self.config.copy()
        if config_overrides:
            config.update(config_overrides)
        
        # Create initial state
        state = WorkflowState(
            script_path=script_path,
            style_path=style_path,
            entities_path=entities_path,
            refs_dir=refs_dir,
            output_dir=output_dir,
            style_text=inputs["style"],
            entities_dict=inputs["entities_dict"],
            config=config,
            budget_usd=config.get("budget_usd", 35.0),
            n_variations=config.get("n_variations", 3),
            max_retries=config.get("max_retries", 2),
            max_edit_retries=config.get("max_edit_retries", 1)
        )
        
        # Log initialization (temporarily disable to avoid circular dependency)
        # log_entry(state, "loader", "success", 
        #          extra={"output_dir": output_dir, "config": config})
        
        # Save initial files to output directory
        self._save_initial_files(state, inputs)
        
        return state
    
    def _save_initial_files(self, state: WorkflowState, inputs: Dict[str, Any]) -> None:
        """Save input files and config to output directory."""
        output_path = Path(state.output_dir)
        
        # Save config
        with open(output_path / "config.yaml", 'w') as f:
            yaml.dump(state.config, f)
        
        # Save raw inputs
        with open(output_path / "script.md", 'w') as f:
            f.write(inputs["script"])
        
        with open(output_path / "style.md", 'w') as f:
            f.write(inputs["style"])
        
        with open(output_path / "entities.md", 'w') as f:
            f.write(inputs["entities"])
        
        # Initialize empty logs.jsonl
        with open(output_path / "logs.jsonl", 'w') as f:
            pass 