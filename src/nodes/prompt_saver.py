"""Node that builds the raw prompt and saves it – no image generation."""

from pathlib import Path
from src.models import WorkflowState
from src.prompt_builder import build_raw_prompt
from src.utils import log_entry


def prompt_saver_node(state: WorkflowState) -> WorkflowState:
    """Build and save raw prompt without image generation for midjourney mode."""
    
    # Check if we have variations to process
    if not state.variations or state.current_variation_idx >= len(state.variations):
        log_entry(state, "prompt_saver", "no_variation")
        return state

    print(f"[PromptSaver] Scene {state.variations[state.current_variation_idx].scene_id} • Shot {state.variations[state.current_variation_idx].shot_id} • Variation {state.current_variation_idx + 1}/{len(state.variations)} – saving prompt...")

    current_variation = state.variations[state.current_variation_idx]
    
    # Build the full prompt using the extracted prompt builder
    full_prompt = build_raw_prompt(state, current_variation)

    # Create prompts folder structure
    prompt_dir = Path(state.output_dir) / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename matching the existing convention
    filename = f"prompt_s{current_variation.scene_id}_sh{current_variation.shot_id}_v{state.current_variation_idx}.txt"
    prompt_path = prompt_dir / filename
    
    try:
        # Save the prompt to disk
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(full_prompt)
        
        # Store the path on state so midjourney converter can find it
        state.current_raw_prompt_path = str(prompt_path)
        
        log_entry(state, "prompt_saver", "success", 
                 extra={"file": filename, "prompt_length": len(full_prompt)})
        
        # Mark policy_action so workflow_controller advances
        state.policy_action = "accept"
        
        print(f"[PromptSaver] Prompt saved to {filename} ({len(full_prompt)} characters)")
        
    except Exception as e:
        log_entry(state, "prompt_saver", "error", error=str(e))
        state.policy_action = "give_up"
    
    return state 