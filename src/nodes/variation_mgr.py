"""Variation Manager node for creating multiple camera angle variations."""

import json
import copy
from typing import Dict, Any, List

from src.models import WorkflowState, ScenePlan, Camera
from src.utils import (
    get_openai_client, call_openai_with_retry, log_entry,
    calculate_cost, parse_json_response, check_budget
)


def variation_mgr_node(state: WorkflowState) -> WorkflowState:
    """Create variations of the reviewed plan with different camera angles."""
    client = get_openai_client()
    
    # Check budget
    if not check_budget(state):
        log_entry(state, "variation_mgr", "budget_exceeded")
        state.policy_action = "give_up"
        return state
    
    # Check if we have a reviewed plan
    if not state.reviewed_plan:
        log_entry(state, "variation_mgr", "no_reviewed_plan")
        return state
    
    base_plan = state.reviewed_plan.approved_plan
    n_variations = state.n_variations
    variation_type = state.config.get("variation_camera", "full")
    
    # Generate variations
    variations = [base_plan]  # Include original as first variation
    
    if n_variations > 1:
        # Get additional variations from GPT
        additional_variations = _generate_variations(
            client, state, base_plan, 
            n_variations - 1, 
            variation_type
        )
        variations.extend(additional_variations)
    
    state.variations = variations
    state.current_variation_idx = 0
    
    log_entry(state, "variation_mgr", "success",
             extra={"variations_created": len(variations), "type": variation_type})
    
    return state


def _generate_variations(client: Any, state: WorkflowState, base_plan: ScenePlan,
                        count: int, variation_type: str) -> List[ScenePlan]:
    """Generate camera variations using GPT-4o."""
    
    prompt = f"""Create {count} variations of this storyboard shot with different camera configurations.

Base Shot:
{json.dumps(base_plan.model_dump(), indent=2)}

Variation Type: {variation_type}

Guidelines:
- Keep the same entities and scene content
- Vary camera angles, distances, and movements based on variation type
- Each variation should offer a distinct visual perspective
- Consider cinematographic principles

For variation_type:
- "full": Vary all camera parameters significantly
- "angle_only": Keep distance same, vary angle and type
- "composition_only": Keep camera same, adjust entity positions/poses

Return JSON array of {count} variations, each with:
{{
    "camera": {{
        "type": "static/tracking/pan/etc",
        "angle": "low/high/eye-level/dutch/etc", 
        "distance": "close-up/medium/wide/full",
        "movement": "zoom-in/pan-left/etc or null"
    }},
    "image_prompt": "adjusted prompt for this camera angle",
    "variation_notes": "what makes this variation unique"
}}"""
    
    try:
        model = state.config["models"]["variation_mgr"]
        response = call_openai_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": "You are a cinematographer creating dynamic camera variations for storyboards."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        cost = calculate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        
        state.total_tokens += tokens
        state.total_cost += cost
        
        # Parse variations
        variations_data = parse_json_response(content)
        if not isinstance(variations_data, list):
            variations_data = variations_data.get("variations", [])
        
        # Create ScenePlan variations
        variations = []
        for i, var_data in enumerate(variations_data[:count]):
            # Copy base plan
            variation = copy.deepcopy(base_plan)
            
            # Update camera
            if "camera" in var_data:
                variation.camera = Camera(**var_data["camera"])
            
            # Update prompt if provided
            if "image_prompt" in var_data:
                variation.image_prompt = var_data["image_prompt"]
            
            # Add variation notes to style notes
            if "variation_notes" in var_data:
                notes = var_data["variation_notes"]
                if variation.style_notes:
                    variation.style_notes += f" | Variation: {notes}"
                else:
                    variation.style_notes = f"Variation: {notes}"
            
            variations.append(variation)
        
        log_entry(state, "variation_mgr", "generated",
                 model=model, tokens=tokens, cost_usd=cost,
                 extra={"count": len(variations)})
        
        return variations
        
    except Exception as e:
        log_entry(state, "variation_mgr", "error", error=str(e))
        # Return empty list on error
        return [] 