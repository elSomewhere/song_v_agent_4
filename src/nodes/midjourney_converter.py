"""Convert a raw prompt into an optimised Midjourney v6 prompt."""

from pathlib import Path
from src.models import WorkflowState
from src.utils import (
    get_openai_client, call_openai_with_retry,
    calculate_cost, log_entry, check_budget
)

MJ_SYSTEM = (
    "You are an expert Midjourney v6 prompt engineer with deep knowledge of visual storytelling. "
    "Transform the input into a rich, detailed Midjourney prompt that maximizes visual impact. "
    "CREATE DETAILED PROMPTS (60-100 words) with:\n"
    " • Specific subject details: character appearance, clothing, poses, expressions\n"
    " • Rich environmental context: lighting, atmosphere, weather, time of day\n"
    " • Camera and composition: shot type, angle, depth of field, focal length\n"
    " • Artistic style: photorealistic, cinematic, artistic movement references\n"
    " • Color palette and mood: specific colors, contrast, saturation\n"
    " • Technical quality markers: 8k, HDR, sharp focus, professional photography\n"
    " • Surface textures and materials: fabric, metal, skin, architectural elements\n"
    " • Use comma-separated descriptive phrases for maximum impact\n"
    " • End with appropriate MJ parameters (--ar, --stylize, --v 6)\n"
    " • Include --no for negative elements when relevant\n"
    "OUTPUT: Single comprehensive midjourney prompt, no markdown formatting."
)


def _derive_mj_suffix(state: WorkflowState, variation) -> str:
    """Derive Midjourney parameters from state and variation."""
    # Map aspect ratios to Midjourney format
    ar_map = {
        "square": "1:1", 
        "landscape": "3:2", 
        "portrait": "2:3", 
        "auto": "1:1"
    }
    ar = ar_map.get(state.config.get("aspect_ratio", "square"), "1:1")
    
    # Default Midjourney parameters - can be made configurable later
    return f"--ar {ar} --stylize 1000 --v 6"


def midjourney_converter_node(state: WorkflowState) -> WorkflowState:
    """Convert raw prompts to midjourney-optimized prompts using GPT-4o."""
    
    # Check budget first
    if not check_budget(state):
        log_entry(state, "mj_convert", "budget_exceeded")
        state.policy_action = "give_up"
        return state
    
    # Check if we have a raw prompt to convert
    if not getattr(state, "current_raw_prompt_path", None):
        log_entry(state, "mj_convert", "no_raw_prompt")
        return state

    current_variation = state.variations[state.current_variation_idx]
    print(f"[MidjourneyConverter] Scene {current_variation.scene_id} • Shot {current_variation.shot_id} • Variation {state.current_variation_idx + 1}/{len(state.variations)} – converting to Midjourney prompt...")

    client = get_openai_client()
    
    # Read the raw prompt
    try:
        with open(state.current_raw_prompt_path, "r", encoding="utf-8") as f:
            raw_prompt = f.read()
    except Exception as e:
        log_entry(state, "mj_convert", "error", error=f"Failed to read raw prompt: {e}")
        return state

    # Build conversion prompt
    suffix = _derive_mj_suffix(state, current_variation)
    user_msg = f"""TRANSFORM INTO DETAILED MIDJOURNEY PROMPT:

{raw_prompt}

CREATE a comprehensive 60-100 word Midjourney prompt that includes:
• Rich character/subject details (appearance, clothing, pose, expression)
• Detailed environment (lighting, atmosphere, setting, weather) 
• Camera work (shot type, angle, composition, depth of field)
• Artistic style (photorealistic, cinematic, art movement references)
• Color palette and mood descriptors
• Technical quality terms (8k, HDR, sharp focus, professional)
• Surface textures and materials
• Comma-separated descriptive phrases

End with: `{suffix}`"""

    try:
        model = state.config["models"].get("midjourney_converter", "gpt-4o")
        response = call_openai_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": MJ_SYSTEM},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.4,
            max_tokens=800  # Increased for longer, more detailed prompts
        )
        
        mj_prompt = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        cost = calculate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        
        state.total_tokens += tokens
        state.total_cost += cost

        # Create midjourney prompts directory
        mj_dir = Path(state.output_dir) / "prompts_midjourney"
        mj_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename (convert prompt_s1_sh1_v0.txt to prompt_mj_s1_sh1_v0.txt)
        original_filename = Path(state.current_raw_prompt_path).name
        mj_filename = original_filename.replace("prompt_", "prompt_mj_")
        mj_path = mj_dir / mj_filename
        
        # Save the converted prompt
        with open(mj_path, "w", encoding="utf-8") as f:
            f.write(mj_prompt)

        log_entry(state, "mj_convert", "success",
                  model=model, cost_usd=cost,
                  extra={
                      "file": mj_filename, 
                      "tokens": tokens,
                      "original_length": len(raw_prompt),
                      "converted_length": len(mj_prompt)
                  })
        
        print(f"[MidjourneyConverter] Converted prompt saved to {mj_filename}")
        print(f"[MidjourneyConverter] Original: {len(raw_prompt)} chars → Midjourney: {len(mj_prompt)} chars")
        
    except Exception as e:
        log_entry(state, "mj_convert", "error", error=str(e))
        print(f"[MidjourneyConverter] Error: {e}")

    return state 