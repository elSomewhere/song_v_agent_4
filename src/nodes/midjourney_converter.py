"""Convert a raw prompt into an optimised Midjourney v6 prompt."""

from pathlib import Path
from src.models import WorkflowState
from src.utils import (
    get_openai_client, call_openai_with_retry,
    calculate_cost, log_entry, check_budget
)

MJ_SYSTEM = (
    "You are an expert Midjourney prompt engineer (version 6, style tuning α). "
    "Rewrite the input so that it works optimally in Midjourney: "
    " • keep key subject words and style notes\n"
    " • compress long prose to short, evocative phrases separated by commas\n"
    " • append valid MJ parameters (e.g. --ar, --stylize, --quality) derived "
    "   from shot metadata\n"
    " • move negative content behind `--no`\n"
    " • never include markdown fences or JSON – return *one* plain‑text prompt."
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
    user_msg = f"{raw_prompt}\n\n---\nAdd appropriate Midjourney parameters. End with `{suffix}`."

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
            max_tokens=400
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