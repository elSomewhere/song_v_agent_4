"""Parallel renderer for processing multiple variations simultaneously."""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from uuid import uuid4
import base64
import io

from src.models import WorkflowState, ScenePlan
from src.utils import (
    get_openai_client, call_openai_with_retry, log_entry,
    calculate_image_cost, save_base64_image, load_image_as_base64
)


def render_variations_parallel(state: WorkflowState, variations: List[ScenePlan], parallel: int = 1) -> List[Dict[str, Any]]:
    """Render multiple variations in parallel.

    If `parallel` is 1 (default) the function falls back to sequential rendering, 
    matching the spec's requirement to avoid unnecessary parallelisation unless 
    explicitly requested.
    """
    # Clamp parallel to sensible bounds
    parallel = max(1, min(parallel, 4))

    # Helper to wrap sequential execution
    def _run_seq() -> List[Dict[str, Any]]:
        results = []
        for idx, variation in enumerate(variations):
            result = _render_single_variation(
                state.config,
                variation,
                idx,
                state.style_text,
                state.output_dir,
                state.reviewed_plan.negative_prompt if state.reviewed_plan else None,
                state.reviewed_plan.visual_context if state.reviewed_plan else []
            )
            results.append(result)
            log_entry(state, "parallel_renderer", result.get("success", False) and "success" or "error",
                     extra={"variation": idx, "cost": result.get("cost", 0)})
        return results

    # If only 1 worker requested → sequential
    if parallel == 1 or len(variations) <= 1:
        return _run_seq()

    # Otherwise run with ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(parallel, len(variations))) as executor:
        futures = {
            executor.submit(
                _render_single_variation,
                state.config,
                variation,
                idx,
                state.style_text,
                state.output_dir,
                state.reviewed_plan.negative_prompt if state.reviewed_plan else None,
                state.reviewed_plan.visual_context if state.reviewed_plan else []
            ): idx for idx, variation in enumerate(variations)
        }
        results = []
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                log_entry(state, "parallel_renderer", "success",
                         extra={"variation": idx, "cost": result.get("cost", 0)})
            except Exception as e:
                log_entry(state, "parallel_renderer", "error",
                         error=str(e), extra={"variation": idx})
                results.append({
                    "variation_idx": idx,
                    "success": False,
                    "error": str(e),
                    "cost": 0
                })
        # Preserve original ordering
        results.sort(key=lambda r: r.get("variation_idx", 0))
        return results


def _render_single_variation(
    config: Dict[str, Any],
    variation: ScenePlan,
    variation_idx: int,
    style_text: str,
    output_dir: str,
    negative_prompt: Optional[str],
    visual_context: List[str]
) -> Dict[str, Any]:
    """Render a single variation - runs in separate process."""
    
    # Get OpenAI client
    import openai
    client = openai.OpenAI()
    
    # Build prompt
    prompt_parts = []
    prompt_parts.append(variation.image_prompt)
    
    if style_text:
        prompt_parts.append(f"Style: {style_text[:200]}")
    
    if variation.style_notes:
        prompt_parts.append(variation.style_notes)
    
    camera = variation.camera
    camera_desc = f"Camera: {camera.distance} shot, {camera.angle} angle"
    if camera.movement:
        camera_desc += f", {camera.movement} movement"
    prompt_parts.append(camera_desc)
    
    if negative_prompt:
        prompt_parts.append(f"Avoid: {negative_prompt}")
    
    full_prompt = " | ".join(prompt_parts)
    
    # Generate image
    model = config["models"]["renderer_new"]
    
    try:
        response = client.images.generate(
            model=model,  # Always use explicit model name from config (gpt-image-1)
            prompt=full_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="b64_json"
        )
        
        image_b64 = response.data[0].b64_json
        cost = 0.04  # Standard cost for DALL-E 3
        
        # Save image
        frame_id = str(uuid4())[:8]
        filename = f"frame_s{variation.scene_id}_sh{variation.shot_id}_v{variation_idx}_{frame_id}.png"
        image_path = Path(output_dir) / "frames" / filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Decode and save
        image_data = base64.b64decode(image_b64)
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        return {
            "variation_idx": variation_idx,
            "success": True,
            "image_b64": image_b64,
            "image_path": str(image_path),
            "filename": filename,
            "cost": cost,
            "model": model
        }
        
    except Exception as e:
        return {
            "variation_idx": variation_idx,
            "success": False,
            "error": str(e),
            "cost": 0
        }


async def render_variations_async(state: WorkflowState, variations: List[ScenePlan]) -> List[Dict[str, Any]]:
    """Async version for rendering variations - alternative approach."""
    
    tasks = []
    for idx, variation in enumerate(variations):
        task = _render_variation_async(state, variation, idx)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "variation_idx": idx,
                "success": False,
                "error": str(result)
            })
        else:
            processed_results.append(result)
    
    return processed_results


async def _render_variation_async(state: WorkflowState, variation: ScenePlan, idx: int) -> Dict[str, Any]:
    """Async render of single variation."""
    # This would need async OpenAI client implementation
    # For now, this is a placeholder
    pass 