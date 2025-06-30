"""Renderer node for generating images using gpt-image-1."""

import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import uuid4
import io

from src.models import WorkflowState
from src.utils import (
    get_openai_client, call_openai_with_retry, log_entry,
    calculate_image_cost, check_budget, save_base64_image,
    load_image_as_base64
)
from src.memory import MemoryService


def renderer_node(state: WorkflowState) -> WorkflowState:
    """Render current variation using gpt-image-1 (NEVER dall-e-3)."""
    client = get_openai_client()
    memory = MemoryService(state)
    
    # Check budget
    if not check_budget(state):
        log_entry(state, "renderer", "budget_exceeded")
        state.policy_action = "give_up"
        return state
    
    # Get current variation
    if not state.variations or state.current_variation_idx >= len(state.variations):
        log_entry(state, "renderer", "no_variation")
        return state
    
    print(f"[Renderer] Scene {state.variations[state.current_variation_idx].scene_id} • Shot {state.variations[state.current_variation_idx].shot_id} • Variation {state.current_variation_idx + 1}/{len(state.variations)} – rendering image ...")
    
    current_variation = state.variations[state.current_variation_idx]
    
    # Determine if this is a retry with edit
    is_edit = (state.policy_action == "retry_edit" and 
               state.current_image_b64 is not None and
               state.edit_retry_count < state.max_edit_retries)
    
    # Get reference images for context
    ref_images = _get_reference_images(state, memory, current_variation)
    
    try:
        if is_edit:
            # Edit existing image
            result = _render_edit(client, state, current_variation, ref_images)
        else:
            # Generate new image
            result = _render_new(client, state, current_variation, ref_images)
        
        # Save generated image
        frame_id = str(uuid4())
        image_filename = f"frame_s{current_variation.scene_id}_sh{current_variation.shot_id}_v{state.current_variation_idx}_{frame_id[:8]}.png"
        image_path = Path(state.output_dir) / "frames" / image_filename
        
        save_base64_image(result["image_b64"], str(image_path))
        
        # Update state
        state.current_image_b64 = result["image_b64"]
        state.current_image_path = str(image_path)
        
        # Update memory
        memory.update_episodic_memory({
            "stage": "renderer",
            "scene_id": current_variation.scene_id,
            "shot_id": current_variation.shot_id,
            "variation": state.current_variation_idx,
            "is_edit": is_edit,
            "model": result["model"]
        })
        
        # Build the full prompt
        full_prompt = _build_image_prompt(state, current_variation)

        # Save prompt to disk for user inspection (one .txt per frame id later)
        prompt_dump_dir = Path(state.output_dir) / "prompts"
        prompt_dump_dir.mkdir(parents=True, exist_ok=True)
        prompt_temp_file = prompt_dump_dir / f"prompt_s{current_variation.scene_id}_sh{current_variation.shot_id}_v{state.current_variation_idx}.txt"
        try:
            with open(prompt_temp_file, "w", encoding="utf-8") as pf:
                pf.write(full_prompt)
        except Exception:
            pass  # Non-fatal
        
        log_entry(state, "renderer", "success",
                 model=result["model"], cost_usd=result["cost"],
                 extra={
                     "is_edit": is_edit,
                     "image_path": image_filename,
                     "prompt_file": str(prompt_temp_file),
                 })
        
    except Exception as e:
        log_entry(state, "renderer", "error", error=str(e))
        state.policy_action = "give_up"
    
    return state


def _render_new(client: Any, state: WorkflowState, variation: Any,
               ref_images: List[Dict]) -> Dict[str, Any]:
    """Generate a new image using gpt-image-1."""
    
    # Build the full prompt
    full_prompt = _build_image_prompt(state, variation)
    
    model = state.config["models"]["renderer_new"]
    
    # Check if we have reference images and should use edit API
    if ref_images and model == "gpt-image-1":
        # Use edit endpoint with reference images for context
        # The edit API can accept multiple images as references
        
        # Prepare reference images as file-like objects
        image_files = []
        for ref in ref_images[:4]:  # Max 4 reference images
            if ref.get("base64"):
                image_data = base64.b64decode(ref["base64"])
                image_file = io.BytesIO(image_data)
                image_files.append(image_file)
        
        if image_files:
            # Use edit API with reference images
            response = call_openai_with_retry(
                client,
                model=model,
                image=image_files,  # Reference images
                prompt=full_prompt
            )
            
            image_b64 = response.data[0].b64_json
            cost = 0.08  # Approximate cost for gpt-image-1 with references
        else:
            # Generate without references
            response = call_openai_with_retry(
                client,
                model=model,
                prompt=full_prompt,
                size="1024x1024",
                quality="medium"
            )
            
            image_b64 = response.data[0].b64_json
            cost = calculate_image_cost(model, "1024x1024", "medium")
    else:
        # Standard generation without references
        response = call_openai_with_retry(
            client,
            model=model,
            prompt=full_prompt,
            size="1024x1024",
            quality="medium"
        )
        
        image_b64 = response.data[0].b64_json
        cost = calculate_image_cost(model, "1024x1024", "medium")
    
    state.total_cost += cost
    
    return {
        "image_b64": image_b64,
        "model": model,
        "cost": cost
    }


def _render_edit(client: Any, state: WorkflowState, variation: Any,
                ref_images: List[Dict]) -> Dict[str, Any]:
    """Edit an existing image using gpt-image-1 edit API."""
    
    model = state.config["models"]["renderer_edit"]
    
    # Build edit instruction
    edit_instruction = _build_edit_instruction(state, variation)
    
    # Convert current image to file-like object
    current_image_data = base64.b64decode(state.current_image_b64)
    current_image_file = io.BytesIO(current_image_data)
    
    # Use gpt-image-1 edit
    response = call_openai_with_retry(
        client,
        model=model,
        image=current_image_file,
        prompt=edit_instruction
    )
    
    image_b64 = response.data[0].b64_json
    cost = 0.04  # Approximate cost for gpt-image-1 edit
    
    state.total_cost += cost
    state.edit_retry_count += 1
    
    return {
        "image_b64": image_b64,
        "model": model,
        "cost": cost
    }


def _get_reference_images(state: WorkflowState, memory: MemoryService, 
                         variation: Any) -> List[Dict[str, Any]]:
    """Get reference images for the current shot."""
    ref_images = []
    
    # Get reference images from reviewed plan context
    if state.reviewed_plan and state.reviewed_plan.visual_context:
        for ref_id in state.reviewed_plan.visual_context[:3]:
            # Search for this reference in memory
            results = memory.search_references(ref_id, limit=1)
            if results and results[0].get("original_path"):
                try:
                    ref_images.append({
                        "frame_id": ref_id,
                        "base64": load_image_as_base64(results[0]["original_path"])
                    })
                except:
                    pass
    
    return ref_images


def _build_image_prompt(state: WorkflowState, variation: Any) -> str:
    """Build the complete image generation prompt."""
    # ------------------------------------------------------------------
    # Standardised multi-section prompt for the image model
    # ------------------------------------------------------------------

    sections: List[str] = []

    # 1) Global narrative / design context
    if state.static_summary:
        sections.append("<CONTEXT>\n" + state.static_summary.strip() + "\n</CONTEXT>")

    # 2) Art-style rules (short excerpt + any shot-specific notes)
    style_block_parts: List[str] = []
    if state.style_text:
        style_block_parts.append(state.style_text[:400].strip())
    if variation.style_notes:
        style_block_parts.append(variation.style_notes.strip())
    if style_block_parts:
        sections.append("<STYLE_GUIDE>\n" + "\n".join(style_block_parts) + "\n</STYLE_GUIDE>")

    # 3) The actual shot description (what to draw)
    sections.append("<SHOT_PROMPT>\n" + variation.image_prompt.strip() + "\n</SHOT_PROMPT>")

    # 4) Camera metadata
    cam = variation.camera
    cam_desc = f"type={cam.type}; angle={cam.angle}; distance={cam.distance}"
    if cam.movement:
        cam_desc += f"; movement={cam.movement}"
    sections.append("<CAMERA>\n" + cam_desc + "\n</CAMERA>")

    # 5) Negative prompt / avoid list
    if state.reviewed_plan and state.reviewed_plan.negative_prompt:
        sections.append("<NEGATIVE>\n" + state.reviewed_plan.negative_prompt.strip() + "\n</NEGATIVE>")

    # Final prompt string (double line breaks between blocks for clarity)
    return "\n\n".join(sections)


def _build_edit_instruction(state: WorkflowState, variation: Any) -> str:
    """Build instruction for editing an existing image."""
    
    # Get retry guidance from QA result
    guidance = ""
    if state.fast_qa_result and state.fast_qa_result.retry_guidance:
        guidance = state.fast_qa_result.retry_guidance
    elif state.vision_qa_result and state.vision_qa_result.retry_guidance:
        guidance = state.vision_qa_result.retry_guidance
    
    instruction = f"Edit this image to improve quality. {guidance}"
    
    # Add specific issues to address
    issues = []
    if state.fast_qa_result:
        issues.extend(state.fast_qa_result.specific_issues)
    if state.vision_qa_result:
        issues.extend(state.vision_qa_result.specific_issues)
    
    if issues:
        instruction += f" Fix these issues: {', '.join(issues[:3])}"
    
    return instruction 