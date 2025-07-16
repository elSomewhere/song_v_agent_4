"""Renderer node for generating images using gpt-image-1."""

import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import uuid4
import io
from datetime import datetime

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
    memory = state.get_memory_service()  # Use singleton memory service
    
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
    
    # Debug logging for reference images
    if ref_images:
        print(f"[Renderer] Found {len(ref_images)} reference images for Scene {current_variation.scene_id} Shot {current_variation.shot_id}")
        for i, ref in enumerate(ref_images):
            print(f"  - Ref {i+1}: {ref.get('frame_id', 'unknown')} ({len(ref.get('base64', '')) // 1000}KB)")
    else:
        print(f"[Renderer] No reference images found for Scene {current_variation.scene_id} Shot {current_variation.shot_id}")
    
    try:
        if is_edit:
            # Edit existing image
            result = _render_edit(client, state, current_variation, ref_images)
        else:
            # Generate new image
            result = _render_new(client, state, current_variation, ref_images)
        
        # Save generated image immediately with attempt tracking
        frame_id = str(uuid4())
        attempt_suffix = f"_r{state.retry_count}_e{state.edit_retry_count}" if (state.retry_count > 0 or state.edit_retry_count > 0) else ""
        image_filename = f"frame_s{current_variation.scene_id}_sh{current_variation.shot_id}_v{state.current_variation_idx}_{frame_id[:8]}{attempt_suffix}.png"
        image_path = Path(state.output_dir) / "frames" / image_filename
        
        # Ensure frames directory exists
        image_path.parent.mkdir(parents=True, exist_ok=True)
        save_base64_image(result["image_b64"], str(image_path))
        
        # Track this attempt for best selection and rejected image handling
        attempt_data = {
            "frame_id": frame_id,
            "image_path": str(image_path),
            "image_b64": result["image_b64"],
            "retry_count": state.retry_count,
            "edit_retry_count": state.edit_retry_count,
            "is_edit": is_edit,
            "model": result["model"],
            "timestamp": str(datetime.now()),
            "quality_score": None  # Will be set by QA
        }
        state.image_attempts.append(attempt_data)
        
        # Update current state
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
    """Generate a new image using gpt-image-1 via Responses API."""
    
    # Build the full prompt
    full_prompt = _build_image_prompt(state, variation)
    
    model = state.config["models"]["renderer_new"]
    
    # Use Responses API for gpt-image-1 with reference images
    if ref_images and model == "gpt-image-1":
        print(f"[Renderer] Using Responses API with {len(ref_images)} reference images")
        
        # Build input content with text and reference images
        input_content = [
            {"type": "input_text", "text": full_prompt}
        ]
        
        # Add reference images (up to 4)
        for ref in ref_images[:4]:
            if ref.get("base64"):
                input_content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{ref['base64']}"
                })
        
        print(f"[Renderer] Sending {len(input_content)-1} images to gpt-image-1 via Responses API")
        
        # Use standard chat completions API with image generation model
        response = call_openai_with_retry(
            client,
            model=model,  # Use gpt-image-1 directly
            messages=[{
                "role": "user", 
                "content": input_content
            }],
            max_tokens=1000
        )
        
        # Extract image from standard response format
        content = response.choices[0].message.content
        
        # For gpt-image-1, the content should contain the base64 image
        # or we may need to extract it differently based on actual API
        if content and "data:image" in content:
            # Extract base64 from data URL
            import re
            match = re.search(r'data:image/[^;]+;base64,([^"]+)', content)
            if match:
                image_b64 = match.group(1)
                cost = 0.08  # Approximate cost for gpt-image-1 with references
            else:
                raise Exception("No valid image data found in response")
        else:
            # Assume the entire content is base64 encoded image
            image_b64 = content
            cost = 0.08
            
    elif model == "gpt-image-1":
        print(f"[Renderer] Using Responses API without reference images")
        
        # Use standard chat completions API without reference images  
        response = call_openai_with_retry(
            client,
            model=model,
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": full_prompt}]
            }],
            max_tokens=1000
        )
        
        # Extract image from standard response format
        content = response.choices[0].message.content
        
        # For gpt-image-1, extract image data
        if content and "data:image" in content:
            import re
            match = re.search(r'data:image/[^;]+;base64,([^"]+)', content)
            if match:
                image_b64 = match.group(1)
                cost = 0.04  # Standard cost for gpt-image-1
            else:
                raise Exception("No valid image data found in response")
        else:
            # Assume the entire content is base64 encoded image
            image_b64 = content
            cost = 0.04
    else:
        # Standard generation for other models (dall-e-3, etc.)
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
    """Edit an existing image using gpt-image-1 via Responses API."""
    
    model = state.config["models"]["renderer_edit"]
    
    # Build edit instruction
    edit_instruction = _build_edit_instruction(state, variation)
    
    if model == "gpt-image-1":
        # Use Responses API for editing with gpt-image-1
        input_content = [
            {"type": "input_text", "text": edit_instruction}
        ]
        
        # Add current image to be edited as first input image
        if state.current_image_b64:
            input_content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{state.current_image_b64}"
            })
        
        # Add reference images for context (up to 3 additional)
        for ref in ref_images[:3]:
            if ref.get("base64"):
                input_content.append({
                    "type": "input_image", 
                    "image_url": f"data:image/png;base64,{ref['base64']}"
                })
        
        # Use standard chat completions API for editing
        response = call_openai_with_retry(
            client,
            model=model,
            messages=[{
                "role": "user",
                "content": input_content
            }],
            max_tokens=1000
        )
        
        # Extract image from standard response format
        content = response.choices[0].message.content
        
        # For gpt-image-1, extract image data
        if content and "data:image" in content:
            import re
            match = re.search(r'data:image/[^;]+;base64,([^"]+)', content)
            if match:
                image_b64 = match.group(1)
                cost = 0.04  # Approximate cost for gpt-image-1 edit
            else:
                raise Exception("No valid image data found in edit response")
        else:
            # Assume the entire content is base64 encoded image
            image_b64 = content
            cost = 0.04
    else:
        # Fallback for other models (though edit mainly uses gpt-image-1)
        # Convert current image to file-like object
        current_image_data = base64.b64decode(state.current_image_b64)
        current_image_file = io.BytesIO(current_image_data)
        
        # Use standard edit API
        response = call_openai_with_retry(
            client,
            model=model,
            image=current_image_file,
            prompt=edit_instruction
        )
        
        image_b64 = response.data[0].b64_json
        cost = 0.04  # Approximate cost for edit
    
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
        print(f"[Renderer] Searching for {len(state.reviewed_plan.visual_context)} reference IDs: {state.reviewed_plan.visual_context}")
        
        for ref_id in state.reviewed_plan.visual_context[:3]:
            # Search for this reference in memory
            results = memory.search_references(ref_id, limit=1)
            print(f"[Renderer] Search for {ref_id}: {len(results) if results else 0} results")
            
            if results and results[0].get("original_path"):
                try:
                    ref_images.append({
                        "frame_id": ref_id,
                        "base64": load_image_as_base64(results[0]["original_path"])
                    })
                    print(f"[Renderer] Successfully loaded reference image {ref_id}")
                except Exception as e:
                    print(f"[Renderer] Failed to load reference image {ref_id}: {e}")
            else:
                print(f"[Renderer] No valid path found for reference {ref_id}")
    else:
        if not state.reviewed_plan:
            print(f"[Renderer] No reviewed_plan found")
        elif not state.reviewed_plan.visual_context:
            print(f"[Renderer] No visual_context in reviewed_plan")
    
    # Fallback: if no references found, try to get some based on scene entities
    if not ref_images:
        print(f"[Renderer] No references from reviewed plan, trying fallback search")
        
        # Get current scene
        current_scene_idx = variation.scene_id - 1
        if current_scene_idx < len(state.scenes):
            current_scene = state.scenes[current_scene_idx]
            
            # Try to search for references based on scene entities
            if current_scene.entities:
                print(f"[Renderer] Searching for entities: {current_scene.entities}")
                for entity in current_scene.entities[:2]:  # Limit to first 2 entities
                    refs = memory.search_references(entity, entity_filter=None, limit=1)
                    if refs and refs[0].get("original_path"):
                        try:
                            ref_images.append({
                                "frame_id": refs[0].get("frame_id", "fallback"),
                                "base64": load_image_as_base64(refs[0]["original_path"])
                            })
                            print(f"[Renderer] Loaded fallback reference for entity '{entity}'")
                        except Exception as e:
                            print(f"[Renderer] Failed to load fallback reference for '{entity}': {e}")
            
            # Note: No longer using "any available" fallback to avoid unrelated images
    
    print(f"[Renderer] Final reference count: {len(ref_images)} images")
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