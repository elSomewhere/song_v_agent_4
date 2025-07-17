"""Enhanced Midjourney converter that creates prompts directly from rich memory context."""

from pathlib import Path
from typing import List, Dict, Any
from src.models import WorkflowState
from src.utils import (
    get_openai_client, call_openai_with_retry,
    calculate_cost, log_entry, check_budget
)
from src.memory import MemoryService

ENHANCED_MJ_SYSTEM = (
    "You are a master Midjourney v6 prompt engineer specializing in cinematic storyboard generation. "
    "Create exceptionally detailed, visually rich Midjourney prompts directly from the provided context. "
    "GENERATE COMPREHENSIVE PROMPTS (80-120 words) with:\n"
    " • Character specificity: exact appearance, canonical traits, clothing details, poses, micro-expressions\n"
    " • Environmental richness: atmospheric conditions, lighting quality, weather, time, season\n"
    " • Cinematic composition: camera angles, shot types, depth of field, focal lengths, framing\n"
    " • Visual continuity: maintain consistency with previous frames and reference images\n"
    " • Artistic excellence: photorealistic quality, film grain, color grading, contrast\n"
    " • Technical precision: 8k resolution, HDR, sharp focus, professional cinematography\n"
    " • Texture and materials: fabric weaves, surface qualities, architectural details\n"
    " • Mood and atmosphere: emotional tone, energy, dramatic tension\n"
    " • Style references: cinematographers, art movements, film genres\n"
    " • Use extensive comma-separated descriptive phrases\n"
    " • Include appropriate parameters (--ar, --stylize, --v 6)\n"
    " • Add comprehensive --no negative elements\n"
    "OUTPUT: Single masterfully detailed midjourney prompt, no explanations."
)


def enhanced_midjourney_converter_node(state: WorkflowState) -> WorkflowState:
    """Create midjourney prompts directly from rich memory context (ENHANCED VERSION)."""
    
    # Check budget first
    if not check_budget(state):
        log_entry(state, "mj_enhanced", "budget_exceeded")
        state.policy_action = "give_up"
        return state
    
    current_variation = state.variations[state.current_variation_idx]
    print(f"[MJEnhanced] Scene {current_variation.scene_id} • Shot {current_variation.shot_id} • Variation {state.current_variation_idx + 1}/{len(state.variations)} – creating direct midjourney prompt...")

    client = get_openai_client()
    memory = state.get_memory_service()
    
    # Gather rich context directly from memory
    rich_context = _gather_rich_context(state, memory, current_variation)
    
    # Build enhanced midjourney prompt from rich context
    prompt = _build_enhanced_midjourney_prompt(state, current_variation, rich_context)
    
    try:
        model = state.config["models"].get("midjourney_converter", "gpt-4o")
        response = call_openai_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": ENHANCED_MJ_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,  # Slightly higher for creative prompt generation
            max_tokens=1000   # Significantly increased for very detailed prompts
        )
        
        mj_prompt = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens
        cost = calculate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        
        state.total_tokens += tokens
        state.total_cost += cost

        # Create enhanced midjourney prompts directory
        mj_dir = Path(state.output_dir) / "prompts_midjourney_enhanced"
        mj_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename 
        filename = f"mj_enhanced_s{current_variation.scene_id}_sh{current_variation.shot_id}_v{state.current_variation_idx}.txt"
        mj_path = mj_dir / filename
        
        # Save the enhanced prompt
        with open(mj_path, "w", encoding="utf-8") as f:
            f.write(mj_prompt)

        log_entry(state, "mj_enhanced", "success",
                  model=model, cost_usd=cost,
                  extra={
                      "file": filename, 
                      "tokens": tokens,
                      "context_sources": len(rich_context.get("reference_tags", [])),
                      "prompt_length": len(mj_prompt)
                  })
        
        print(f"[MJEnhanced] Enhanced prompt saved to {filename}")
        print(f"[MJEnhanced] Used {len(rich_context.get('reference_tags', []))} reference sources → {len(mj_prompt)} char prompt")
        
    except Exception as e:
        log_entry(state, "mj_enhanced", "error", error=str(e))
        print(f"[MJEnhanced] Error: {e}")

    return state


def _gather_rich_context(state: WorkflowState, memory: MemoryService, variation) -> Dict[str, Any]:
    """Gather all available rich context for enhanced midjourney prompt creation."""
    
    # Get enhanced visual context (same as reviewer uses)
    if state.config.get("context_mode") == "enhanced" and hasattr(memory, 'get_enhanced_visual_context'):
        nearby_frames, relevant_refs, global_context = memory.get_enhanced_visual_context(
            variation.scene_id,
            variation.shot_id,
            window_size=state.config.get("ctx_window", 4)
        )
    else:
        nearby_frames, relevant_refs = memory.get_visual_context(
            variation.scene_id,
            variation.shot_id,
            window_size=state.config.get("ctx_window", 4)
        )
        global_context = {}
    
    # Get canonical entity descriptions
    canonical_entities = {}
    for entity in variation.entities:
        canonical_desc = memory.lookup_canonical(entity.name)
        if canonical_desc:
            canonical_entities[entity.name] = canonical_desc
    
    # Extract reference image tags and analysis
    reference_tags = []
    for ref in relevant_refs[:5]:  # Top 5 most relevant
        if ref and isinstance(ref, dict):
            # Handle tags properly - could be None, list, or numpy array
            tags = ref.get("tags")
            if tags is None:
                tags_list = []
            else:
                # Convert to list and slice to get top 8 tags
                try:
                    tags_list = list(tags)[:8]
                except (TypeError, ValueError):
                    tags_list = []
            
            ref_analysis = {
                "entity": ref.get("entity", "unknown"),
                "category": ref.get("category", "unknown"), 
                "tags": tags_list,
                "confidence": ref.get("confidence", 0.0)
            }
            reference_tags.append(ref_analysis)
    
    # Extract recent frame context
    frame_continuity = []
    for frame in nearby_frames[:3]:  # Last 3 frames
        if frame and isinstance(frame, dict):
            # Handle entities properly - could be None, list, or numpy array
            entities = frame.get("entities")
            if entities is None:
                entities_list = []
            else:
                # Convert to list safely
                try:
                    entities_list = list(entities)
                except (TypeError, ValueError):
                    entities_list = []
            
            frame_info = {
                "scene_id": frame.get("scene_id"),
                "shot_id": frame.get("shot_id"),
                "entities": entities_list,
                "prompt_excerpt": (frame.get("prompt") or "")[:150]  # Handle None values
            }
            frame_continuity.append(frame_info)
    
    return {
        "canonical_entities": canonical_entities,
        "reference_tags": reference_tags,
        "frame_continuity": frame_continuity,
        "global_context": global_context,
        "style_text": state.style_text,
        "scene_data": state.scenes[variation.scene_id - 1] if variation.scene_id <= len(state.scenes) else None
    }


def _build_enhanced_midjourney_prompt(state: WorkflowState, variation, rich_context: Dict[str, Any]) -> str:
    """Build comprehensive midjourney prompt creation request from rich context."""
    
    # Extract midjourney parameters
    ar_map = {"square": "1:1", "landscape": "3:2", "portrait": "2:3", "auto": "1:1"}
    aspect_ratio = ar_map.get(state.config.get("aspect_ratio", "square"), "1:1")
    
    prompt = f"""CREATE OPTIMAL MIDJOURNEY V6 PROMPT

**CORE SHOT:**
Scene {variation.scene_id}, Shot {variation.shot_id}
Shot Description: {variation.image_prompt}
Camera: {variation.camera.type} {variation.camera.angle} {variation.camera.distance}
{f"Movement: {variation.camera.movement}" if variation.camera.movement else ""}

**ENTITIES IN SHOT:**
"""
    
    # Add entity details with canonical descriptions
    for entity in variation.entities:
        canonical = rich_context["canonical_entities"].get(entity.name, "")
        description = entity.description or "unknown"
        pose = entity.pose or "unknown"
        emotion = entity.emotion or "neutral"
        
        prompt += f"- {entity.name}: {description}"
        if canonical:
            canonical_text = canonical or ""
            prompt += f" (Canonical: {canonical_text[:100]})"
        prompt += f" | Pose: {pose} | Emotion: {emotion}\n"
    
    # Add reference image intelligence
    if rich_context["reference_tags"]:
        prompt += f"\n**REFERENCE IMAGE ANALYSIS ({len(rich_context['reference_tags'])} sources):**\n"
        for ref in rich_context["reference_tags"]:
            tags_str = ", ".join(ref["tags"][:5])
            prompt += f"- {ref['entity']} ({ref['category']}): {tags_str}\n"
    
    # Add frame continuity context
    if rich_context["frame_continuity"]:
        prompt += f"\n**VISUAL CONTINUITY (last {len(rich_context['frame_continuity'])} frames):**\n"
        for frame in rich_context["frame_continuity"]:
            # Safely build entities string
            entities_list = frame.get("entities", [])
            entities_str = ", ".join(str(entity) for entity in entities_list[:3])
            prompt += f"- Scene {frame['scene_id']}.{frame['shot_id']}: {entities_str} | {frame['prompt_excerpt']}\n"
    
    # Add style and environment context
    if rich_context["style_text"]:
        style_text = rich_context["style_text"] or ""
        if style_text:  # Only add if not empty after None check
            prompt += f"\n**STYLE GUIDE:**\n{style_text[:400]}\n"
    
    if variation.style_notes:
        prompt += f"\n**SHOT-SPECIFIC STYLE:**\n{variation.style_notes}\n"
    
    # Add scene context
    if rich_context["scene_data"]:
        scene = rich_context["scene_data"]
        prompt += f"\n**SCENE CONTEXT:**\n"
        prompt += f"Location: {getattr(scene, 'location', 'Unknown')}\n"
        prompt += f"Time: {getattr(scene, 'time_of_day', 'Unknown')}\n"
        if hasattr(scene, 'narrative') and scene.narrative:
            prompt += f"Narrative: {scene.narrative[:200]}\n"
    
    # Add negative guidance
    negative_elements = []
    if state.reviewed_plan and state.reviewed_plan.negative_prompt:
        negative_elements.append(state.reviewed_plan.negative_prompt)
    
    prompt += f"""
**YOUR TASK:**
Create a single, exceptionally detailed Midjourney v6 prompt (80-120 words) that:
1. Captures ALL visual elements from the shot description with specific details
2. Integrates canonical character descriptions with precise physical traits
3. Uses reference image tags to build rich visual textures and materials
4. Maintains strong visual continuity with previous frames
5. Applies comprehensive style guide elements
6. Uses extensive comma-separated descriptive phrases
7. Includes detailed cinematography language (camera, lighting, composition)
8. Specifies surface textures, materials, and environmental details
9. Adds quality markers (8k, HDR, sharp focus, professional cinematography)
10. Ends with: --ar {aspect_ratio} --stylize 1000 --v 6
{f"11. Includes comprehensive negatives: --no {', '.join(negative_elements)}" if negative_elements else ""}

GENERATE A MASTERFULLY DETAILED PROMPT - NO EXPLANATIONS, MAXIMUM VISUAL RICHNESS."""
    
    return prompt


def _derive_mj_suffix(state: WorkflowState, variation) -> str:
    """Derive Midjourney parameters from state and variation."""
    ar_map = {"square": "1:1", "landscape": "3:2", "portrait": "2:3", "auto": "1:1"}
    ar = ar_map.get(state.config.get("aspect_ratio", "square"), "1:1")
    return f"--ar {ar} --stylize 1000 --v 6" 