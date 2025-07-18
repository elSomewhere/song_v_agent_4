"""Enhanced Midjourney converter that creates entity-consistent prompts with strong composition focus."""

from pathlib import Path
from typing import List, Dict, Any
from src.models import WorkflowState
from src.utils import (
    get_openai_client, call_openai_with_retry,
    calculate_cost, log_entry, check_budget
)
from src.memory import MemoryService

ENHANCED_MJ_SYSTEM = (
    "You are a master Midjourney v6 prompt engineer specializing in consistent character storyboard generation. "
    "Create exceptionally detailed, entity-consistent Midjourney prompts that maintain character accuracy across shots. "
    "GENERATE COMPREHENSIVE PROMPTS (120-180 words) with:\n"
    " • Entity consistency: use exact canonical descriptions for all characters/objects, maintain identical physical traits\n"
    " • Character specificity: precise appearance, clothing details, poses, micro-expressions from canonical descriptions\n"
    " • Composition mastery: sophisticated camera angles, rule of thirds, leading lines, depth layers, focal hierarchy\n"
    " • Environmental richness: atmospheric conditions, lighting quality, weather, time, season, spatial relationships\n"
    " • Cinematic excellence: professional shot types, depth of field, focal lengths, framing, perspective\n"
    " • Style-driven rendering: analyze style guide to determine visual approach (photorealistic, illustrated, animated, painterly, etc.)\n"
    " • Technical quality: appropriate resolution and detail level matching the derived style aesthetic\n"
    " • Material authenticity: surface qualities, architectural details, environmental textures suited to chosen style\n"
    " • Mood integration: emotional tone matching scene context and character states\n"
    " • Style adaptation: derive appropriate stylistic parameters and visual approach from style guide elements\n"
    " • Use extensive comma-separated descriptive phrases for maximum detail\n"
    " • Include flexible parameters based on style guide analysis\n"
    " • Add comprehensive --no negative elements\n"
    "OUTPUT: Single masterfully detailed midjourney prompt with consistent entities, no explanations."
)


def enhanced_midjourney_converter_node(state: WorkflowState) -> WorkflowState:
    """Create midjourney prompts with consistent entity descriptions and strong composition focus."""
    
    # Check budget first
    if not check_budget(state):
        log_entry(state, "mj_enhanced", "budget_exceeded")
        state.policy_action = "give_up"
        return state
    
    current_variation = state.variations[state.current_variation_idx]
    print(f"[MJEnhanced] Scene {current_variation.scene_id} • Shot {current_variation.shot_id} • Variation {state.current_variation_idx + 1}/{len(state.variations)} – creating entity-consistent midjourney prompt...")

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
            temperature=0.1,  # Lower temperature for more consistent entity descriptions
            max_tokens=1200   # Increased for longer detailed prompts (100-150 words)
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
                      "entities_count": len(rich_context.get("canonical_entities", {})),
                      "reference_sources": len(rich_context.get("reference_tags", [])),
                      "prompt_length": len(mj_prompt)
                  })
        
        print(f"[MJEnhanced] Enhanced prompt saved to {filename}")
        print(f"[MJEnhanced] Used {len(rich_context.get('canonical_entities', {}))} entities + {len(rich_context.get('reference_tags', []))} refs → {len(mj_prompt)} char prompt")
        
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
    
    # Get canonical entity descriptions - this is crucial for consistency
    canonical_entities = {}
    for entity in variation.entities:
        canonical_desc = memory.lookup_canonical(entity.name)
        if canonical_desc:
            canonical_entities[entity.name] = canonical_desc
    
    # Extract reference image tags and analysis for entity appearance details
    reference_tags = []
    for ref in relevant_refs[:8]:  # Increased for more entity reference data
        if ref and isinstance(ref, dict):
            # Handle tags properly - could be None, list, or numpy array
            tags = ref.get("tags")
            if tags is None:
                tags_list = []
            else:
                # Convert to list and slice to get top 12 tags for more detail
                try:
                    tags_list = list(tags)[:12]
                except (TypeError, ValueError):
                    tags_list = []
            
            ref_analysis = {
                "entity": ref.get("entity", "unknown"),
                "category": ref.get("category", "unknown"), 
                "tags": tags_list,
                "confidence": ref.get("confidence", 0.0)
            }
            reference_tags.append(ref_analysis)
    
    return {
        "canonical_entities": canonical_entities,
        "reference_tags": reference_tags,
        "global_context": global_context,
        "style_text": state.style_text,
        "scene_data": state.scenes[variation.scene_id - 1] if variation.scene_id <= len(state.scenes) else None
    }


def _build_enhanced_midjourney_prompt(state: WorkflowState, variation, rich_context: Dict[str, Any]) -> str:
    """Build comprehensive midjourney prompt creation request emphasizing entity consistency and composition."""
    
    # Extract base aspect ratio for flexible parameter derivation
    ar_map = {"square": "1:1", "landscape": "3:2", "portrait": "2:3", "auto": "1:1"}
    aspect_ratio = ar_map.get(state.config.get("aspect_ratio", "square"), "1:1")
    
    prompt = f"""CREATE OPTIMAL MIDJOURNEY V6 PROMPT WITH ENTITY CONSISTENCY

**CORE SHOT:**
Scene {variation.scene_id}, Shot {variation.shot_id}
Shot Description: {variation.image_prompt}
Camera: {variation.camera.type} {variation.camera.angle} {variation.camera.distance}
{f"Movement: {variation.camera.movement}" if variation.camera.movement else ""}

**ENTITIES IN SHOT (MAINTAIN EXACT CONSISTENCY):**
"""
    
    # Enhanced entity details with canonical descriptions for consistency
    for entity in variation.entities:
        canonical = rich_context["canonical_entities"].get(entity.name, "")
        description = entity.description or "unknown"
        pose = entity.pose or "unknown"
        emotion = entity.emotion or "neutral"
        
        prompt += f"- {entity.name}: {description}"
        if canonical:
            canonical_text = canonical or ""
            # Provide more canonical detail for consistency
            prompt += f"\n  CANONICAL DESCRIPTION (use exactly): {canonical_text[:200]}"
        prompt += f"\n  Current Pose: {pose} | Emotion: {emotion}\n"
    
    # Enhanced reference image intelligence for entity appearance
    if rich_context["reference_tags"]:
        prompt += f"\n**ENTITY REFERENCE DETAILS ({len(rich_context['reference_tags'])} sources):**\n"
        for ref in rich_context["reference_tags"]:
            tags_str = ", ".join(ref["tags"][:8])  # More tags for detail
            prompt += f"- {ref['entity']} ({ref['category']}): {tags_str} [confidence: {ref['confidence']:.2f}]\n"
    
    # Comprehensive style guide for flexible parameter derivation
    if rich_context["style_text"]:
        style_text = rich_context["style_text"] or ""
        if style_text:
            prompt += f"\n**STYLE GUIDE (derive appropriate parameters):**\n{style_text[:600]}\n"
    
    if variation.style_notes:
        prompt += f"\n**SHOT-SPECIFIC STYLE:**\n{variation.style_notes}\n"
    
    # Enhanced scene context for composition
    if rich_context["scene_data"]:
        scene = rich_context["scene_data"]
        prompt += f"\n**SCENE CONTEXT:**\n"
        prompt += f"Location: {getattr(scene, 'location', 'Unknown')}\n"
        prompt += f"Time: {getattr(scene, 'time_of_day', 'Unknown')}\n"
        if hasattr(scene, 'narrative') and scene.narrative:
            prompt += f"Narrative: {scene.narrative[:300]}\n"
    
    # Composition-focused guidance
    prompt += f"\n**COMPOSITION REQUIREMENTS:**\n"
    prompt += f"- Apply rule of thirds for subject placement\n"
    prompt += f"- Create clear focal hierarchy and depth layers\n"
    prompt += f"- Use leading lines and perspective to guide viewer attention\n"
    prompt += f"- Balance negative space with detailed elements\n"
    prompt += f"- Consider camera angle impact on emotional tone\n"
    
    # Add negative guidance
    negative_elements = []
    if state.reviewed_plan and state.reviewed_plan.negative_prompt:
        negative_elements.append(state.reviewed_plan.negative_prompt)
    
    prompt += f"""
**YOUR TASK:**
Create a single, exceptionally detailed Midjourney v6 prompt (100-150 words) that:
1. Uses EXACT canonical descriptions for ALL entities to ensure consistency across storyboard
2. Maintains identical physical traits, clothing, and appearance details for each character
3. Applies sophisticated composition principles (rule of thirds, depth, focal hierarchy)
4. Integrates reference image details for rich visual textures and entity accuracy
5. ANALYZES STYLE GUIDE to determine visual approach (photorealistic, illustrated, animated, cartoon, painterly, etc.)
6. Includes detailed cinematography language (camera work, lighting, framing) appropriate to derived style
7. Specifies precise surface textures, materials, and environmental details matching the visual style
8. Adds technical quality markers appropriate to the style (avoid defaulting to photorealistic terms)
9. Derives appropriate stylistic parameters (--stylize, --chaos, --style, etc.) from style guide analysis
10. Base aspect ratio: --ar {aspect_ratio} (adjust if style guide suggests different ratio)
11. Always include --v 6 for latest model
{f"12. Include comprehensive negatives: --no {', '.join(negative_elements)}" if negative_elements else ""}

ENTITY CONSISTENCY IS PARAMOUNT - use identical descriptions every time an entity appears.
EMPHASIZE COMPOSITION MASTERY - create visually compelling, professionally framed shots.
DERIVE STYLE FROM GUIDE - analyze style guide to determine appropriate visual aesthetic and technical approach.

GENERATE MASTERFULLY DETAILED PROMPT - NO EXPLANATIONS, MAXIMUM ENTITY CONSISTENCY."""
    
    return prompt


def _derive_mj_suffix(state: WorkflowState, variation) -> str:
    """Derive base Midjourney aspect ratio - stylistic parameters now AI-derived from style guide."""
    ar_map = {"square": "1:1", "landscape": "3:2", "portrait": "2:3", "auto": "1:1"}
    ar = ar_map.get(state.config.get("aspect_ratio", "square"), "1:1")
    return f"--ar {ar} --v 6"  # Base parameters only, AI derives stylistic ones 