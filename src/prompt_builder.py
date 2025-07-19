"""Prompt building utilities for both OpenAI and Midjourney rendering."""

from typing import List, Any, Dict
from src.models import WorkflowState
from src.utils import get_openai_client, call_openai_with_retry, parse_json_response


def build_raw_prompt(state: WorkflowState, variation: Any) -> str:
    """Build the complete image generation prompt.
    
    This is the canonical prompt builder used by both OpenAI image generation
    and Midjourney prompt conversion.
    """
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
    
    # 4) Basic spatial positioning (if entities are available)
    if (hasattr(variation, 'entities') and variation.entities and 
        state.config.get("use_spatial_analysis", True)):
        try:
            # Simple spatial analysis for basic prompts
            spatial_info = _analyze_spatial_relationships(state, variation)
            # Extract just the key positioning info for basic prompts
            simplified_spatial = spatial_info.split('\n')[0:3]  # Take first 3 lines
            sections.append("<POSITIONING>\n" + '\n'.join(simplified_spatial) + "\n</POSITIONING>")
        except:
            # Fallback if spatial analysis fails
            entity_names = [e.name for e in variation.entities if hasattr(e, 'name')]
            if entity_names:
                sections.append(f"<POSITIONING>\nClearly position entities: {', '.join(entity_names)} with explicit spatial relationships.\n</POSITIONING>")

    # 5) Camera metadata
    cam = variation.camera
    cam_desc = f"type={cam.type}; angle={cam.angle}; distance={cam.distance}"
    if cam.movement:
        cam_desc += f"; movement={cam.movement}"
    sections.append("<CAMERA>\n" + cam_desc + "\n</CAMERA>")

    # 6) Negative prompt / avoid list
    if state.reviewed_plan and state.reviewed_plan.negative_prompt:
        sections.append("<NEGATIVE>\n" + state.reviewed_plan.negative_prompt.strip() + "\n</NEGATIVE>")

    # Final prompt string (double line breaks between blocks for clarity)
    return "\n\n".join(sections) 


def build_enhanced_raw_prompt(state: WorkflowState, variation: Any, rich_context: Dict[str, Any] = None) -> str:
    """Build an enhanced image generation prompt with entity consistency and spatial relationships.
    
    This enhanced version includes detailed entity descriptions, spatial relationships,
    and consistency elements similar to the Midjourney approach but optimized for GPT-image-1.
    """
    sections: List[str] = []
    
    # 1) Global narrative / design context
    if state.static_summary:
        sections.append("<CONTEXT>\n" + state.static_summary.strip() + "\n</CONTEXT>")

    # 2) Enhanced entity specifications with consistency
    if variation.entities:
        entity_details = []
        entity_details.append("ENTITIES IN SHOT (maintain exact consistency):")
        
        # Get memory service for canonical descriptions
        memory = state.get_memory_service()
        
        for entity in variation.entities:
            canonical = memory.lookup_canonical(entity.name) if memory else ""
            description = entity.description or "unknown"
            pose = entity.pose or "unknown"
            emotion = entity.emotion or "neutral"
            
            entity_line = f"- {entity.name}: {description}"
            if canonical:
                canonical_text = canonical or ""
                entity_line += f"\n  CANONICAL APPEARANCE: {canonical_text[:150]}"
            entity_line += f"\n  POSE: {pose} | EMOTION: {emotion}"
            entity_details.append(entity_line)
        
        sections.append("<ENTITIES>\n" + "\n".join(entity_details) + "\n</ENTITIES>")

    # 3) AI-driven spatial relationship analysis
    if (variation.entities and state.config.get("use_spatial_analysis", True)):
        spatial_analysis = _analyze_spatial_relationships(state, variation)
        sections.append("<SPATIAL_RELATIONSHIPS>\n" + spatial_analysis + "\n</SPATIAL_RELATIONSHIPS>")
    else:
        # Fallback for shots without explicit entities or when spatial analysis is disabled
        if variation.entities:
            entity_names = [e.name for e in variation.entities if hasattr(e, 'name')]
            sections.append(f"<SPATIAL_RELATIONSHIPS>\nEnsure clear spatial positioning of entities: {', '.join(entity_names)}.\n</SPATIAL_RELATIONSHIPS>")
        else:
            sections.append("<SPATIAL_RELATIONSHIPS>\nEnsure clear spatial composition with explicit positioning of all visual elements.\n</SPATIAL_RELATIONSHIPS>")

    # 4) Enhanced art-style with technical specifications
    style_block_parts: List[str] = []
    if state.style_text:
        style_block_parts.append("STYLE GUIDE:")
        style_block_parts.append(state.style_text[:500].strip())
        style_block_parts.append("\nDERIVE from style guide:")
        style_block_parts.append("- Visual approach (photorealistic, illustrated, animated, painterly)")
        style_block_parts.append("- Technical quality markers appropriate to style")
        style_block_parts.append("- Surface textures and materials matching aesthetic")
        style_block_parts.append("- Lighting and mood integration")
    
    if variation.style_notes:
        style_block_parts.append("\nSHOT-SPECIFIC STYLE:")
        style_block_parts.append(variation.style_notes.strip())
    
    if style_block_parts:
        sections.append("<STYLE_GUIDE>\n" + "\n".join(style_block_parts) + "\n</STYLE_GUIDE>")

    # 5) The shot description with composition emphasis
    shot_details = []
    shot_details.append("SHOT DESCRIPTION:")
    shot_details.append(variation.image_prompt.strip())
    shot_details.append("\nCOMPOSITION REQUIREMENTS:")
    shot_details.append("- Apply rule of thirds for subject placement")
    shot_details.append("- Create clear focal hierarchy and depth layers")
    shot_details.append("- Use leading lines and perspective to guide attention")
    shot_details.append("- Balance negative space with detailed elements")
    sections.append("<SHOT_PROMPT>\n" + "\n".join(shot_details) + "\n</SHOT_PROMPT>")

    # 6) Enhanced camera with cinematic language
    cam = variation.camera
    cam_details = []
    cam_details.append(f"CAMERA SETUP: {cam.type} {cam.angle} {cam.distance}")
    if cam.movement:
        cam_details.append(f"MOVEMENT: {cam.movement}")
    cam_details.append("CINEMATIC APPROACH:")
    cam_details.append("- Professional framing and perspective")
    cam_details.append("- Appropriate depth of field for shot type")
    cam_details.append("- Consider emotional impact of camera angle")
    sections.append("<CAMERA>\n" + "\n".join(cam_details) + "\n</CAMERA>")

    # 7) Reference context if available
    if rich_context and rich_context.get("reference_tags"):
        ref_details = []
        ref_details.append(f"REFERENCE DETAILS ({len(rich_context['reference_tags'])} sources):")
        for ref in rich_context["reference_tags"][:4]:  # Limit to top 4 references
            tags_str = ", ".join(ref["tags"][:6])  # Top 6 tags per reference
            ref_details.append(f"- {ref['entity']} ({ref['category']}): {tags_str}")
        sections.append("<REFERENCES>\n" + "\n".join(ref_details) + "\n</REFERENCES>")

    # 8) Scene context for environmental details
    if rich_context and rich_context.get("scene_data"):
        scene = rich_context["scene_data"]
        scene_details = []
        scene_details.append("SCENE CONTEXT:")
        scene_details.append(f"Location: {getattr(scene, 'location', 'Unknown')}")
        scene_details.append(f"Time: {getattr(scene, 'time_of_day', 'Unknown')}")
        if hasattr(scene, 'narrative') and scene.narrative:
            scene_details.append(f"Narrative: {scene.narrative[:200]}")
        sections.append("<SCENE_CONTEXT>\n" + "\n".join(scene_details) + "\n</SCENE_CONTEXT>")

    # 9) Quality and consistency directives
    quality_directives = []
    quality_directives.append("GENERATION REQUIREMENTS:")
    quality_directives.append("- Use EXACT canonical descriptions for all entities")
    quality_directives.append("- Maintain identical physical traits and appearance")
    quality_directives.append("- Position entities with precise spatial relationships")
    quality_directives.append("- Apply sophisticated composition principles")
    quality_directives.append("- Ensure high visual quality and detail")
    quality_directives.append("- Match style guide aesthetic consistently")
    sections.append("<REQUIREMENTS>\n" + "\n".join(quality_directives) + "\n</REQUIREMENTS>")

    # 10) Negative prompt / avoid list
    if state.reviewed_plan and state.reviewed_plan.negative_prompt:
        sections.append("<NEGATIVE>\n" + state.reviewed_plan.negative_prompt.strip() + "\n</NEGATIVE>")

    # Final prompt string with clear separators
    return "\n\n".join(sections) 


def _analyze_spatial_relationships(state: WorkflowState, variation: Any) -> str:
    """Use AI to analyze spatial relationships in the shot description."""
    if not variation.entities or not variation.image_prompt:
        return "No spatial analysis available - insufficient entity or shot information."
    
    try:
        # Add debug logging
        entity_count = len(variation.entities)
        print(f"[SpatialAnalysis] Analyzing {entity_count} entities for spatial relationships...")
        
        client = get_openai_client()
        
        # Build entity list for analysis
        entity_list = []
        for entity in variation.entities:
            entity_desc = f"- {entity.name}: {entity.description or 'unknown'}"
            if entity.pose:
                entity_desc += f" (pose: {entity.pose})"
            if entity.emotion:
                entity_desc += f" (emotion: {entity.emotion})"
            entity_list.append(entity_desc)
        
        prompt = f"""EXTRACT SPATIAL RELATIONSHIPS from this shot description:

SHOT: {variation.image_prompt}

ENTITIES PRESENT:
{chr(10).join(entity_list)}

ANALYZE and OUTPUT explicit spatial positioning information in this format:

SPATIAL_ANALYSIS:
- Entity positions: [specify left/right/center, foreground/background, above/below arrangements]
- Orientations: [who faces whom, body directions, head orientations]
- Interactions: [who looks at whom, pointing gestures, reaching toward]
- Relative positioning: [proximity relationships, groupings, formations]
- Depth layers: [which entities are in front/behind others]

Focus on CONCRETE spatial details that can be visualized. If information is not explicit in the shot description, use reasonable spatial assumptions based on the scene context.

Be specific about entity positioning using directional terms (left, right, behind, in front, above, below, near, far)."""

        response = call_openai_with_retry(
            client,
            model="gpt-4o-mini",  # Use cheaper model for spatial analysis
            messages=[
                {"role": "system", "content": "You are a spatial relationship analyst for visual scene composition. Extract concrete positioning information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        spatial_analysis = response.choices[0].message.content.strip()
        
        # Update costs (approximate for gpt-4o-mini)
        state.total_cost += 0.002  # Small cost for spatial analysis
        
        print(f"[SpatialAnalysis] ✅ Generated spatial analysis ({len(spatial_analysis)} chars)")
        return spatial_analysis
        
    except Exception as e:
        print(f"[SpatialAnalysis] ❌ Error: {str(e)}")
        # Fallback to basic spatial guidelines if AI analysis fails
        return f"Spatial analysis unavailable (error: {str(e)}). Use clear entity positioning with explicit left/right, front/back relationships." 