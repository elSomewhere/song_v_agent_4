"""Planner node for creating scene plans using GPT-4o."""

import json
from typing import Dict, Any, List

from src.models import WorkflowState, ScenePlan, Entity, Camera
from src.utils import (
    get_openai_client, call_openai_with_retry, log_entry,
    calculate_cost, parse_json_response, check_budget
)
from src.memory import MemoryService


def planner_node(state: WorkflowState) -> WorkflowState:
    """Create a plan for the current scene using GPT-4o."""
    client = get_openai_client()
    memory = MemoryService(state)
    
    # Check budget
    if not check_budget(state):
        log_entry(state, "planner", "budget_exceeded")
        state.policy_action = "give_up"
        return state
    
    # Get current scene
    if state.current_scene_idx >= len(state.scenes):
        log_entry(state, "planner", "no_more_scenes")
        return state
    
    current_scene = state.scenes[state.current_scene_idx]
    
    # Get visual context from memory
    nearby_frames, relevant_refs = memory.get_visual_context(
        current_scene.scene_id, 
        state.current_shot_idx + 1,
        window_size=state.config.get("ctx_window", 4)
    )
    
    # Build context for planner
    context = _build_planner_context(state, current_scene, nearby_frames, relevant_refs)
    
    # Create prompt
    prompt = f"""You are planning a shot for a storyboard. Create a detailed plan for this scene.

Scene Information:
{json.dumps(current_scene.model_dump(), indent=2)}

Style Guide:
{state.style_text[:500]}

Entities:
{json.dumps(state.entities_dict, indent=2)}

Visual Context:
- Nearby frames: {len(nearby_frames)} frames
- Reference images: {len(relevant_refs)} images

Previous frames summary:
{context['frames_summary']}

Create a shot plan with:
1. entities: List of entities in the shot with poses/emotions
2. camera: Camera configuration (type, angle, distance, movement)
3. image_prompt: Detailed prompt for image generation
4. style_notes: Specific style instructions

Return as JSON matching this structure:
{{
    "entities": [
        {{"name": "character_name", "pose": "standing", "emotion": "happy", "description": "wearing blue shirt"}}
    ],
    "camera": {{
        "type": "static",
        "angle": "eye-level", 
        "distance": "medium",
        "movement": null
    }},
    "image_prompt": "A detailed description...",
    "style_notes": "Additional style guidance..."
}}"""
    
    try:
        model = state.config["models"]["planner"]
        response = call_openai_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional storyboard artist and cinematographer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        cost = calculate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        
        state.total_tokens += tokens
        state.total_cost += cost
        
        # Parse response
        plan_data = parse_json_response(content)
        
        # Create ScenePlan
        entities = [Entity(**e) for e in plan_data.get("entities", [])]
        camera = Camera(**plan_data.get("camera", {
            "type": "static",
            "angle": "eye-level", 
            "distance": "medium"
        }))
        
        plan = ScenePlan(
            scene_id=current_scene.scene_id,
            shot_id=state.current_shot_idx + 1,
            entities=entities,
            camera=camera,
            image_prompt=plan_data.get("image_prompt", ""),
            style_notes=plan_data.get("style_notes")
        )
        
        state.current_plan = plan
        
        # Update episodic memory
        memory.update_episodic_memory({
            "stage": "planner",
            "scene_id": current_scene.scene_id,
            "shot_id": state.current_shot_idx + 1,
            "plan": plan.model_dump()
        })
        
        log_entry(state, "planner", "success", 
                 model=model, tokens=tokens, cost_usd=cost,
                 extra={"scene_id": current_scene.scene_id, "shot_id": state.current_shot_idx + 1})
        
    except Exception as e:
        log_entry(state, "planner", "error", error=str(e))
        state.policy_action = "give_up"
    
    return state


def _build_planner_context(state: WorkflowState, current_scene: Any,
                          nearby_frames: List[Dict], relevant_refs: List[Dict]) -> Dict[str, Any]:
    """Build context information for the planner."""
    context = {
        "frames_summary": "",
        "refs_summary": ""
    }
    
    # Summarize nearby frames
    if nearby_frames:
        frame_summaries = []
        for frame in nearby_frames[:3]:  # Limit to 3 most recent
            summary = f"Scene {frame['scene_id']} Shot {frame['shot_id']}: {frame['prompt'][:100]}..."
            frame_summaries.append(summary)
        context["frames_summary"] = "\n".join(frame_summaries)
    else:
        context["frames_summary"] = "No previous frames"
    
    # Summarize reference images
    if relevant_refs:
        ref_summaries = []
        for ref in relevant_refs[:3]:  # Limit to 3 most relevant
            tags = ", ".join(ref['tags'][:5])
            summary = f"{ref['entity']} ({ref['category']}): {tags}"
            ref_summaries.append(summary)
        context["refs_summary"] = "\n".join(ref_summaries)
    else:
        context["refs_summary"] = "No reference images"
    
    return context 