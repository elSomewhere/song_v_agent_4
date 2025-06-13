"""Reviewer node for reviewing and adjusting plans with visual context."""

import json
from typing import Dict, Any, List, Optional

from src.models import WorkflowState, ReviewedPlan, ScenePlan
from src.utils import (
    get_openai_client, call_openai_with_retry, log_entry,
    calculate_cost, parse_json_response, check_budget,
    load_image_as_base64, count_tokens_approx
)
from src.memory import MemoryService


def reviewer_node(state: WorkflowState) -> WorkflowState:
    """Review and adjust the plan with visual context using GPT-4o."""
    client = get_openai_client()
    memory = MemoryService(state)
    
    # Check budget
    if not check_budget(state):
        log_entry(state, "reviewer", "budget_exceeded")
        state.policy_action = "give_up"
        return state
    
    # Check if we have a plan to review
    if not state.current_plan:
        log_entry(state, "reviewer", "no_plan")
        return state
    
    print(f"[Reviewer] Scene {state.current_plan.scene_id} • Shot {state.current_plan.shot_id} – reviewing plan ...")
    
    plan = state.current_plan
    
    # Get extended visual context
    nearby_frames, relevant_refs = memory.get_visual_context(
        plan.scene_id,
        plan.shot_id,
        window_size=state.config.get("ctx_window", 4)
    )
    
    # Get top reference images to show
    visual_refs = _get_visual_references(state, relevant_refs, limit=state.config.get("ctx_images", 3))
    
    # Build prompt with visual context
    prompt = _build_reviewer_prompt(state, plan, nearby_frames, relevant_refs, visual_refs)
    
    # Build messages with images
    messages = [
        {"role": "system", "content": "You are a senior storyboard director reviewing shot plans for consistency and quality."},
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    
    # Add reference images to the prompt
    for ref_info in visual_refs:
        if ref_info.get("base64"):
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{ref_info['base64']}",
                    "detail": "low"
                }
            })
    
    try:
        model = state.config["models"]["reviewer"]
        response = call_openai_with_retry(
            client,
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else count_tokens_approx(prompt) + 500
        cost = calculate_cost(model, tokens // 2, tokens // 2)  # Approximate
        
        state.total_tokens += tokens
        state.total_cost += cost
        
        # Parse response
        review_data = parse_json_response(content)
        
        # Create reviewed plan
        reviewed_plan = ReviewedPlan(
            approved_plan=plan,  # Keep original plan (could be modified based on review)
            visual_context=[ref["frame_id"] for ref in relevant_refs[:3]],
            negative_prompt=review_data.get("negative_prompt", ""),
            estimated_tokens=review_data.get("estimated_tokens", 1000)
        )
        
        # Apply any suggested modifications to the plan
        if "modified_prompt" in review_data:
            plan.image_prompt = review_data["modified_prompt"]
        
        if "style_adjustments" in review_data:
            plan.style_notes = review_data.get("style_adjustments", plan.style_notes)
        
        state.reviewed_plan = reviewed_plan
        
        # Update episodic memory
        memory.update_episodic_memory({
            "stage": "reviewer",
            "scene_id": plan.scene_id,
            "shot_id": plan.shot_id,
            "review": review_data
        })
        
        log_entry(state, "reviewer", "success",
                 model=model, tokens=tokens, cost_usd=cost,
                 extra={"visual_refs": len(visual_refs)})
        
    except Exception as e:
        log_entry(state, "reviewer", "error", error=str(e))
        # If review fails, use original plan
        state.reviewed_plan = ReviewedPlan(
            approved_plan=plan,
            visual_context=[],
            negative_prompt="",
            estimated_tokens=1000
        )
    
    return state


def _get_visual_references(state: WorkflowState, refs: List[Dict], 
                          limit: int) -> List[Dict[str, Any]]:
    """Get visual reference images with base64 data."""
    visual_refs = []
    
    for ref in refs[:limit]:
        ref_info = {
            "frame_id": ref["frame_id"],
            "entity": ref["entity"],
            "tags": ref["tags"][:5]  # Limit tags
        }
        
        # Try to load original image
        if ref.get("original_path"):
            try:
                ref_info["base64"] = load_image_as_base64(ref["original_path"])
            except:
                # Try thumbnail as fallback
                if ref.get("thumb_path"):
                    try:
                        ref_info["base64"] = load_image_as_base64(ref["thumb_path"])
                    except:
                        pass
        
        visual_refs.append(ref_info)
    
    return visual_refs


def _build_reviewer_prompt(state: WorkflowState, plan: ScenePlan,
                          nearby_frames: List[Dict], relevant_refs: List[Dict],
                          visual_refs: List[Dict]) -> str:
    """Build the prompt for the reviewer."""
    
    # Format nearby frames context
    frames_context = ""
    if nearby_frames:
        frame_summaries = []
        for frame in nearby_frames[:3]:
            summary = f"- Scene {frame['scene_id']} Shot {frame['shot_id']}: {frame['prompt'][:150]}..."
            frame_summaries.append(summary)
        frames_context = "\n".join(frame_summaries)
    
    prompt = f"""Review this storyboard shot plan for visual consistency and quality.

Current Plan:
{json.dumps(plan.model_dump(), indent=2)}

Style Guide:
{state.style_text[:500]}

Visual Context from Previous Frames:
{frames_context or "No previous frames"}

Reference Images Available: {len(visual_refs)} images showing relevant characters/environments

Your review should:
1. Check visual consistency with previous frames
2. Ensure style guide adherence
3. Verify character/entity consistency
4. Suggest improvements to the image prompt
5. Provide a negative prompt to avoid common issues
6. Estimate token usage for the generation

Return JSON with:
{{
    "approval": true/false,
    "consistency_score": 0.0-1.0,
    "issues": ["list of any issues found"],
    "modified_prompt": "improved version of the image prompt if needed",
    "style_adjustments": "any style-specific adjustments",
    "negative_prompt": "things to avoid in the image",
    "estimated_tokens": 1000,
    "notes": "additional guidance for the renderer"
}}"""
    
    return prompt 