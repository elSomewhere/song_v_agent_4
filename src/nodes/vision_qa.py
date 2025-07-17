"""Vision QA node for deep quality audit using GPT-4o vision."""

import json
from typing import Dict, Any, List

from src.models import WorkflowState, QAResult
from src.utils import (
    get_openai_client, call_openai_with_retry, log_entry,
    calculate_cost, parse_json_response, check_budget,
    load_image_as_base64
)
from src.memory import MemoryService


def vision_qa_node(state: WorkflowState) -> WorkflowState:
    """Perform deep vision quality audit on sampled frames."""
    
    # Only run if flagged by fast QA
    if not state.fast_qa_flag:
        return state
    
    client = get_openai_client()
    print("[VisionQA] Deep audit on sampled frame ...")
    memory = state.get_memory_service()  # Use singleton memory service
    
    # Check budget
    if not check_budget(state):
        log_entry(state, "vision_qa", "budget_exceeded")
        state.policy_action = "give_up"
        return state
    
    # Check if we have an image to assess
    if not state.current_image_b64:
        log_entry(state, "vision_qa", "no_image")
        return state
    
    # Get current variation and context
    current_variation = state.variations[state.current_variation_idx]
    
    # Get visual context for comparison
    nearby_frames, relevant_refs = memory.get_visual_context(
        current_variation.scene_id,
        current_variation.shot_id,
        window_size=3
    )
    
    # Perform deep QA
    qa_result = _perform_vision_qa(client, state, current_variation, nearby_frames, relevant_refs)
    state.vision_qa_result = qa_result
    
    # Update quality score in latest attempt (vision QA overrides fast QA score)
    if state.image_attempts:
        state.image_attempts[-1]["quality_score"] = qa_result.quality_score
    
    log_entry(state, "vision_qa", qa_result.status,
             extra={
                 "quality_score": qa_result.quality_score,
                 "issues": len(qa_result.specific_issues)
             })
    
    return state


def _perform_vision_qa(client: Any, state: WorkflowState, variation: Any,
                      nearby_frames: List[Dict], relevant_refs: List[Dict]) -> QAResult:
    """Perform deep vision quality assessment."""
    
    model = state.config["models"]["vision_qa"]
    
    # Build context summary
    context_summary = _build_context_summary(nearby_frames, relevant_refs)
    
    prompt = f"""Perform a DEEP quality audit of this storyboard frame.

Scene Context:
- Scene {variation.scene_id}, Shot {variation.shot_id}
- Full Prompt: {variation.image_prompt}
- Camera: {variation.camera.model_dump()}
- Style Guide: {state.style_text[:300]}

Entities Expected:
{json.dumps([e.model_dump() for e in variation.entities], indent=2)}

Visual Context:
{context_summary}

Deep Assessment Criteria:
1. Character/Entity Consistency
   - Are all entities present and recognizable?
   - Do they match reference images (if any)?
   - Are poses/emotions correctly depicted?

2. Visual Continuity
   - Does it maintain consistency with previous frames?
   - Are lighting/colors consistent?
   - Does the environment match the scene?

3. Technical Quality
   - Resolution and clarity
   - No distortions or artifacts
   - Proper composition and framing

4. Style Adherence
   - Matches the specified art style
   - Consistent with style guide
   - Appropriate for storyboard use

5. Narrative Clarity
   - Does it clearly convey the scene?
   - Is the action/emotion readable?
   - Would it work in sequence?

Return detailed JSON:
{{
    "status": "pass/retry/fail",
    "quality_score": 0.0-1.0,
    "specific_issues": ["detailed list of issues"],
    "retry_guidance": "specific instructions for improvement",
    "consistency_scores": {{
        "character": 0.0-1.0,
        "environment": 0.0-1.0,
        "style": 0.0-1.0,
        "narrative": 0.0-1.0
    }},
    "recommendations": ["list of specific recommendations"]
}}"""
    
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{state.current_image_b64}",
                            "detail": "high"  # Use high detail for deep analysis
                        }
                    }
                ]
            }
        ]
        
        # Add comparison frames if available
        comparison_images = _get_comparison_images(state, nearby_frames[:2])
        for comp_image in comparison_images:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{comp_image['base64']}",
                    "detail": "low"
                }
            })
        
        response = call_openai_with_retry(
            client,
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=state.config.get("api_max_tokens", {}).get("vision_qa", 1000)
        )
        
        content = response.choices[0].message.content
        cost = 0.02  # Approximate cost for vision QA
        
        state.total_cost += cost
        
        # Parse response
        qa_data = parse_json_response(content)
        
        # Calculate overall score from sub-scores
        consistency_scores = qa_data.get("consistency_scores", {})
        avg_consistency = sum(consistency_scores.values()) / len(consistency_scores) if consistency_scores else 0.7
        
        # Create QAResult
        qa_result = QAResult(
            status=qa_data.get("status", "pass"),
            quality_score=float(qa_data.get("quality_score", avg_consistency)),
            specific_issues=qa_data.get("specific_issues", []),
            retry_guidance=qa_data.get("retry_guidance")
        )
        
        # Stricter thresholds for deep QA
        if qa_result.quality_score < 0.5:
            qa_result.status = "fail"
        elif qa_result.quality_score < 0.7 and len(qa_result.specific_issues) > 1:
            qa_result.status = "retry"
        
        log_entry(state, "vision_qa", "assessed",
                 model=model, cost_usd=cost,
                 extra={
                     "score": qa_result.quality_score,
                     "consistency": consistency_scores
                 })
        
        return qa_result
        
    except Exception as e:
        log_entry(state, "vision_qa", "error", error=str(e))
        # Default to fast QA result on error, but fail if fast QA also failed
        if state.fast_qa_result and state.fast_qa_result.status != "fail":
            return state.fast_qa_result
        else:
            # If fast QA failed or doesn't exist, fail safe
            return QAResult(
                status="fail",
                quality_score=0.0,
                specific_issues=["Vision QA system error: " + str(e)],
                retry_guidance="Vision QA failed due to system error"
            )


def _build_context_summary(nearby_frames: List[Dict], relevant_refs: List[Dict]) -> str:
    """Build a summary of visual context."""
    parts = []
    
    if nearby_frames:
        # Filter out None values from nearby_frames
        valid_frames = [frame for frame in nearby_frames if frame is not None and isinstance(frame, dict)]
        if valid_frames:
            parts.append(f"Previous Frames: {len(valid_frames)} frames from nearby scenes")
            for frame in valid_frames[:2]:
                scene_id = frame.get('scene_id', 'unknown')
                shot_id = frame.get('shot_id', 'unknown')
                prompt = frame.get('prompt', 'No prompt available')
                # Handle None prompt safely
                if prompt is None:
                    prompt = 'No prompt available'
                parts.append(f"  - Scene {scene_id} Shot {shot_id}: {prompt[:100]}...")
    
    if relevant_refs:
        parts.append(f"\nReference Images: {len(relevant_refs)} relevant references")
        for ref in relevant_refs[:3]:
            if ref is not None and isinstance(ref, dict):
                tags = ", ".join(ref.get('tags', [])[:5])
                entity = ref.get('entity', 'unknown')
                parts.append(f"  - {entity}: {tags}")
    
    return "\n".join(parts) if parts else "No visual context available"


def _get_comparison_images(state: WorkflowState, frames: List[Dict]) -> List[Dict[str, Any]]:
    """Get comparison images for visual QA."""
    comparison_images = []
    
    for frame in frames:
        if frame.get("image_path"):
            try:
                comparison_images.append({
                    "frame_id": frame["frame_id"],
                    "base64": load_image_as_base64(frame["image_path"])
                })
            except:
                pass
    
    return comparison_images 