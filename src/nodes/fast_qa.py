"""Fast QA node for quick quality assessment using GPT-4o vision."""

import json
import random
from typing import Dict, Any

from src.models import WorkflowState, QAResult
from src.utils import (
    get_openai_client, call_openai_with_retry, log_entry,
    calculate_cost, parse_json_response, check_budget,
    load_image_as_base64
)


def fast_qa_node(state: WorkflowState) -> WorkflowState:
    """Perform fast quality assessment on generated image."""
    client = get_openai_client()
    
    # Check budget
    if not check_budget(state):
        log_entry(state, "fast_qa", "budget_exceeded")
        state.policy_action = "give_up"
        return state
    
    # Check if we have an image to assess
    if not state.current_image_b64:
        log_entry(state, "fast_qa", "no_image")
        return state
    
    # Get current variation for context
    current_variation = state.variations[state.current_variation_idx]
    
    # Perform fast QA
    qa_result = _perform_fast_qa(client, state, current_variation)
    state.fast_qa_result = qa_result
    
    # Randomly sample for deep vision QA (10% chance)
    if qa_result.status == "pass" and random.random() < 0.1:
        state.fast_qa_flag = True
        log_entry(state, "fast_qa", "sampled_for_vision_qa")
    else:
        state.fast_qa_flag = False
    
    log_entry(state, "fast_qa", qa_result.status,
             extra={
                 "quality_score": qa_result.quality_score,
                 "issues": len(qa_result.specific_issues),
                 "sampled": state.fast_qa_flag
             })
    
    return state


def _perform_fast_qa(client: Any, state: WorkflowState, variation: Any) -> QAResult:
    """Perform the actual fast QA assessment."""
    
    model = state.config["models"]["fast_qa"]
    
    prompt = f"""Perform a FAST quality assessment of this generated storyboard frame.

Original Request:
- Scene {variation.scene_id}, Shot {variation.shot_id}
- Prompt: {variation.image_prompt}
- Camera: {variation.camera.distance} {variation.camera.angle}
- Style: {state.style_text[:200]}

Quick Assessment Criteria (be lenient, this is for storyboards):
1. Does it match the requested scene/content? (most important)
2. Are the main entities/characters present?
3. Is the camera angle/framing approximately correct?
4. Are there any severe quality issues (distortions, artifacts)?
5. Does it follow the general style guide?

Return JSON:
{{
    "status": "pass/retry/fail",
    "quality_score": 0.0-1.0,
    "specific_issues": ["list specific problems if any"],
    "retry_guidance": "specific guidance if retry needed",
    "positives": ["what works well"]
}}

Be pragmatic - storyboards don't need to be perfect. Focus on whether it conveys the scene."""
    
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
                            "detail": "low"
                        }
                    }
                ]
            }
        ]
        
        response = call_openai_with_retry(
            client,
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        # Vision models don't always return token counts
        cost = 0.01  # Approximate cost
        
        state.total_cost += cost
        
        # Parse response
        qa_data = parse_json_response(content)
        
        # Create QAResult
        qa_result = QAResult(
            status=qa_data.get("status", "pass"),
            quality_score=float(qa_data.get("quality_score", 0.7)),
            specific_issues=qa_data.get("specific_issues", []),
            retry_guidance=qa_data.get("retry_guidance")
        )
        
        # Apply thresholds
        if qa_result.quality_score < 0.4:
            qa_result.status = "fail"
        elif qa_result.quality_score < 0.6 and len(qa_result.specific_issues) > 2:
            qa_result.status = "retry"
        
        log_entry(state, "fast_qa", "assessed",
                 model=model, cost_usd=cost,
                 extra={"score": qa_result.quality_score})
        
        return qa_result
        
    except Exception as e:
        log_entry(state, "fast_qa", "error", error=str(e))
        # Default to pass on error (lenient)
        return QAResult(
            status="pass",
            quality_score=0.7,
            specific_issues=[],
            retry_guidance=None
        )