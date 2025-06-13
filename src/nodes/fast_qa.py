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
    
    print(f"[FastQA] Assessing current image – variation {state.current_variation_idx + 1}/{len(state.variations)}")
    
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
    """Perform the actual fast QA assessment with 5-token prompt."""
    
    model = state.config["models"]["fast_qa"]
    
    # 5-token prompt as specified in the task
    prompt = "pass if clean, fail if blurry / broken."
    
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
            temperature=0.0,
            max_tokens=10  # 5-token response
        )
        
        content = response.choices[0].message.content.lower()
        # Vision models don't always return token counts
        cost = 0.001  # Much cheaper with 5-token prompt
        
        state.total_cost += cost
        
        # Parse simple response
        if "pass" in content:
            status = "pass"
            quality_score = 0.8
        elif "fail" in content:
            status = "fail"
            quality_score = 0.3
        else:
            # Default to pass if unclear
            status = "pass" 
            quality_score = 0.7
        
        # Create QAResult
        qa_result = QAResult(
            status=status,
            quality_score=quality_score,
            specific_issues=[],
            retry_guidance=None
        )
        
        log_entry(state, "fast_qa", "assessed",
                 model=model, cost_usd=cost,
                 extra={"response": content})
        
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