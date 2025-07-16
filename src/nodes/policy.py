"""Policy node for rule-based decisions on accept/retry/give-up."""

from typing import Dict, Any
from pathlib import Path
import shutil

from src.models import WorkflowState
from src.utils import log_entry, check_budget


def policy_node(state: WorkflowState) -> WorkflowState:
    """Make policy decision based on QA results and retry counts."""
    
    # Check budget first
    if not check_budget(state):
        state.policy_action = "give_up"
        log_entry(state, "policy", "budget_exceeded")
        return state
    
    # Get QA result (prefer vision QA if available)
    qa_result = state.vision_qa_result if state.vision_qa_result else state.fast_qa_result
    
    if not qa_result:
        # No QA performed, accept by default
        state.policy_action = "accept"
        log_entry(state, "policy", "no_qa_accept")
        return state
    
    # Apply policy rules
    if qa_result.status == "pass":
        state.policy_action = "accept"
        decision_reason = "qa_passed"
        
    elif qa_result.status == "retry":
        # Check retry limits
        if state.retry_count >= state.max_retries:
            # Try edit if we haven't exceeded edit retries
            if state.edit_retry_count < state.max_edit_retries:
                state.policy_action = "retry_edit"
                decision_reason = "retry_with_edit"
            else:
                # Choose the best image instead of accepting low quality
                _choose_best_image(state)
                state.policy_action = "accept"
                decision_reason = "retry_limit_accept_best"
        else:
            # Retry with new generation
            state.policy_action = "retry_new"
            decision_reason = "retry_new_generation"
            
    else:  # fail
        # Check if we should give up or retry
        total_attempts = state.retry_count + state.edit_retry_count
        
        if total_attempts >= state.max_retries + state.max_edit_retries:
            # Choose the best image instead of giving up
            _choose_best_image(state)
            state.policy_action = "accept"
            decision_reason = "max_attempts_accept_best"
        else:
            # Try one more time
            if state.retry_count < state.max_retries:
                state.policy_action = "retry_new"
                decision_reason = "fail_retry_new"
            else:
                state.policy_action = "retry_edit"
                decision_reason = "fail_retry_edit"
    
    # Special case: if this is already a retry and quality is very low, give up
    if (state.retry_count > 0 and qa_result.quality_score < 0.3):
        state.policy_action = "give_up"
        decision_reason = "low_quality_give_up"
    
    # Console progress output
    print(f"[Policy] Decision: {state.policy_action} (reason: {decision_reason})")
    
    # Log decision
    log_entry(state, "policy", state.policy_action,
             extra={
                 "reason": decision_reason,
                 "qa_status": qa_result.status,
                 "qa_score": qa_result.quality_score,
                 "retry_count": state.retry_count,
                 "edit_retry_count": state.edit_retry_count
             })
    
    return state


def _choose_best_image(state: WorkflowState) -> None:
    """Choose the best performing image from all attempts and move rejected images to separate folder."""
    if not state.image_attempts:
        return
    
    # Find the best image based on quality score
    best_attempt = max(state.image_attempts, key=lambda x: x.get("quality_score", 0.0))
    
    print(f"[Policy] Choosing best image from {len(state.image_attempts)} attempts (score: {best_attempt.get('quality_score', 0.0):.2f})")
    
    # Update current state to use the best image
    state.current_image_b64 = best_attempt["image_b64"]
    state.current_image_path = best_attempt["image_path"]
    
    # Update image_attempts to contain only the best attempt
    # This ensures QA nodes will evaluate the chosen image, not the last attempt
    state.image_attempts = [best_attempt]
    
    # Move rejected images to a separate directory for manual curation
    rejected_dir = Path(state.output_dir) / "rejected_frames"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    
    for attempt in state.image_attempts:
        if attempt["frame_id"] != best_attempt["frame_id"]:
            # This is a rejected image - move it to rejected folder
            try:
                source_path = Path(attempt["image_path"])
                if source_path.exists():
                    rejected_path = rejected_dir / source_path.name
                    shutil.move(str(source_path), str(rejected_path))
                    print(f"[Policy] Moved rejected image to {rejected_path}")
                    
                    # Also save metadata about why it was rejected
                    metadata_path = rejected_dir / f"{source_path.stem}_metadata.json"
                    import json
                    with open(metadata_path, 'w') as f:
                        json.dump({
                            "frame_id": attempt["frame_id"],
                            "quality_score": attempt.get("quality_score", 0.0),
                            "retry_count": attempt["retry_count"],
                            "edit_retry_count": attempt["edit_retry_count"],
                            "timestamp": attempt["timestamp"],
                            "reason": "rejected_after_retries"
                        }, f, indent=2)
            except Exception as e:
                print(f"[Policy] Error moving rejected image: {e}")
    
    log_entry(state, "policy", "best_image_selected",
             extra={
                 "best_score": best_attempt.get("quality_score", 0.0),
                 "total_attempts": len(state.image_attempts),
                 "rejected_count": len(state.image_attempts) - 1
             }) 