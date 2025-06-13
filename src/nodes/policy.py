"""Policy node for rule-based decisions on accept/retry/give-up."""

from typing import Dict, Any

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
                # Accept with lower quality
                state.policy_action = "accept"
                decision_reason = "retry_limit_accept"
        else:
            # Retry with new generation
            state.policy_action = "retry_new"
            decision_reason = "retry_new_generation"
            
    else:  # fail
        # Check if we should give up or retry
        total_attempts = state.retry_count + state.edit_retry_count
        
        if total_attempts >= state.max_retries + state.max_edit_retries:
            state.policy_action = "give_up"
            decision_reason = "max_attempts_reached"
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