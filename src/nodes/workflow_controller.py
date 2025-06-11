"""Workflow controller node for managing scene/shot progression."""

from src.models import WorkflowState
from src.utils import log_entry


def workflow_controller_node(state: WorkflowState) -> WorkflowState:
    """Control workflow progression through scenes and shots."""
    
    # Check if current shot processing is complete
    if state.policy_action in ["accept", "give_up"]:
        # Reset state for next shot/scene
        state.current_variation_idx = 0
        state.current_plan = None
        state.reviewed_plan = None
        state.variations = []
        state.fast_qa_result = None
        state.vision_qa_result = None
        state.policy_action = None
        state.retry_count = 0
        state.edit_retry_count = 0
        state.current_image_b64 = None
        state.current_image_path = None
        
        # Move to next shot
        state.current_shot_idx += 1
        
        # Check if we need to move to next scene
        # Assuming 10 shots per scene as a default
        if state.current_shot_idx >= 10:
            state.current_shot_idx = 0
            state.current_scene_idx += 1
            
            if state.current_scene_idx >= len(state.scenes):
                # All scenes processed
                log_entry(state, "workflow_controller", "complete",
                         extra={"scenes": state.current_scene_idx,
                               "frames": len(state.accepted_frames)})
                state.workflow_complete = True
            else:
                log_entry(state, "workflow_controller", "next_scene",
                         extra={"scene": state.current_scene_idx})
        else:
            log_entry(state, "workflow_controller", "next_shot",
                     extra={"scene": state.current_scene_idx,
                           "shot": state.current_shot_idx})
    
    # Check budget
    if state.total_cost >= state.budget_usd:
        log_entry(state, "workflow_controller", "budget_exceeded",
                 extra={"spent": state.total_cost, "budget": state.budget_usd})
        state.workflow_complete = True
    
    return state