"""Memory update node for updating memory based on policy decisions."""

import json
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4

from src.models import WorkflowState
from src.utils import log_entry, save_workflow_state
from src.memory import MemoryService


def memory_update_node(state: WorkflowState) -> WorkflowState:
    """Update memory based on policy decision."""
    memory = state.get_memory_service()  # Use singleton memory service
    
    if state.policy_action == "accept":
        # Check if we have variations to accept
        if not state.variations or state.current_variation_idx >= len(state.variations):
            log_entry(state, "memory_update", "no_variation_to_accept")
            # Move to next iteration instead of crashing
            state.policy_action = "give_up"
        else:
            # Accept the current frame
            _accept_frame(state, memory)
        
        # Move to next variation or shot
        if state.current_variation_idx < len(state.variations) - 1:
            # Try next variation
            state.current_variation_idx += 1
            state.retry_count = 0
            state.edit_retry_count = 0
            state.fast_qa_result = None
            state.vision_qa_result = None
            state.image_attempts = []  # Clear attempts for next variation
            log_entry(state, "memory_update", "next_variation",
                     extra={"variation": state.current_variation_idx})
        else:
            # All variations done, move to next shot
            _advance_shot(state)
            
    elif state.policy_action == "retry_new":
        # Retry with new generation
        state.retry_count += 1
        state.fast_qa_result = None
        state.vision_qa_result = None
        log_entry(state, "memory_update", "retry_new",
                 extra={"retry_count": state.retry_count})
        
    elif state.policy_action == "retry_edit":
        # Retry with edit (keep current image for editing)
        state.fast_qa_result = None
        state.vision_qa_result = None
        log_entry(state, "memory_update", "retry_edit",
                 extra={"edit_retry_count": state.edit_retry_count})
        
    elif state.policy_action == "give_up":
        # Give up on current variation - save rejected images
        _save_rejected_images(state)
        log_entry(state, "memory_update", "give_up",
                 extra={"variation": state.current_variation_idx})
        
        # Try next variation or advance
        if state.current_variation_idx < len(state.variations) - 1:
            state.current_variation_idx += 1
            state.retry_count = 0
            state.edit_retry_count = 0
            state.fast_qa_result = None
            state.vision_qa_result = None
            state.image_attempts = []  # Clear attempts for next variation
        else:
            _advance_shot(state)
    
    # Save state periodically
    if len(state.accepted_frames) % 5 == 0:
        save_workflow_state(state)
    
    return state


def _accept_frame(state: WorkflowState, memory: MemoryService) -> None:
    """Accept the current frame and update memory."""
    current_variation = state.variations[state.current_variation_idx]
    
    print(f"[MemoryUpdate] ✅ Accepted frame for Scene {current_variation.scene_id} • Shot {current_variation.shot_id} (variation {state.current_variation_idx + 1})")
    
    # Create frame data
    frame_data = {
        "frame_id": str(uuid4()),
        "scene_id": current_variation.scene_id,
        "shot_id": current_variation.shot_id,
        "variation_idx": state.current_variation_idx,
        "prompt": current_variation.image_prompt,
        "negative_prompt": state.reviewed_plan.negative_prompt if state.reviewed_plan else "",
        "entities": [e.name for e in current_variation.entities],
        "camera": current_variation.camera.model_dump(),
        "image_path": state.current_image_path,
        "quality_score": state.fast_qa_result.quality_score if state.fast_qa_result else 0.7,
        "retry_count": state.retry_count,
        "edit_retry_count": state.edit_retry_count
    }
    
    # Index in memory
    memory.index_generated_frame(frame_data)
    
    # Add to accepted frames
    state.accepted_frames.append(frame_data)
    
    # Update visual memory
    memory.update_visual_memory(frame_data)
    
    # Save metadata
    _save_frame_metadata(state, frame_data)
    
    log_entry(state, "memory_update", "frame_accepted",
             extra={
                 "frame_id": frame_data["frame_id"],
                 "scene_id": current_variation.scene_id,
                 "shot_id": current_variation.shot_id,
                 "variation": state.current_variation_idx
             })


def _advance_shot(state: WorkflowState) -> None:
    """Advance to the next shot and update current scene if needed."""
    # Clear image attempts when advancing shot
    state.image_attempts = []
    
    # Reset workflow state for next shot
    state.current_variation_idx = 0
    state.retry_count = 0
    state.edit_retry_count = 0
    state.current_plan = None
    state.reviewed_plan = None
    state.variations = []
    state.fast_qa_result = None
    state.vision_qa_result = None
    state.current_image_b64 = None
    state.current_image_path = None
    
    print(f"[MemoryUpdate] ➡️ Moving to next shot ...")
    
    # Use configurable shots per scene, default to 1 for script-driven workflows
    shots_per_scene = state.config.get("shots_per_scene", 1)
    
    state.current_shot_idx += 1
    
    if state.current_shot_idx >= shots_per_scene:
        # Move to next scene
        state.current_shot_idx = 0
        state.current_scene_idx += 1
        
        # Check if we have more scenes
        if state.current_scene_idx < len(state.scenes):
            print(f"[MemoryUpdate] 🎬 Starting Scene {state.current_scene_idx + 1}")
            log_entry(state, "memory_update", "next_scene",
                     extra={"scene_idx": state.current_scene_idx})
        else:
            # No more scenes - workflow complete
            state.workflow_complete = True
            log_entry(state, "memory_update", "workflow_complete",
                     extra={"total_scenes": len(state.scenes),
                           "total_frames": len(state.accepted_frames)})
    else:
        log_entry(state, "memory_update", "next_shot",
                 extra={"shot_idx": state.current_shot_idx})


def _save_frame_metadata(state: WorkflowState, frame_data: Dict[str, Any]) -> None:
    """Save frame metadata to JSON file."""
    metadata_path = Path(state.output_dir) / "frames" / "metadata.json"
    
    # Load existing metadata
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"frames": []}
    
    # Add new frame
    metadata["frames"].append(frame_data)
    
    # Save back
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str) 


def _save_rejected_images(state: WorkflowState) -> None:
    """Save rejected images to a separate directory for manual curation."""
    if not state.image_attempts:
        return
    
    from pathlib import Path
    import shutil
    import json
    
    # Create rejected frames directory
    rejected_dir = Path(state.output_dir) / "rejected_frames"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    
    for attempt in state.image_attempts:
        try:
            source_path = Path(attempt["image_path"])
            if source_path.exists():
                rejected_path = rejected_dir / source_path.name
                shutil.move(str(source_path), str(rejected_path))
                print(f"[MemoryUpdate] Moved rejected image to {rejected_path}")
                
                # Save metadata about why it was rejected
                metadata_path = rejected_dir / f"{source_path.stem}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump({
                        "frame_id": attempt["frame_id"],
                        "quality_score": attempt.get("quality_score", 0.0),
                        "retry_count": attempt["retry_count"],
                        "edit_retry_count": attempt["edit_retry_count"],
                        "timestamp": attempt["timestamp"],
                        "reason": "variation_given_up"
                    }, f, indent=2)
        except Exception as e:
            print(f"[MemoryUpdate] Error saving rejected image: {e}")
    
    print(f"[MemoryUpdate] Saved {len(state.image_attempts)} rejected images")
    
    # Clear attempts after saving
    state.image_attempts = [] 