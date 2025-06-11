#!/usr/bin/env python3
"""Main entry point for the VC-RAG-SBG system."""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from src.loader import Loader
from src.preprocess import ScriptPreprocessor, ReferencePreprocessor
from src.memory import MemoryService
from src.models import WorkflowState, Metrics
from src.utils import log_entry, save_workflow_state, ensure_directory

# Import workflow nodes
from src.nodes.planner import planner_node
from src.nodes.reviewer import reviewer_node
from src.nodes.variation_mgr import variation_mgr_node
from src.nodes.renderer import renderer_node
from src.nodes.fast_qa import fast_qa_node
from src.nodes.vision_qa import vision_qa_node
from src.nodes.policy import policy_node
from src.nodes.memory_update import memory_update_node


def preprocess_script_node(state: WorkflowState) -> WorkflowState:
    """Preprocess script to parse scenes."""
    with open(state.script_path, 'r') as f:
        script_content = f.read()
    
    preprocessor = ScriptPreprocessor(state)
    state.scenes = preprocessor.parse_script(script_content)
    
    log_entry(state, "preprocess_script", "success",
             extra={"scenes_found": len(state.scenes)})
    
    return state


def preprocess_refs_node(state: WorkflowState) -> WorkflowState:
    """Preprocess reference images if provided."""
    if not state.refs_dir:
        log_entry(state, "preprocess_refs", "skipped")
        return state
    
    preprocessor = ReferencePreprocessor(state)
    ref_metas = preprocessor.process_references(state.refs_dir)
    state.ref_index = ref_metas
    
    # Index in memory
    memory = MemoryService(state)
    memory.index_references(ref_metas)
    
    log_entry(state, "preprocess_refs", "success",
             extra={"refs_processed": len(ref_metas)})
    
    return state


def should_continue_workflow(state: WorkflowState) -> str:
    """Determine next step in workflow based on state."""
    
    # Check if we're done with all scenes
    if state.current_scene_idx >= len(state.scenes):
        return "end"
    
    # Check policy action
    if state.policy_action == "accept":
        # Check if more variations to process
        if state.current_variation_idx < len(state.variations) - 1:
            return "renderer"  # Process next variation
        else:
            return "planner"  # Start new shot
    elif state.policy_action == "retry_new":
        return "renderer"  # Retry rendering
    elif state.policy_action == "retry_edit":
        return "renderer"  # Retry with edit
    elif state.policy_action == "give_up":
        # Check if more variations to try
        if state.current_variation_idx < len(state.variations) - 1:
            return "renderer"  # Try next variation
        else:
            return "planner"  # Start new shot
    else:
        # No policy action yet, continue normal flow
        if not state.current_plan:
            return "planner"
        elif not state.reviewed_plan:
            return "reviewer"
        elif not state.variations:
            return "variation_mgr"
        elif not state.current_image_b64:
            return "renderer"
        elif not state.fast_qa_result:
            return "fast_qa"
        elif state.fast_qa_flag and not state.vision_qa_result:
            return "vision_qa"
        else:
            return "policy"


def build_workflow() -> StateGraph:
    """Build the LangGraph workflow."""
    
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # Add preprocessing nodes
    workflow.add_node("preprocess_script", preprocess_script_node)
    workflow.add_node("preprocess_refs", preprocess_refs_node)
    
    # Add main workflow nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("variation_mgr", variation_mgr_node)
    workflow.add_node("renderer", renderer_node)
    workflow.add_node("fast_qa", fast_qa_node)
    workflow.add_node("vision_qa", vision_qa_node)
    workflow.add_node("policy", policy_node)
    workflow.add_node("memory_update", memory_update_node)
    
    # Set entry point
    workflow.set_entry_point("preprocess_script")
    
    # Add edges
    workflow.add_edge("preprocess_script", "preprocess_refs")
    workflow.add_edge("preprocess_refs", "planner")
    
    # Main loop edges
    workflow.add_edge("planner", "reviewer")
    workflow.add_edge("reviewer", "variation_mgr")
    workflow.add_edge("variation_mgr", "renderer")
    workflow.add_edge("renderer", "fast_qa")
    workflow.add_edge("fast_qa", "vision_qa")
    workflow.add_edge("vision_qa", "policy")
    workflow.add_edge("policy", "memory_update")
    
    # Conditional routing from memory_update
    workflow.add_conditional_edges(
        "memory_update",
        should_continue_workflow,
        {
            "planner": "planner",
            "renderer": "renderer",
            "end": END
        }
    )
    
    return workflow.compile()


def generate_final_report(state: WorkflowState) -> None:
    """Generate final metrics and report."""
    end_time = datetime.now()
    elapsed = (end_time - state.start_time).total_seconds()
    
    # Count model usage
    model_usage = {}
    for log in state.logs:
        if log.get("model"):
            model = log["model"]
            model_usage[model] = model_usage.get(model, 0) + 1
    
    # Create metrics
    metrics = Metrics(
        run_id=state.trace_id,
        start_time=state.start_time,
        end_time=end_time,
        elapsed_s=elapsed,
        total_tokens=state.total_tokens,
        total_cost_usd=state.total_cost,
        scenes_processed=state.current_scene_idx + 1,
        shots_generated=len(state.accepted_frames),
        variations_created=sum(1 for log in state.logs if log.get("stage") == "variation_mgr"),
        frames_accepted=len(state.accepted_frames),
        frames_rejected=sum(1 for log in state.logs if log.get("stage") == "policy" and log.get("status") == "give_up"),
        retry_attempts=sum(1 for log in state.logs if log.get("stage") == "policy" and log.get("status") == "retry_new"),
        edit_attempts=sum(1 for log in state.logs if log.get("stage") == "policy" and log.get("status") == "retry_edit"),
        accept_rate=len(state.accepted_frames) / max(1, len(state.accepted_frames) + sum(1 for log in state.logs if log.get("stage") == "policy" and log.get("status") == "give_up")),
        models_used=model_usage
    )
    
    # Save metrics
    metrics_path = Path(state.output_dir) / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics.model_dump(), f, indent=2, default=str)
    
    # Generate summary report
    report = f"""
# VC-RAG-SBG Run Report

**Run ID:** {state.trace_id}
**Duration:** {elapsed:.1f} seconds
**Total Cost:** ${state.total_cost:.2f}
**Total Tokens:** {state.total_tokens:,}

## Generation Stats
- Scenes Processed: {metrics.scenes_processed}
- Shots Generated: {metrics.shots_generated}
- Variations Created: {metrics.variations_created}
- Frames Accepted: {metrics.frames_accepted}
- Accept Rate: {metrics.accept_rate:.1%}

## Quality Control
- Retry Attempts: {metrics.retry_attempts}
- Edit Attempts: {metrics.edit_attempts}
- Frames Rejected: {metrics.frames_rejected}

## Model Usage
"""
    
    for model, count in model_usage.items():
        report += f"- {model}: {count} calls\n"
    
    report += f"\n## Output Location\n{state.output_dir}\n"
    
    # Save report
    report_path = Path(state.output_dir) / "report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n{'='*50}")
    print(report)
    print(f"{'='*50}\n")


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="VC-RAG-SBG: Visual-Context-Aware RAG Storyboard Generator")
    parser.add_argument("--script", default="data/script.md", help="Path to script markdown file")
    parser.add_argument("--style", default="data/style.md", help="Path to style markdown file")
    parser.add_argument("--entities", default="data/entities.md", help="Path to entities markdown file")
    parser.add_argument("--refs", default="data/refs", help="Path to reference images directory")
    parser.add_argument("--budget", type=float, help="Budget in USD (overrides config)")
    parser.add_argument("--variations", type=int, help="Number of variations per shot (overrides config)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--no-refs", action="store_true", help="Skip reference image processing")
    
    args = parser.parse_args()
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        sys.exit(1)
    
    # Initialize loader
    loader = Loader(args.config)
    
    # Build config overrides
    config_overrides = {}
    if args.budget:
        config_overrides["budget_usd"] = args.budget
    if args.variations:
        config_overrides["n_variations"] = args.variations
    
    # Initialize workflow state
    try:
        state = loader.initialize_state(
            script_path=args.script,
            style_path=args.style,
            entities_path=args.entities,
            refs_dir=None if args.no_refs else args.refs,
            config_overrides=config_overrides
        )
    except Exception as e:
        print(f"Error initializing: {e}")
        sys.exit(1)
    
    print(f"\nStarting VC-RAG-SBG run...")
    print(f"Output directory: {state.output_dir}")
    print(f"Budget: ${state.budget_usd}")
    print(f"Variations per shot: {state.n_variations}")
    print()
    
    # Build and run workflow
    workflow = build_workflow()
    
    try:
        # Run the workflow
        final_state = workflow.invoke(state, {"recursion_limit": 50})
        
        # Generate final report
        generate_final_report(final_state)
        
        # Save final state
        save_workflow_state(final_state)
        
        print(f"\nWorkflow completed successfully!")
        print(f"Output saved to: {final_state.output_dir}")
        
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        save_workflow_state(state)
        generate_final_report(state)
    except Exception as e:
        print(f"\nError during workflow execution: {e}")
        save_workflow_state(state)
        generate_final_report(state)
        raise


if __name__ == "__main__":
    main() 