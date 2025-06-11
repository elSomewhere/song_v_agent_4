#!/usr/bin/env python3
"""Main entry point for the VC-RAG-SBG system."""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from src.loader import Loader
from src.preprocess import ScriptPreprocessor, ReferencePreprocessor
from src.memory import MemoryService
from src.models import WorkflowState, Metrics
from src.utils import log_entry, save_workflow_state, ensure_directory
from src.metrics import MetricsCollector

# Import workflow nodes
from src.nodes.planner import planner_node
from src.nodes.reviewer import reviewer_node
from src.nodes.variation_mgr import variation_mgr_node
from src.nodes.renderer import renderer_node
from src.nodes.fast_qa import fast_qa_node
from src.nodes.vision_qa import vision_qa_node
from src.nodes.policy import policy_node
from src.nodes.memory_update import memory_update_node
from src.nodes.workflow_controller import workflow_controller_node


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


def should_sample_vision_qa(state: WorkflowState) -> str:
    """Conditional edge after fast_qa to either vision_qa or policy."""
    return "vision_qa" if state.fast_qa_flag else "policy"


def should_retry_or_update(state: WorkflowState) -> str:
    """Conditional edge after policy to either retry or update memory."""
    if state.policy_action in {"retry_new", "retry_edit"}:
        return "renderer"
    else:  # accept or give_up
        return "memory_update"


def should_continue_workflow(state: WorkflowState) -> str:
    """Conditional edge after memory_update to continue or end."""
    if state.workflow_complete:
        return "end"
    else:
        return "workflow_controller"


def should_controller_to_planner(state: WorkflowState) -> str:
    """Conditional edge after workflow_controller."""
    if state.workflow_complete:
        return "end"
    else:
        return "planner"


def build_workflow() -> StateGraph:
    """Build the LangGraph workflow exactly as specified in section 5."""
    
    # Create the graph
    graph = StateGraph(WorkflowState)
    
    # Bootstrap nodes
    graph.add_node("preprocess_script", preprocess_script_node)
    graph.add_node("preprocess_refs", preprocess_refs_node)
    
    # Set entry point
    graph.set_entry_point("preprocess_script")
    
    # Bootstrap edges
    graph.add_edge("preprocess_script", "preprocess_refs")
    graph.add_edge("preprocess_refs", "planner")
    
    # Main loop nodes - exactly as specified
    for name, node in [
        ("planner", planner_node),
        ("reviewer", reviewer_node),
        ("variation_mgr", variation_mgr_node),
        ("renderer", renderer_node),
        ("fast_qa", fast_qa_node),
        ("vision_qa", vision_qa_node),
        ("policy", policy_node),
        ("memory_update", memory_update_node),
        ("workflow_controller", workflow_controller_node)
    ]:
        graph.add_node(name, node)
    
    # Main loop edges - exactly as specified
    graph.add_edge("planner", "reviewer")
    graph.add_edge("reviewer", "variation_mgr")
    graph.add_edge("variation_mgr", "renderer")
    graph.add_edge("renderer", "fast_qa")
    
    # Conditional edges as specified
    graph.add_conditional_edges("fast_qa", should_sample_vision_qa)
    graph.add_edge("vision_qa", "policy")
    graph.add_conditional_edges("policy", should_retry_or_update)
    
    # Memory update continues to workflow controller
    graph.add_conditional_edges("memory_update", should_continue_workflow)
    
    # Workflow controller back to planner or END
    graph.add_conditional_edges("workflow_controller", should_controller_to_planner)
    
    return graph.compile()


def generate_final_report(state: WorkflowState) -> None:
    """Generate final metrics and report using MetricsCollector."""
    # Use metrics collector
    collector = MetricsCollector(state)
    
    # Save metrics.json
    metrics_path = collector.save_metrics()
    
    # Get metrics for report
    metrics = collector.collect_from_logs()
    
    # Generate summary report
    report = f"""
# VC-RAG-SBG Run Report

**Run ID:** {state.trace_id}
**Duration:** {metrics.elapsed_s:.1f} seconds
**Total Cost:** ${metrics.total_cost_usd:.2f}
**Total Tokens:** {metrics.total_tokens:,}

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
    
    for model, count in metrics.models_used.items():
        report += f"- {model}: {count} calls\n"
    
    report += f"\n## Output Location\n{state.output_dir}\n"
    
    # Save report
    report_path = Path(state.output_dir) / "report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Append detailed metrics
    collector.append_to_report(report_path)
    
    print(f"\n{'='*50}")
    print(report)
    print(f"{'='*50}\n")


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="VC-RAG-SBG: Visual-Context-Aware RAG Storyboard Generator")
    parser.add_argument("--data", required=True, help="Path to data directory")
    parser.add_argument("--out", required=True, help="Path to output directory")
    parser.add_argument("--n-variations", type=int, default=3, help="Number of variations per shot")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum retry attempts")
    parser.add_argument("--budget-usd", type=float, default=35, help="Budget in USD")
    parser.add_argument("--ai-preprocess-script", action="store_true", help="Use AI to preprocess script")
    parser.add_argument("--ai-preprocess-refs", action="store_true", help="Use AI to preprocess references")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Initialize loader
    loader = Loader(args.config)
    
    # Build paths
    data_path = Path(args.data)
    script_path = data_path / "script.md"
    style_path = data_path / "style.md"
    entities_path = data_path / "entities.md"
    refs_dir = data_path / "refs" if (data_path / "refs").exists() else None
    
    # Build config overrides
    config_overrides = {
        "budget_usd": args.budget_usd,
        "n_variations": args.n_variations,
        "max_retries": args.max_retries,
        "preprocess": {
            "script": "auto" if args.ai_preprocess_script else "heuristic",
            "refs": "auto" if args.ai_preprocess_refs else "skip"
        }
    }
    
    # Initialize workflow state
    try:
        state = loader.initialize_state(
            script_path=str(script_path),
            style_path=str(style_path),
            entities_path=str(entities_path),
            refs_dir=str(refs_dir) if refs_dir else None,
            output_base_dir=args.out,
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
        # Run the workflow with increased recursion limit for multi-scene runs
        final_state = workflow.invoke(state, {"recursion_limit": 1000})
        
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