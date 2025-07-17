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
from src.preprocess import ScriptPreprocessor, ReferencePreprocessor, EntitiesPreprocessor, EntitiesEnricher
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
    
    # Dump parsed scenes to disk for user inspection
    try:
        scenes_dump = [s.model_dump() for s in state.scenes]
        dump_path = Path(state.output_dir) / "scenes_parsed.json"
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(scenes_dump, f, indent=2, ensure_ascii=False)
        print(f"[PreprocessScript] Parsed scenes written to {dump_path.relative_to(state.output_dir)}")
    except Exception as e:
        print(f"[PreprocessScript] Warning: could not write scenes dump: {e}")
    
    # ------------------------------------------------------------------
    # Index canonical entity descriptions (once per run)
    # ------------------------------------------------------------------
    if state.entities_dict:
        state.get_memory_service().index_canonical_entities(state.entities_dict)
    
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
    memory = state.get_memory_service()  # Use singleton memory service
    memory.index_references(ref_metas)
    
    log_entry(state, "preprocess_refs", "success",
             extra={"refs_processed": len(ref_metas)})
    
    return state


def enrich_entities_node(state: WorkflowState) -> WorkflowState:
    """Merge textual entities with visual reference metadata."""

    # Require entities_dict and ref_index to proceed
    if not state.entities_dict or not state.ref_index:
        log_entry(state, "enrich_entities", "skipped",
                 extra={"entities": bool(state.entities_dict), "refs": bool(state.ref_index)})
        return state

    preprocessor = EntitiesEnricher(state)
    merged = preprocessor.enrich()

    if merged and isinstance(merged, dict):
        state.entities_dict = merged

        # Dump for inspection
        try:
            dump_path = Path(state.output_dir) / "entities_enriched.json"
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            print(f"[EnrichEntities] Enriched entities written to {dump_path.relative_to(state.output_dir)}")
        except Exception as e:
            print(f"[EnrichEntities] Warning: couldn't write enriched entities: {e}")

        # Re-index in memory
        state.get_memory_service().index_canonical_entities(merged)

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


def preprocess_entities_node(state: WorkflowState) -> WorkflowState:
    """Parse entities.md into structured dict using GPT if needed."""
    # Skip if we already have parsed entities (JSON was present)
    if state.entities_dict:
        log_entry(state, "preprocess_entities", "skipped")
        return state

    preprocess_setting = state.config.get("preprocess", {}).get("entities", "auto")
    if preprocess_setting != "auto":
        # In heuristic or skip mode, we do not invoke GPT. If JSON wasn't parsed, leave empty.
        log_entry(state, "preprocess_entities", "skipped")
        return state

    # Load entities.md text
    with open(state.entities_path, "r", encoding="utf-8") as f:
        entities_md = f.read()

    preprocessor = EntitiesPreprocessor(state)
    entities_dict = preprocessor.parse_entities(entities_md)
    state.entities_dict = entities_dict or {}

    # Dump entities dict to disk for inspection
    if state.entities_dict:
        try:
            dump_path = Path(state.output_dir) / "entities_parsed.json"
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(state.entities_dict, f, indent=2, ensure_ascii=False)
            print(f"[PreprocessEntities] Parsed entities written to {dump_path.relative_to(state.output_dir)}")
        except Exception as e:
            print(f"[PreprocessEntities] Warning: could not write entities dump: {e}")

    # Index canonical entities into memory for retrieval
    if state.entities_dict:
        state.get_memory_service().index_canonical_entities(state.entities_dict)

    return state


def build_workflow() -> StateGraph:
    """Build the LangGraph workflow exactly as specified in section 5."""
    
    # Create the graph
    graph = StateGraph(WorkflowState)
    
    # Bootstrap nodes
    graph.add_node("preprocess_script", preprocess_script_node)
    graph.add_node("preprocess_entities", preprocess_entities_node)
    graph.add_node("preprocess_refs", preprocess_refs_node)
    graph.add_node("enrich_entities", enrich_entities_node)
    
    # Set entry point
    graph.set_entry_point("preprocess_script")
    
    # Bootstrap edges
    graph.add_edge("preprocess_script", "preprocess_entities")
    graph.add_edge("preprocess_entities", "preprocess_refs")
    graph.add_edge("preprocess_refs", "enrich_entities")
    graph.add_edge("enrich_entities", "planner")
    
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
    
    # Export memory data to output directory (if accessible)
    # Handle both WorkflowState objects and AddableValuesDict from LangGraph
    try:
        if hasattr(state, 'get_memory_service'):
            memory = state.get_memory_service()
            output_dir = state.output_dir
        else:
            # For AddableValuesDict, create memory service directly
            from src.memory import MemoryService
            # Create a minimal state-like object for MemoryService
            class StateAdapter:
                def __init__(self, state_dict):
                    self.__dict__.update(state_dict)
            
            adapter = StateAdapter(dict(state))
            memory = MemoryService(adapter)
            output_dir = state["output_dir"]
        
        memory_output_dir = Path(output_dir) / "memory"
        memory.export_memory_to_output(str(memory_output_dir))
    except Exception as e:
        print(f"Warning: Could not export memory data: {e}")
        # Get output_dir for report generation
        if hasattr(state, 'output_dir'):
            output_dir = state.output_dir
        else:
            output_dir = state["output_dir"]
    
    # Get metrics for report
    metrics = collector.collect_from_logs()
    
    # Generate console report
    duration = metrics.elapsed_s
    cost = metrics.total_cost_usd
    tokens = metrics.total_tokens
    
    print("\n" + "="*50)
    print("\n# VC-RAG-SBG Run Report")
    print(f"\n**Run ID:** {metrics.run_id}")
    print(f"**Duration:** {duration:.1f} seconds")
    print(f"**Total Cost:** ${cost:.2f}")
    print(f"**Total Tokens:** {tokens:,}")
    
    print(f"\n## Generation Stats")
    print(f"- Scenes Processed: {metrics.scenes_processed}")
    print(f"- Shots Generated: {metrics.shots_generated}")
    print(f"- Variations Created: {metrics.variations_created}")
    print(f"- Frames Accepted: {metrics.frames_accepted}")
    print(f"- Accept Rate: {metrics.accept_rate:.1%}")
    
    print(f"\n## Quality Control")
    print(f"- Retry Attempts: {metrics.retry_attempts}")
    print(f"- Edit Attempts: {metrics.edit_attempts}")
    print(f"- Frames Rejected: {metrics.frames_rejected}")
    
    print(f"\n## Model Usage")
    for model, count in metrics.models_used.items():
        print(f"- {model}: {count} calls")
    
    print(f"\n## Output Location")
    print(f"{output_dir}")
    
    print("\n" + "="*50)


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
    parser.add_argument("--ai-preprocess-entities", action="store_true", help="Use AI to preprocess entities")
    parser.add_argument("--enable-style-embedding", action="store_true", help="Enable visual style embedding for improved reference retrieval")
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
        "style_embedding_enabled": args.enable_style_embedding,
        "preprocess": {
            "script": "auto" if args.ai_preprocess_script else "heuristic",
            "refs": "auto" if args.ai_preprocess_refs else "skip",
            "entities": "auto" if args.ai_preprocess_entities else "heuristic",
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
    print(f"Style embedding: {'enabled' if args.enable_style_embedding else 'disabled'}")
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