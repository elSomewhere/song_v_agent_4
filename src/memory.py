"""Memory service using LanceDB for vector storage and retrieval."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import lancedb
import pyarrow as pa
from datetime import datetime

from src.models import RefMeta, WorkflowState
from src.utils import get_openai_client, call_openai_with_retry, calculate_cost, log_entry


class MemoryService:
    """Manages vector storage and retrieval with LanceDB."""
    
    def __init__(self, state: WorkflowState):
        self.state = state
        self.client = get_openai_client()
        self.db_path = Path(state.output_dir) / "memory" / "lancedb"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize LanceDB
        self.db = lancedb.connect(str(self.db_path))
        
        # Create tables
        self._init_tables()
    
    def _init_tables(self):
        """Initialize LanceDB tables."""
        # Reference images table
        if "references" not in self.db.table_names():
            # Define schema for references
            schema = pa.schema([
                pa.field("frame_id", pa.string()),
                pa.field("category", pa.string()),
                pa.field("entity", pa.string()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("confidence", pa.float32()),
                pa.field("embedding", pa.list_(pa.float32(), 1536)),
                pa.field("thumb_path", pa.string()),
                pa.field("source", pa.string()),
                pa.field("original_path", pa.string()),
                pa.field("timestamp", pa.string())
            ])
            
            # Create empty table
            self.refs_table = self.db.create_table("references", schema=schema)
        else:
            self.refs_table = self.db.open_table("references")
        
        # Generated frames table  
        if "frames" not in self.db.table_names():
            schema = pa.schema([
                pa.field("frame_id", pa.string()),
                pa.field("scene_id", pa.int32()),
                pa.field("shot_id", pa.int32()),
                pa.field("prompt", pa.string()),
                pa.field("negative_prompt", pa.string()),
                pa.field("entities", pa.list_(pa.string())),
                pa.field("embedding", pa.list_(pa.float32(), 1536)),
                pa.field("image_path", pa.string()),
                pa.field("quality_score", pa.float32()),
                pa.field("timestamp", pa.string())
            ])
            
            self.frames_table = self.db.create_table("frames", schema=schema)
        else:
            self.frames_table = self.db.open_table("frames")
    
    def index_references(self, ref_metas: List[RefMeta]) -> None:
        """Index reference images in LanceDB."""
        if not ref_metas:
            return
        
        # Convert to records for LanceDB
        records = []
        for ref in ref_metas:
            record = {
                "frame_id": ref.frame_id,
                "category": ref.category,
                "entity": ref.entity,
                "tags": ref.tags,
                "confidence": float(ref.confidence),
                "embedding": ref.clip_embedding,
                "thumb_path": ref.thumb_path,
                "source": ref.source,
                "original_path": ref.original_path or "",
                "timestamp": ref.timestamp.isoformat()
            }
            records.append(record)
        
        # Add to table
        self.refs_table.add(records)
        
        log_entry(self.state, "memory_index_refs", "success",
                 extra={"count": len(records)})
    
    def index_generated_frame(self, frame_data: Dict[str, Any]) -> None:
        """Index a generated frame in LanceDB."""
        # Generate embedding for the prompt
        embedding = self._generate_embedding(frame_data["prompt"])
        
        record = {
            "frame_id": frame_data["frame_id"],
            "scene_id": frame_data["scene_id"],
            "shot_id": frame_data["shot_id"],
            "prompt": frame_data["prompt"],
            "negative_prompt": frame_data.get("negative_prompt", ""),
            "entities": frame_data.get("entities", []),
            "embedding": embedding,
            "image_path": frame_data["image_path"],
            "quality_score": float(frame_data.get("quality_score", 0.0)),
            "timestamp": datetime.now().isoformat()
        }
        
        self.frames_table.add([record])
        
        log_entry(self.state, "memory_index_frame", "success",
                 extra={"frame_id": frame_data["frame_id"]})
    
    def search_references(self, query: str, entity_filter: Optional[str] = None,
                         category_filter: Optional[str] = None, 
                         limit: int = 5) -> List[Dict[str, Any]]:
        """Search reference images using hybrid search."""
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Build search
        search = self.refs_table.search(query_embedding)
        
        # Apply filters
        filter_conditions = []
        if entity_filter:
            filter_conditions.append(f"entity = '{entity_filter}'")
        if category_filter:
            filter_conditions.append(f"category = '{category_filter}'")
        
        if filter_conditions:
            filter_str = " AND ".join(filter_conditions)
            search = search.where(filter_str)
        
        # Execute search
        results = search.limit(limit).to_list()
        
        log_entry(self.state, "memory_search_refs", "success",
                 extra={"query": query[:50], "results": len(results)})
        
        return results
    
    def search_frames(self, query: str, scene_id: Optional[int] = None,
                     limit: int = 5) -> List[Dict[str, Any]]:
        """Search generated frames."""
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Build search
        search = self.frames_table.search(query_embedding)
        
        # Apply filters
        if scene_id is not None:
            search = search.where(f"scene_id = {scene_id}")
        
        # Execute search
        results = search.limit(limit).to_list()
        
        log_entry(self.state, "memory_search_frames", "success",
                 extra={"query": query[:50], "results": len(results)})
        
        return results
    
    def get_visual_context(self, scene_id: int, shot_id: int, 
                          window_size: int = 4) -> Tuple[List[Dict], List[Dict]]:
        """Get visual context for current shot."""
        # Get nearby frames
        nearby_frames = self._get_nearby_frames(scene_id, shot_id, window_size)
        
        # Get relevant references based on current scene
        current_scene = self.state.scenes[scene_id - 1] if scene_id <= len(self.state.scenes) else None
        relevant_refs = []
        
        if current_scene and current_scene.entities:
            for entity in current_scene.entities:
                refs = self.search_references(
                    query=f"{entity} {current_scene.description or ''}",
                    entity_filter=entity,
                    limit=2
                )
                relevant_refs.extend(refs)
        
        return nearby_frames, relevant_refs
    
    def _get_nearby_frames(self, scene_id: int, shot_id: int, 
                          window_size: int) -> List[Dict[str, Any]]:
        """Get frames near current position."""
        # Query frames around current scene/shot
        all_frames = self.frames_table.to_pandas()
        
        if all_frames.empty:
            return []
        
        # Filter to nearby scenes
        scene_range = range(max(1, scene_id - 1), scene_id + 2)
        nearby = all_frames[all_frames['scene_id'].isin(scene_range)]
        
        # Sort by scene and shot
        nearby = nearby.sort_values(['scene_id', 'shot_id'])
        
        # Convert to list of dicts
        return nearby.to_dict('records')[:window_size]
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate text embedding."""
        model = self.state.config["models"]["embedding_text"]
        
        try:
            response = call_openai_with_retry(
                self.client,
                model=model,
                input=text,
                dimensions=1536  # Specify 1536 dimensions for compatibility
            )
            
            embedding = response.data[0].embedding
            tokens = response.usage.total_tokens
            cost = calculate_cost(model, tokens, 0)
            
            self.state.total_tokens += tokens
            self.state.total_cost += cost
            
            return embedding
            
        except Exception as e:
            # Return zero vector on error
            return [0.0] * 1536
    
    def update_episodic_memory(self, event: Dict[str, Any]) -> None:
        """Update episodic memory with workflow events."""
        self.state.episodic_memory.append({
            "timestamp": datetime.now().isoformat(),
            "event": event
        })
        
        # Keep only recent events
        max_events = 50
        if len(self.state.episodic_memory) > max_events:
            self.state.episodic_memory = self.state.episodic_memory[-max_events:]
    
    def update_visual_memory(self, frame_info: Dict[str, Any]) -> None:
        """Update visual memory with accepted frames."""
        self.state.visual_memory.append({
            "timestamp": datetime.now().isoformat(),
            "frame": frame_info
        })
        
        # Keep only recent frames
        max_frames = 20
        if len(self.state.visual_memory) > max_frames:
            self.state.visual_memory = self.state.visual_memory[-max_frames:]
    
    def export_memory_stats(self) -> Dict[str, Any]:
        """Export memory statistics."""
        refs_count = len(self.refs_table.to_pandas()) if self.refs_table else 0
        frames_count = len(self.frames_table.to_pandas()) if self.frames_table else 0
        
        return {
            "reference_count": refs_count,
            "frame_count": frames_count,
            "episodic_events": len(self.state.episodic_memory),
            "visual_memory_size": len(self.state.visual_memory),
            "db_path": str(self.db_path)
        } 