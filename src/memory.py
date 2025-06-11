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
        # Per spec, LanceDB lives under .cache/lancedb/
        self.db_path = Path(".cache") / "lancedb"
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize LanceDB
        self.db = lancedb.connect(str(self.db_path))
        
        # Create tables
        self._init_tables()
        
        # Alias frames_table to visual_ctx_table for backward compatibility
        self.frames_table = self.visual_ctx_table
    
    def _init_tables(self):
        """Initialize LanceDB tables according to spec."""
        # canonical_text table
        if "canonical_text" not in self.db.table_names():
            schema = pa.schema([
                pa.field("chunk_id", pa.string()),
                pa.field("chunk_text", pa.string()),
                pa.field("text_embedding", pa.list_(pa.float32(), 1536))
            ])
            self.canonical_text_table = self.db.create_table("canonical_text", schema=schema)
        else:
            self.canonical_text_table = self.db.open_table("canonical_text")
        
        # episodic_text table
        if "episodic_text" not in self.db.table_names():
            schema = pa.schema([
                pa.field("scene_id", pa.int32()),
                pa.field("shot_id", pa.int32()),
                pa.field("summary", pa.string()),
                pa.field("text_embedding", pa.list_(pa.float32(), 1536)),
                pa.field("entities", pa.list_(pa.string())),
                pa.field("timestamp", pa.string()),
                pa.field("quality_score", pa.float32())
            ])
            self.episodic_text_table = self.db.create_table("episodic_text", schema=schema)
        else:
            self.episodic_text_table = self.db.open_table("episodic_text")
        
        # visual_ctx table (for reference images)
        if "visual_ctx" not in self.db.table_names():
            schema = pa.schema([
                pa.field("frame_id", pa.string()),
                pa.field("scene_id", pa.int32()),
                pa.field("clip_embedding", pa.list_(pa.float32(), 1536)),
                pa.field("thumb_path", pa.string()),
                pa.field("trace_id", pa.string()),
                pa.field("category", pa.string()),
                pa.field("entity", pa.string()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("source", pa.string()),
                pa.field("confidence", pa.float32()),
                pa.field("prompt", pa.string()),
                pa.field("shot_id", pa.int32()),
                pa.field("original_path", pa.string())
            ])
            self.visual_ctx_table = self.db.create_table("visual_ctx", schema=schema)
        else:
            self.visual_ctx_table = self.db.open_table("visual_ctx")
        
        # failures table
        if "failures" not in self.db.table_names():
            schema = pa.schema([
                pa.field("frame_id", pa.string()),
                pa.field("err_code", pa.string()),
                pa.field("neg_prompt_token", pa.string()),
                pa.field("timestamp", pa.string())
            ])
            self.failures_table = self.db.create_table("failures", schema=schema)
        else:
            self.failures_table = self.db.open_table("failures")
    
    def index_references(self, ref_metas: List[RefMeta]) -> None:
        """Index reference images in visual_ctx table."""
        if not ref_metas:
            return
        
        # Convert to records for LanceDB
        records = []
        for ref in ref_metas:
            record = {
                "frame_id": ref.frame_id,
                "scene_id": -1,  # -1 for reference images
                "clip_embedding": ref.clip_embedding,
                "thumb_path": ref.thumb_path,
                "trace_id": self.state.trace_id,
                "category": ref.category,
                "entity": ref.entity,
                "tags": ref.tags,
                "source": ref.source,
                "confidence": float(ref.confidence),
                "original_path": ref.original_path or "",
                "prompt": "",
                "shot_id": -1  # -1 for reference images
            }
            records.append(record)
        
        # Add to table
        self.visual_ctx_table.add(records)
        
        log_entry(self.state, "memory_index_refs", "success",
                 extra={"count": len(records)})
    
    def index_generated_frame(self, frame_data: Dict[str, Any]) -> None:
        """Index a generated frame in episodic_text and visual_ctx tables."""
        # Generate embedding for the prompt
        embedding = self._generate_embedding(frame_data["prompt"])
        
        # Add to episodic_text table
        episodic_record = {
            "scene_id": frame_data["scene_id"],
            "shot_id": frame_data["shot_id"],
            "summary": frame_data["prompt"],
            "text_embedding": embedding,
            "entities": frame_data.get("entities", []),
            "timestamp": datetime.now().isoformat(),
            "quality_score": float(frame_data.get("quality_score", 0.0))
        }
        self.episodic_text_table.add([episodic_record])
        
        # Add to visual_ctx table
        visual_record = {
            "frame_id": frame_data["frame_id"],
            "scene_id": frame_data["scene_id"],
            "clip_embedding": embedding,  # Using text embedding as proxy
            "thumb_path": frame_data.get("thumb_path", ""),
            "trace_id": self.state.trace_id,
            "category": "generated",
            "entity": "generated_frame",
            "tags": frame_data.get("tags", []),
            "source": "generated",
            "confidence": float(frame_data.get("quality_score", 0.0)),
            "prompt": frame_data["prompt"],
            "shot_id": frame_data["shot_id"],
            "original_path": frame_data.get("image_path", ""),
        }
        self.visual_ctx_table.add([visual_record])
        
        log_entry(self.state, "memory_index_frame", "success",
                 extra={"frame_id": frame_data["frame_id"]})
    
    def hybrid_retrieve(self, scene_embed: List[float], entities: List[str], shot_id: int,
                       k_txt: int = 5, k_img: int = 3) -> Tuple[List[Any], List[Any]]:
        """Hybrid retrieval as specified in section 3.4."""
        # Text search
        txt_hits = self.episodic_text_table.search(scene_embed, "text_embedding").limit(k_txt * 3)
        
        # Image search  
        img_hits = self.visual_ctx_table.search(scene_embed, "clip_embedding").limit(k_img * 3)
        
        # Convert to pandas for scoring
        txt_df = txt_hits.to_pandas() if txt_hits else None
        img_df = img_hits.to_pandas() if img_hits else None
        
        def score_txt(r):
            """Score text results."""
            sem = 1 - r._distance
            ent = self._jaccard(set(r.entities) if hasattr(r, 'entities') else set(), set(entities))
            rec = 1 / (1 + abs(shot_id - r.shot_id) / 100) if hasattr(r, 'shot_id') else 0.5
            return 0.6 * sem + 0.3 * ent + 0.1 * rec
        
        def score_img(r):
            """Score image results."""
            sim = 1 - r._distance
            conf = 0.5 + 0.5 * r.confidence if hasattr(r, 'confidence') else 0.75
            boost = 1.2 if hasattr(r, 'entity') and r.entity in entities else 1.0
            return sim * conf * boost
        
        # Score and sort
        txt_results = []
        if txt_df is not None and not txt_df.empty:
            txt_results = sorted(txt_df.itertuples(), key=score_txt, reverse=True)[:k_txt]
        
        img_results = []
        if img_df is not None and not img_df.empty:
            img_results = sorted(img_df.itertuples(), key=score_img, reverse=True)[:k_img]
        
        return txt_results, img_results
    
    def _jaccard(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def record_failure(self, frame_id: str, err_code: str, neg_prompt_token: str) -> None:
        """Record a failure in the failures table."""
        record = {
            "frame_id": frame_id,
            "err_code": err_code,
            "neg_prompt_token": neg_prompt_token,
            "timestamp": datetime.now().isoformat()
        }
        self.failures_table.add([record])
        log_entry(self.state, "memory_record_failure", "success",
                 extra={"frame_id": frame_id, "err_code": err_code})
    
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
        
        # Sort by scene and shot if available
        if 'shot_id' in nearby.columns:
            nearby = nearby.sort_values(['scene_id', 'shot_id'])
        else:
            nearby = nearby.sort_values(['scene_id'])
        
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
        canonical_count = len(self.canonical_text_table.to_pandas()) if hasattr(self, 'canonical_text_table') else 0
        episodic_count = len(self.episodic_text_table.to_pandas()) if hasattr(self, 'episodic_text_table') else 0
        visual_count = len(self.visual_ctx_table.to_pandas()) if hasattr(self, 'visual_ctx_table') else 0
        failures_count = len(self.failures_table.to_pandas()) if hasattr(self, 'failures_table') else 0
        
        return {
            "canonical_text_count": canonical_count,
            "episodic_text_count": episodic_count,
            "visual_ctx_count": visual_count,
            "failures_count": failures_count,
            "episodic_events": len(self.state.episodic_memory),
            "visual_memory_size": len(self.state.visual_memory),
            "db_path": str(self.db_path)
        }
    
    # ---------------------------------------------------------------------
    # Public helpers expected by other modules
    # ---------------------------------------------------------------------

    def search_references(
        self,
        query: str,
        entity_filter: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search visual_ctx table for reference images.

        There are two usage patterns in the code base:

        1.  Named query + optional entity_filter (from _get_visual_context)
        2.  Direct lookup by `frame_id` (from renderer._get_reference_images)

        The method supports both – if *query* resembles a UUID (has hyphens) we
        do a simple equality filter on frame_id; otherwise we treat it as free
        text, embed it, and perform a vector similarity search.
        """

        # Fast path: exact lookup by frame_id
        if "-" in query and len(query) >= 8:
            df = self.visual_ctx_table.to_pandas()
            hits = df[df["frame_id"] == query]
            if entity_filter:
                hits = hits[hits["entity"] == entity_filter]
            return hits.to_dict("records")[:limit]

        # Otherwise do embedding search
        embed = self._generate_embedding(query)
        search_res = self.visual_ctx_table.search(embed, "clip_embedding").limit(limit * 3)

        if not search_res:
            return []

        df = search_res.to_pandas()

        if entity_filter:
            df = df[df["entity"] == entity_filter]

        # Basic ranking by similarity * confidence
        def _score(row):
            sim = 1 - row._distance if hasattr(row, "_distance") else 0.5
            conf = row.confidence if "confidence" in row else 0.5
            return sim * (0.5 + 0.5 * conf)

        ranked = sorted(df.itertuples(), key=_score, reverse=True)[:limit]
        return [r._asdict() for r in ranked] 