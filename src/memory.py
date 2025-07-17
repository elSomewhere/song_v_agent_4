"""Memory service using LanceDB for vector storage and retrieval."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import lancedb
import pyarrow as pa
from datetime import datetime

from src.models import RefMeta, WorkflowState
from src.utils import (
    get_openai_client,
    call_openai_with_retry,
    calculate_cost,
    log_entry,
    COST_PER_1K_TOKENS,
    parse_json_response,
    get_style_embedding,
    load_image_as_base64,
)


class MemoryService:
    """Manages vector storage and retrieval with LanceDB."""
    
    def __init__(self, state: WorkflowState):
        self.state = state
        self.client = get_openai_client()
        # Per spec, LanceDB lives under .cache/lancedb/
        self.db_path = Path(".cache") / "lancedb"
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Configure embedding dimension from config
        self.embedding_dim = state.config.get("embedding_dimension", 1536)
        self.style_embedding_dim = state.config.get("style_embedding_dimension", 1024)
        
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
                pa.field("text_embedding", pa.list_(pa.float32(), self.embedding_dim))
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
                pa.field("text_embedding", pa.list_(pa.float32(), self.embedding_dim)),
                pa.field("entities", pa.list_(pa.string())),
                pa.field("timestamp", pa.string()),
                pa.field("quality_score", pa.float32())
            ])
            self.episodic_text_table = self.db.create_table("episodic_text", schema=schema)
        else:
            self.episodic_text_table = self.db.open_table("episodic_text")
        
        # visual_ctx table (renamed from frames for clarity)
        if "visual_ctx" not in self.db.table_names():
            schema = pa.schema([
                pa.field("frame_id", pa.string()),
                pa.field("scene_id", pa.int32()),
                pa.field("shot_id", pa.int32()),
                pa.field("clip_embedding", pa.list_(pa.float32(), self.embedding_dim)),
                pa.field("style_embedding", pa.list_(pa.float32(), self.style_embedding_dim)),
                pa.field("thumb_path", pa.string()),
                pa.field("original_path", pa.string()),
                pa.field("trace_id", pa.string()),
                pa.field("category", pa.string()),
                pa.field("entity", pa.string()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("source", pa.string()),
                pa.field("confidence", pa.float32()),
                pa.field("prompt", pa.string())
            ])
            self.visual_ctx_table = self.db.create_table("visual_ctx", schema=schema)
        else:
            self.visual_ctx_table = self.db.open_table("visual_ctx")
            # Check if new columns exist, add migration handling for missing fields
            existing_fields = set(self.visual_ctx_table.schema.names)
            required_fields = {"style_embedding", "shot_id", "prompt"}
            missing_fields = required_fields - existing_fields
            
            if missing_fields:
                log_entry(self.state, "memory_migration", "missing_fields_detected", 
                         extra={"missing": list(missing_fields)})
                # For LanceDB, we'll handle missing fields during insertion by providing defaults
        
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
    
    def _ensure_visual_ctx_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure record has all required fields for visual_ctx table with defaults."""
        # Get existing schema fields
        existing_fields = set(self.visual_ctx_table.schema.names)
        
        # Add missing fields with appropriate defaults
        defaults = {
            "shot_id": -1,
            "prompt": "",
            "style_embedding": [0.0] * self.style_embedding_dim
        }
        
        for field, default_value in defaults.items():
            if field not in existing_fields:
                continue  # Skip if table doesn't have this field yet
            if field not in record:
                record[field] = default_value
                
        return record
    
    def index_references(self, ref_metas: List[RefMeta]) -> None:
        """Index reference images in visual_ctx table."""
        if not ref_metas:
            return
        
        # Convert to records for LanceDB
        records = []
        for ref in ref_metas:
            # Compute style embedding if enabled
            style_embedding = None
            if self.state.config.get("style_embedding_enabled", False):
                try:
                    # Load image as base64 for style embedding
                    if ref.original_path and Path(ref.original_path).exists():
                        image_b64 = load_image_as_base64(ref.original_path)
                        style_embedding = get_style_embedding(image_b64, self.state.config, self.state)
                    else:
                        # Fallback to zero vector if image not available
                        style_embedding = [0.0] * self.style_embedding_dim
                except Exception as e:
                    log_entry(self.state, "memory_index_refs", "style_embedding_error", 
                             error=str(e), extra={"frame_id": ref.frame_id})
                    style_embedding = [0.0] * self.style_embedding_dim
            else:
                style_embedding = [0.0] * self.style_embedding_dim
            
            record = {
                "frame_id": ref.frame_id,
                "scene_id": -1,  # -1 for reference images
                "shot_id": -1,  # -1 for reference images
                "clip_embedding": ref.clip_embedding,
                "style_embedding": style_embedding,
                "thumb_path": ref.thumb_path,
                "original_path": ref.original_path or "",
                "trace_id": self.state.trace_id,
                "category": ref.category,
                "entity": ref.entity,
                "tags": ref.tags,
                "source": ref.source,
                "confidence": float(ref.confidence),
                "prompt": ""
            }
            record = self._ensure_visual_ctx_fields(record)
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
        
        # Compute style embedding if enabled
        style_embedding = None
        if self.state.config.get("style_embedding_enabled", False):
            try:
                # Load image as base64 for style embedding
                image_path = frame_data.get("image_path")
                if image_path and Path(image_path).exists():
                    image_b64 = load_image_as_base64(image_path)
                    style_embedding = get_style_embedding(image_b64, self.state.config, self.state)
                else:
                    # Fallback to zero vector if image not available
                    style_embedding = [0.0] * self.style_embedding_dim
            except Exception as e:
                log_entry(self.state, "memory_index_frame", "style_embedding_error", 
                         error=str(e), extra={"frame_id": frame_data["frame_id"]})
                style_embedding = [0.0] * self.style_embedding_dim
        else:
            style_embedding = [0.0] * self.style_embedding_dim
        
        # Add to visual_ctx table
        visual_record = {
            "frame_id": frame_data["frame_id"],
            "scene_id": frame_data["scene_id"],
            "shot_id": frame_data["shot_id"],
            "clip_embedding": embedding,  # Using text embedding as proxy
            "style_embedding": style_embedding,
            "thumb_path": frame_data.get("thumb_path", ""),
            "original_path": frame_data.get("image_path", ""),
            "trace_id": self.state.trace_id,
            "category": "generated",
            "entity": "generated_frame",
            "tags": frame_data.get("tags", []),
            "source": "generated",
            "confidence": float(frame_data.get("quality_score", 0.0)),
            "prompt": frame_data["prompt"]
        }
        visual_record = self._ensure_visual_ctx_fields(visual_record)
        self.visual_ctx_table.add([visual_record])
        
        log_entry(self.state, "memory_index_frame", "success",
                 extra={"frame_id": frame_data["frame_id"]})
    
    def hybrid_retrieve(self, scene_embed: List[float], entities: List[str], shot_id: int,
                       k_txt: int = 5, k_img: int = 3) -> Tuple[List[Any], List[Any]]:
        """Hybrid retrieval that applies a cheap LLM metadata-only re-rank.

        We first pull an *oversampled* set of ANN hits (3× the desired counts)
        for text and image tables, concatenate them, then ask the LLM to
        provide a relevance score purely from the metadata. Finally we split
        the ranked list back into text vs. image hits so that downstream
        components receive the expected two-tuple output.
        """

        print(f"[Memory] hybrid_retrieve called with entities: {entities}")

        # ------------------------------------------------------------------
        # Initial ANN search with optional style embedding fusion
        # ------------------------------------------------------------------
        try:
            # Increase oversampling for better results, ensure minimum limit of 1
            txt_hits = self.episodic_text_table.search(scene_embed, "text_embedding").limit(max(1, k_txt * 5))
            img_hits = self.visual_ctx_table.search(scene_embed, "clip_embedding").limit(max(1, k_img * 5))

            txt_results = list(txt_hits.to_pandas().itertuples()) if txt_hits is not None else []
            img_results = list(img_hits.to_pandas().itertuples()) if img_hits is not None else []
            
            print(f"[Memory] Initial content search: {len(txt_results)} text hits, {len(img_results)} image hits")
            
            # Filter image results to reference images only (scene_id = -1)
            img_results = [r for r in img_results if getattr(r, 'scene_id', 0) == -1]
            print(f"[Memory] After filtering to references: {len(img_results)} image hits")
            
            # Style embedding search if enabled
            if self.state.config.get("style_embedding_enabled", False) and len(img_results) > 0:
                try:
                    # Generate style embedding for query (use average of existing reference embeddings as proxy)
                    # In a real implementation, we might generate this from current scene context
                    query_style_embed = self._get_query_style_embedding(entities)
                    
                    if query_style_embed is not None:
                        style_hits = self.visual_ctx_table.search(query_style_embed, "style_embedding").limit(max(1, k_img * 5))
                        style_results = list(style_hits.to_pandas().itertuples()) if style_hits is not None else []
                        style_results = [r for r in style_results if getattr(r, 'scene_id', 0) == -1]
                        
                        print(f"[Memory] Style search: {len(style_results)} style hits")
                        
                        # Fuse content and style results using weighted rank fusion
                        img_results = self._fuse_content_style_results(img_results, style_results, k_img)
                        print(f"[Memory] After style fusion: {len(img_results)} fused results")
                        
                except Exception as e:
                    print(f"[Memory] Error in style embedding search: {e}")
                    # Continue with content-only results
            
        except Exception as e:
            print(f"[Memory] Error in initial search: {e}")
            txt_results = []
            img_results = []

        # ------------------------------------------------------------------
        # Fallback search by entity names if no good results
        # ------------------------------------------------------------------
        if len(img_results) < k_img and entities:
            print(f"[Memory] Insufficient image results ({len(img_results)} < {k_img}), trying entity-based search")
            for entity in entities:
                try:
                    entity_embed = self._generate_embedding(entity)
                    entity_hits = self.visual_ctx_table.search(entity_embed, "clip_embedding").limit(max(1, k_img * 2))
                    if entity_hits:
                        entity_results = list(entity_hits.to_pandas().itertuples())
                        # Filter to reference images
                        entity_results = [r for r in entity_results if getattr(r, 'scene_id', 0) == -1]
                        img_results.extend(entity_results)
                        print(f"[Memory] Added {len(entity_results)} results for entity '{entity}'")
                except Exception as e:
                    print(f"[Memory] Error searching for entity '{entity}': {e}")

        # Remove duplicates and limit results
        seen_ids = set()
        unique_img_results = []
        for r in img_results:
            frame_id = getattr(r, 'frame_id', None)
            if frame_id and frame_id not in seen_ids:
                unique_img_results.append(r)
                seen_ids.add(frame_id)
                if len(unique_img_results) >= k_img:
                    break
        
        img_results = unique_img_results
        print(f"[Memory] After deduplication: {len(img_results)} image results")

        # ------------------------------------------------------------------
        # Metadata-only LLM re-rank
        # ------------------------------------------------------------------
        candidates = txt_results + img_results

        if candidates and len(candidates) > 0:
            shot_desc = f"Shot {shot_id} | entities: {', '.join(entities)}"
            try:
                candidates = self._text_rerank(shot_desc, candidates, top_k=k_txt + k_img)
                print(f"[Memory] Reranked to {len(candidates)} candidates")
            except Exception as e:
                print(f"[Memory] Error in reranking: {e}")
                # Continue with original candidates if reranking fails

        # Split back into their respective modalities while preserving order
        txt_ranked = [c for c in candidates if hasattr(c, "chunk_text")][:k_txt]
        img_ranked = [c for c in candidates if hasattr(c, "thumb_path")][:k_img]

        print(f"[Memory] Final hybrid_retrieve result: {len(txt_ranked)} text, {len(img_ranked)} images")
        
        return txt_ranked, img_ranked
    
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
        
        # Debug logging
        if current_scene:
            print(f"[Memory] Scene {scene_id} entities: {current_scene.entities}")
        else:
            print(f"[Memory] No scene found for scene_id {scene_id}")
        
        # Strategy 1: Search by scene entities with improved matching
        if current_scene and current_scene.entities:
            for entity in current_scene.entities:
                # Try exact entity name first
                refs = self.search_references(
                    query=entity,
                    entity_filter=entity,
                    limit=2
                )
                if refs:
                    print(f"[Memory] Found {len(refs)} refs for entity '{entity}' (exact match)")
                    relevant_refs.extend(refs)
                else:
                    # Try without entity filter but with entity in query
                    refs = self.search_references(
                        query=entity,
                        entity_filter=None,
                        limit=2
                    )
                    if refs:
                        print(f"[Memory] Found {len(refs)} refs for entity '{entity}' (query match)")
                        relevant_refs.extend(refs)
        
        # Strategy 2: Search by scene description if available and not enough refs
        if len(relevant_refs) < 2 and current_scene and current_scene.description:
            refs = self.search_references(
                query=current_scene.description,
                entity_filter=None,
                limit=3
            )
            if refs:
                print(f"[Memory] Found {len(refs)} refs for scene description")
                relevant_refs.extend(refs)
        
        # Strategy 3: Try key entities from parsed entities (derived from state)
        if len(relevant_refs) < 2 and self.state.entities_dict:
            # Derive entities from state.entities_dict.keys() instead of hard-coded list
            key_entities = list(self.state.entities_dict.keys())[:5]  # Limit to first 5 entities
            for entity in key_entities:
                if len(relevant_refs) >= 3:  # Limit total refs
                    break
                refs = self.search_references(
                    query=entity,
                    entity_filter=None,
                    limit=1
                )
                if refs:
                    print(f"[Memory] Found {len(refs)} refs for key entity '{entity}'")
                    relevant_refs.extend(refs)
        
        # Remove duplicates by frame_id
        seen_ids = set()
        unique_refs = []
        for ref in relevant_refs:
            if ref.get("frame_id") not in seen_ids:
                unique_refs.append(ref)
                seen_ids.add(ref.get("frame_id"))
        
        print(f"[Memory] Final result: {len(unique_refs)} relevant references for Scene {scene_id} Shot {shot_id}")
        
        return nearby_frames, unique_refs
    
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
        """Generate text embedding with configurable dimension."""
        if not text or not text.strip():
            return [0.0] * self.embedding_dim
        
        try:
            model = self.state.config["models"]["embedding_text"]
            # Only pass dimensions parameter for models that support it
            kwargs = {
                "model": model,
                "input": text
            }
            if model == "text-embedding-3-large" or model == "text-embedding-3-small":
                kwargs["dimensions"] = self.embedding_dim
                
            response = call_openai_with_retry(self.client, **kwargs)
            
            embedding = response.data[0].embedding
            
            # Verify embedding dimension matches configuration
            if len(embedding) != self.embedding_dim:
                log_entry(self.state, "memory", "embedding_dimension_mismatch",
                         extra={"expected": self.embedding_dim, "actual": len(embedding)})
                # Pad or truncate to match expected dimension
                if len(embedding) < self.embedding_dim:
                    embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
                else:
                    embedding = embedding[:self.embedding_dim]
            
            cost = calculate_cost(model, len(text.split()), 0)
            self.state.total_cost += cost
            
            return embedding
            
        except Exception as e:
            log_entry(self.state, "memory", "embedding_error", error=str(e))
            return [0.0] * self.embedding_dim
    
    def _get_query_style_embedding(self, entities: List[str]) -> Optional[List[float]]:
        """Generate a query style embedding by averaging existing reference embeddings for entities."""
        if not entities:
            return None
            
        try:
            # Get style embeddings from existing references that match the entities
            df = self.visual_ctx_table.to_pandas()
            if df.empty or 'style_embedding' not in df.columns:
                return None
                
            # Filter to reference images with matching entities
            entity_refs = df[
                (df['scene_id'] == -1) &  # Reference images only
                (df['entity'].isin(entities))
            ]
            
            if entity_refs.empty:
                # Fallback: use any reference images
                entity_refs = df[df['scene_id'] == -1]
                
            if entity_refs.empty:
                return None
                
            # Average the style embeddings (excluding zero vectors)
            style_embeddings = []
            for _, row in entity_refs.iterrows():
                style_embed = row.get('style_embedding', [])
                if style_embed and sum(abs(x) for x in style_embed) > 0:  # Skip zero vectors
                    style_embeddings.append(style_embed)
                    
            if not style_embeddings:
                return None
                
            # Compute average embedding
            avg_embedding = [
                sum(emb[i] for emb in style_embeddings) / len(style_embeddings)
                for i in range(len(style_embeddings[0]))
            ]
            
            return avg_embedding
            
        except Exception as e:
            log_entry(self.state, "memory", "query_style_embedding_error", error=str(e))
            return None
    
    def _fuse_content_style_results(self, content_results: List[Any], style_results: List[Any], k_img: int) -> List[Any]:
        """Fuse content and style search results using weighted rank fusion."""
        try:
            style_weight = self.state.config.get("retrieval", {}).get("style_weight", 0.45)
            content_weight = 1.0 - style_weight
            
            # Create rankings for both result sets
            content_ranks = {getattr(r, 'frame_id', ''): i for i, r in enumerate(content_results)}
            style_ranks = {getattr(r, 'frame_id', ''): i for i, r in enumerate(style_results)}
            
            # Collect all unique frame_ids
            all_frame_ids = set(content_ranks.keys()) | set(style_ranks.keys())
            
            # Calculate fusion scores
            fusion_scores = []
            for frame_id in all_frame_ids:
                if not frame_id:
                    continue
                    
                # Get ranks (lower is better, so invert for scoring)
                content_rank = content_ranks.get(frame_id, len(content_results))
                style_rank = style_ranks.get(frame_id, len(style_results))
                
                # Reciprocal rank fusion with weights
                content_score = content_weight / (content_rank + 1)
                style_score = style_weight / (style_rank + 1)
                
                fusion_score = content_score + style_score
                fusion_scores.append((fusion_score, frame_id))
            
            # Sort by fusion score (descending)
            fusion_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Build final result list preserving original objects
            frame_id_to_result = {}
            for result in content_results + style_results:
                frame_id = getattr(result, 'frame_id', '')
                if frame_id and frame_id not in frame_id_to_result:
                    frame_id_to_result[frame_id] = result
            
            fused_results = []
            for _, frame_id in fusion_scores[:k_img * 2]:  # Get more candidates for final filtering
                if frame_id in frame_id_to_result:
                    fused_results.append(frame_id_to_result[frame_id])
            
            return fused_results[:k_img]
            
        except Exception as e:
            log_entry(self.state, "memory", "fusion_error", error=str(e))
            # Fallback to content results only
            return content_results[:k_img]
    
    def update_episodic_memory(self, event: Dict[str, Any]) -> None:
        """Update episodic memory with workflow events."""
        self.state.episodic_memory.append({
            "timestamp": datetime.now().isoformat(),
            "event": event
        })
        
        # Keep recent events proportional to context window
        ctx_window = self.state.config.get("ctx_window", 4)
        max_events = max(20, ctx_window * 12)  # Proportional to context window
        if len(self.state.episodic_memory) > max_events:
            self.state.episodic_memory = self.state.episodic_memory[-max_events:]
    
    def update_visual_memory(self, frame_info: Dict[str, Any]) -> None:
        """Update visual memory with accepted frames."""
        self.state.visual_memory.append({
            "timestamp": datetime.now().isoformat(),
            "frame": frame_info
        })
        
        # Keep recent frames proportional to visual context window
        ctx_images = self.state.config.get("ctx_images", 3)
        max_frames = max(10, ctx_images * 6)  # Proportional to visual context window
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
    
    def export_memory_to_output(self, output_memory_dir: str) -> None:
        """Export memory data to output directory for inspection."""
        output_path = Path(output_memory_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export canonical entities
            if hasattr(self, 'canonical_text_table'):
                canonical_df = self.canonical_text_table.to_pandas()
                if not canonical_df.empty:
                    # Remove embeddings for readability, keep metadata
                    export_canonical = canonical_df[['chunk_id', 'chunk_text']].copy()
                    export_canonical.to_json(
                        output_path / "canonical_entities.json", 
                        orient='records', indent=2
                    )
            
            # Export episodic memory (scene summaries and generated frames)
            if hasattr(self, 'episodic_text_table'):
                episodic_df = self.episodic_text_table.to_pandas()
                if not episodic_df.empty:
                    # Remove embeddings, keep important metadata
                    export_episodic = episodic_df[[
                        'scene_id', 'shot_id', 'summary', 'entities', 
                        'timestamp', 'quality_score'
                    ]].copy()
                    export_episodic.to_json(
                        output_path / "episodic_memory.json",
                        orient='records', indent=2
                    )
            
            # Export visual context (reference images and generated frames)
            if hasattr(self, 'visual_ctx_table'):
                visual_df = self.visual_ctx_table.to_pandas()
                if not visual_df.empty:
                    # Remove embeddings, keep visual metadata
                    export_visual = visual_df[[
                        'frame_id', 'scene_id', 'shot_id', 'category', 'entity',
                        'tags', 'source', 'confidence', 'prompt', 'original_path'
                    ]].copy()
                    export_visual.to_json(
                        output_path / "visual_context.json",
                        orient='records', indent=2
                    )
            
            # Export failures if any
            if hasattr(self, 'failures_table'):
                failures_df = self.failures_table.to_pandas()
                if not failures_df.empty:
                    failures_df.to_json(
                        output_path / "failures.json",
                        orient='records', indent=2
                    )
            
            # Export memory statistics
            stats = self.export_memory_stats()
            with open(output_path / "memory_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Create a human-readable summary
            summary = self._create_memory_summary(stats)
            with open(output_path / "memory_summary.md", 'w') as f:
                f.write(summary)
            
            log_entry(self.state, "export_memory", "success",
                     extra={"output_path": str(output_path), "files_created": 5})
                     
        except Exception as e:
            log_entry(self.state, "export_memory", "error", error=str(e))
    
    def _create_memory_summary(self, stats: Dict[str, Any]) -> str:
        """Create a human-readable memory summary."""
        return f"""# Memory System Summary

## Database Statistics
- **Canonical Entities**: {stats['canonical_text_count']} entries
- **Episodic Memory**: {stats['episodic_text_count']} scenes/frames
- **Visual Context**: {stats['visual_ctx_count']} references/frames  
- **Failures Recorded**: {stats['failures_count']} failures
- **Database Location**: `{stats['db_path']}`

## In-Memory State
- **Episodic Events**: {stats['episodic_events']} workflow events
- **Visual Memory**: {stats['visual_memory_size']} accepted frames

## Files Exported
- `canonical_entities.json` - Character/entity definitions
- `episodic_memory.json` - Scene summaries and frame metadata
- `visual_context.json` - Reference images and generated frame context
- `failures.json` - Any generation failures (if applicable)
- `memory_stats.json` - Raw statistics
- `memory_summary.md` - This summary

## Enhanced Context Features
The memory system tracks:
- ✅ Character appearances across scenes
- ✅ Environment/setting continuity  
- ✅ Visual style consistency
- ✅ Narrative flow progression

This data powers the enhanced context system for improved consistency across long storyboards.
"""
    
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
        
        try:
            # Increase limit to get more candidates for filtering
            search_res = self.visual_ctx_table.search(embed, "clip_embedding").limit(limit * 5)

            if not search_res:
                return []

            df = search_res.to_pandas()
            
            # Filter to reference images only (scene_id = -1)
            df = df[df['scene_id'] == -1]
            
            if df.empty:
                return []

            # Apply entity filter with flexible matching
            if entity_filter:
                # Try exact match first
                exact_match = df[df["entity"] == entity_filter]
                if not exact_match.empty:
                    df = exact_match
                else:
                    # Try case-insensitive partial match
                    entity_lower = entity_filter.lower()
                    partial_match = df[df["entity"].str.lower().str.contains(entity_lower, na=False)]
                    if not partial_match.empty:
                        df = partial_match
                        print(f"[Memory] Using partial match for entity '{entity_filter}'")
                    else:
                        # No entity filter match found, continue with all results
                        print(f"[Memory] No entity match for '{entity_filter}', using all results")

            # Enhanced ranking by similarity * confidence with more lenient threshold
            def _score(row):
                # Use very lenient distance threshold - accept more distant matches
                sim = max(0.2, 1 - getattr(row, "_distance", 0.3)) if hasattr(row, "_distance") else 0.5
                conf = getattr(row, "confidence", 0.7) if hasattr(row, "confidence") else 0.7
                return sim * (0.4 + 0.6 * conf)  # Weight confidence but not too heavily

            # Convert to list for sorting
            results = []
            for row in df.itertuples():
                result_dict = row._asdict()
                result_dict["_score"] = _score(row)
                results.append(result_dict)
            
            # Sort by score and return top results
            results.sort(key=lambda x: x["_score"], reverse=True)
            
            # Remove the temporary score field
            for result in results:
                result.pop("_score", None)
            
            final_results = results[:limit]
            print(f"[Memory] Search for '{query}' (entity: {entity_filter}) returned {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            print(f"[Memory] Error in search_references: {e}")
            return []

    # ------------------------------------------------------------------
    # Cheap metadata-only LLM re-ranking for retrieval candidates
    # ------------------------------------------------------------------

    def _text_rerank(
        self,
        shot_desc: str,
        candidates: List[Any],
        top_k: Optional[int] = None,
    ) -> List[Any]:
        """Re-order ANN hits using a cheap GPT-4o tier.

        The model only sees *text* (no images) – we therefore construct a
        natural-language table of the candidate metadata and ask the model to
        assign a 0-100 relevance score. Returns the candidates sorted by that
        score, capped at *top_k*.
        """

        model = self.state.config["models"].get("reranker_text", "gpt-4o-mini")
        client = self.client  # already initialised in __init__

        if top_k is None:
            top_k = (
                self.state.config.get("retrieval", {}).get("text_rerank_k", 10)
            )

        # ------------------------------------------------------------------
        # Build a simple text table for the LLM
        # ------------------------------------------------------------------
        rows = []
        idx2cand = {}
        for idx, cand in enumerate(candidates, 1):
            # namedtuple rows from pandas keep attributes as properties – use
            # getattr to stay generic.
            entity = getattr(cand, "entity", "?")
            tags_raw = getattr(cand, "tags", [])
            # Handle pandas arrays/Series and ensure we have a list
            if tags_raw is None:
                tags = []
            elif hasattr(tags_raw, 'tolist'):  # pandas array/series
                tags = tags_raw.tolist() if hasattr(tags_raw, 'tolist') else list(tags_raw)
            elif isinstance(tags_raw, (list, tuple)):
                tags = list(tags_raw)
            else:
                tags = []
            scene_id = getattr(cand, "scene_id", "?")
            shot_id = getattr(cand, "shot_id", "?")
            confidence = getattr(cand, "confidence", 0.0)

            rows.append(
                f"[{idx}] {entity} | tags: {', '.join(tags[:5])} | "
                f"scene {scene_id} shot {shot_id} | conf {confidence:.2f}"
            )

            idx2cand[idx] = cand

        table_txt = "\n".join(rows)

        prompt_header = (
            f"{shot_desc}\n\nFor each reference line below, give a relevance score 0-100.\n"
            "Return **only** a JSON array of objects {\"id\": <int>, \"score\": <float>}.")

        messages = [
            {"role": "system", "content": "You are a helpful storyboard assistant."},
            {"role": "user", "content": f"{prompt_header}\n{table_txt}"},
        ]

        scores: dict[int, float] = {}
        try:
            resp = call_openai_with_retry(
                client,
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=120,
            )

            parsed = parse_json_response(resp.choices[0].message.content)

            # The helper returns either a list directly or under a "data" key –
            # handle both.
            if isinstance(parsed, dict) and "data" in parsed:
                arr = parsed["data"]
            else:
                arr = parsed

            if isinstance(arr, list) and len(arr) > 0:
                scores = {int(d.get("id", 0)): float(d.get("score", 0)) for d in arr if isinstance(d, dict)}

        except Exception as e:
            # Log and fall back to original candidate order (ANN similarity)
            log_entry(self.state, "text_rerank", "error", error=str(e))
            return candidates[:top_k]

        # Attach scores and sort (fallback to 0 for missing)
        scored = [(scores.get(idx, 0.0), cand) for idx, cand in idx2cand.items()]
        scored.sort(key=lambda x: -x[0])

        # Budget accounting – assume ~300 input tokens, negligible output
        approx_tokens = 300
        cost = approx_tokens / 1000 * COST_PER_1K_TOKENS.get(model, {}).get("input", 0)

        self.state.total_cost += cost

        log_entry(
            self.state,
            "text_rerank",
            "success",
            model=model,
            cost_usd=cost,
            extra={"cands": len(candidates)},
        )

        return [cand for _, cand in scored[:top_k]]

    def index_canonical_entities(self, entities_dict: Dict[str, Any]) -> None:
        """Index canonical entity descriptions into canonical_text table (idempotent)."""
        if not entities_dict:
            return

        # Fetch existing ids to avoid duplicates
        existing_df = self.canonical_text_table.to_pandas()
        existing_ids = set(existing_df["chunk_id"]) if not existing_df.empty else set()

        new_records = []
        for name, desc in entities_dict.items():
            # Description may be dict or string
            if isinstance(desc, dict):
                text_desc = json.dumps(desc, ensure_ascii=False)
            else:
                text_desc = str(desc)

            if name in existing_ids:
                continue  # Skip already indexed

            embed = self._generate_embedding(text_desc)

            new_records.append({
                "chunk_id": name,
                "chunk_text": text_desc,
                "text_embedding": embed,
            })

        if new_records:
            self.canonical_text_table.add(new_records)
            log_entry(self.state, "memory_index_canonical", "success", extra={"count": len(new_records)})

    def lookup_canonical(self, entity_name: str) -> Optional[str]:
        """Return canonical description text for an entity if exists."""
        df = self.canonical_text_table.to_pandas()
        if df.empty:
            return None
        row = df[df["chunk_id"] == entity_name]
        if row.empty:
            return None
        return str(row.iloc[0]["chunk_text"])

    # ========================================================================
    # ENHANCED GLOBAL CONTEXT METHODS FOR LONG STORYBOARDS
    # ========================================================================
    
    def get_enhanced_visual_context(self, scene_id: int, shot_id: int, 
                                  window_size: int = 4) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
        """Get enhanced visual context with global consistency tracking."""
        print(f"[Memory] get_enhanced_visual_context called for Scene {scene_id} Shot {shot_id}")
        
        # Get standard context with improved retrieval
        nearby_frames, relevant_refs = self.get_visual_context(scene_id, shot_id, window_size)
        
        # Additional enhanced retrieval if we still don't have enough references
        if len(relevant_refs) < 3:
            print(f"[Memory] Only {len(relevant_refs)} refs from basic search, trying enhanced methods")
            
            # Try to get current scene for context
            current_scene = self.state.scenes[scene_id - 1] if scene_id <= len(self.state.scenes) else None
            
            if current_scene:
                # Generate embedding for scene description + entities
                scene_text = f"{current_scene.description or ''} {' '.join(current_scene.entities)}"
                if scene_text.strip():
                    try:
                        scene_embed = self._generate_embedding(scene_text)
                        entity_names = current_scene.entities if current_scene.entities and len(current_scene.entities) > 0 else ["Helena", "Joy", "Tanaka"]
                        
                        # Use hybrid retrieve for additional references
                        _, extra_refs = self.hybrid_retrieve(scene_embed, entity_names, shot_id, k_txt=0, k_img=5)
                        
                        # Convert to dict format and merge
                        if extra_refs is not None and len(extra_refs) > 0:
                            extra_ref_dicts = [r._asdict() for r in extra_refs]
                            # Avoid duplicates
                            existing_ids = {r.get("frame_id") for r in relevant_refs}
                            for ref in extra_ref_dicts:
                                if ref.get("frame_id") not in existing_ids:
                                    relevant_refs.append(ref)
                            print(f"[Memory] Added {len(extra_ref_dicts)} enhanced references")
                    except Exception as e:
                        print(f"[Memory] Error in enhanced retrieval: {e}")
        
        # NEW: Global context if enabled
        global_context = {}
        if self.state.config.get("global_context_enabled", False):
            global_context = self._get_global_context(scene_id, shot_id)
        
        print(f"[Memory] Enhanced context result: {len(nearby_frames)} frames, {len(relevant_refs)} refs, global_enabled: {bool(global_context)}")
        
        return nearby_frames, relevant_refs, global_context
    
    def _get_global_context(self, scene_id: int, shot_id: int) -> Dict[str, Any]:
        """Get global context for long-range consistency."""
        config = self.state.config
        global_window = config.get("global_ctx_window", 20)
        consistency_tracking = config.get("consistency_tracking", {})
        
        global_context = {
            "character_consistency": [],
            "environment_consistency": [],
            "style_consistency": [],
            "narrative_flow": []
        }
        
        # Character consistency tracking
        if consistency_tracking.get("characters", True):
            char_window = config.get("character_consistency_window", 50)
            global_context["character_consistency"] = self._get_character_appearances(
                scene_id, shot_id, char_window
            )
        
        # Environment consistency tracking  
        if consistency_tracking.get("environments", True):
            env_window = config.get("environment_consistency_window", 30)
            global_context["environment_consistency"] = self._get_environment_continuity(
                scene_id, shot_id, env_window
            )
            
        # Style consistency tracking
        if consistency_tracking.get("style_elements", True):
            global_context["style_consistency"] = self._get_style_consistency(
                scene_id, shot_id, global_window
            )
            
        # Narrative flow tracking
        if consistency_tracking.get("narrative_flow", True):
            global_context["narrative_flow"] = self._get_narrative_flow(
                scene_id, shot_id, global_window
            )
        
        return global_context
    
    def _get_character_appearances(self, scene_id: int, shot_id: int, window: int) -> List[Dict[str, Any]]:
        """Track character appearances across many scenes for consistency."""
        try:
            # Query all frames that mention characters
            df = self.episodic_text_table.to_pandas()
            if df.empty:
                return []
                
            # Look for character mentions in recent history
            recent_frames = df[
                (df['scene_id'] < scene_id) | 
                ((df['scene_id'] == scene_id) & (df['shot_id'] < shot_id))
            ].tail(window)
            
            character_history = []
            for _, frame in recent_frames.iterrows():
                if frame.get('entities'):
                    character_history.append({
                        'scene_id': frame['scene_id'],
                        'shot_id': frame['shot_id'],
                        'entities': frame['entities'],
                        'summary': frame['summary'][:100],
                        'quality_score': frame.get('quality_score', 0.0)
                    })
            
            return character_history
        except Exception:
            return []
    
    def _get_environment_continuity(self, scene_id: int, shot_id: int, window: int) -> List[Dict[str, Any]]:
        """Track environment/setting continuity across scenes."""
        try:
            df = self.visual_ctx_table.to_pandas()
            if df.empty:
                return []
                
            # Look for environment-related frames
            env_frames = df[
                (df['scene_id'] < scene_id) & 
                (df['category'].isin(['environment', 'background', 'setting']))
            ].tail(window)
            
            environments = []
            for _, frame in env_frames.iterrows():
                environments.append({
                    'scene_id': frame['scene_id'],
                    'shot_id': frame.get('shot_id', -1),
                    'category': frame['category'],
                    'entity': frame['entity'],
                    'tags': frame.get('tags', []),
                    'confidence': frame.get('confidence', 0.0)
                })
            
            return environments
        except Exception:
            return []
    
    def _get_style_consistency(self, scene_id: int, shot_id: int, window: int) -> List[Dict[str, Any]]:
        """Track visual style consistency across the storyboard."""
        try:
            df = self.episodic_text_table.to_pandas()
            if df.empty:
                return []
                
            # Get recent frames for style analysis
            recent_frames = df[
                (df['scene_id'] <= scene_id)
            ].tail(window)
            
            style_elements = []
            for _, frame in recent_frames.iterrows():
                style_elements.append({
                    'scene_id': frame['scene_id'],
                    'shot_id': frame['shot_id'],
                    'summary_excerpt': frame['summary'][:150],
                    'quality_score': frame.get('quality_score', 0.0)
                })
            
            return style_elements
        except Exception:
            return []
    
    def _get_narrative_flow(self, scene_id: int, shot_id: int, window: int) -> List[Dict[str, Any]]:
        """Track narrative progression and flow."""
        try:
            df = self.episodic_text_table.to_pandas()
            if df.empty:
                return []
                
            # Get sequential narrative context
            narrative_frames = df[
                (df['scene_id'] <= scene_id)
            ].sort_values(['scene_id', 'shot_id']).tail(window)
            
            narrative_flow = []
            for _, frame in narrative_frames.iterrows():
                narrative_flow.append({
                    'scene_id': frame['scene_id'],
                    'shot_id': frame['shot_id'],
                    'narrative_summary': frame['summary'][:100],
                    'entities_involved': frame.get('entities', []),
                    'quality_score': frame.get('quality_score', 0.0)
                })
                
            return narrative_flow
        except Exception:
            return []
    
    def build_global_consistency_prompt(self, global_context: Dict[str, Any]) -> str:
        """Build a prompt section that includes global consistency information."""
        if not global_context:
            return ""
            
        sections = []
        
        # Character consistency section
        if global_context.get("character_consistency"):
            char_info = global_context["character_consistency"]
            char_summary = f"Recent character appearances ({len(char_info)} frames):\n"
            for i, char in enumerate(char_info[-5:]):  # Last 5 appearances
                entities_str = ", ".join(char.get('entities', []))
                char_summary += f"- Scene {char['scene_id']}.{char['shot_id']}: {entities_str}\n"
            sections.append(f"<CHARACTER_CONSISTENCY>\n{char_summary}</CHARACTER_CONSISTENCY>")
        
        # Environment consistency section  
        if global_context.get("environment_consistency"):
            env_info = global_context["environment_consistency"]
            if env_info:
                env_summary = f"Environment continuity ({len(env_info)} references):\n"
                for env in env_info[-3:]:  # Last 3 environments
                    tags_str = ", ".join(env.get('tags', [])[:3])
                    env_summary += f"- {env['entity']}: {tags_str}\n"
                sections.append(f"<ENVIRONMENT_CONSISTENCY>\n{env_summary}</ENVIRONMENT_CONSISTENCY>")
        
        # Style consistency section
        if global_context.get("style_consistency"):
            style_info = global_context["style_consistency"]
            if style_info:
                style_summary = f"Visual style progression ({len(style_info)} frames):\n"
                avg_quality = sum(s.get('quality_score', 0) for s in style_info) / len(style_info)
                style_summary += f"- Average quality: {avg_quality:.2f}\n"
                style_summary += f"- Recent style notes: {style_info[-1]['summary_excerpt'] if style_info else 'None'}\n"
                sections.append(f"<STYLE_CONSISTENCY>\n{style_summary}</STYLE_CONSISTENCY>")
        
        return "\n\n".join(sections) if sections else "" 