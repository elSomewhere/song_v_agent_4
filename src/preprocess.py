"""Preprocessing module for script parsing and reference image tagging."""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

from src.models import SceneData, RefMeta, WorkflowState
from src.utils import (
    get_openai_client, log_entry, call_openai_with_retry,
    calculate_cost, load_image_as_base64, create_thumbnail,
    get_image_hash, parse_json_response, count_tokens_approx
)


class ScriptPreprocessor:
    """Parses script.md into structured scenes."""
    
    def __init__(self, state: WorkflowState):
        self.state = state
        self.client = get_openai_client()
    
    def parse_script(self, script_content: str) -> List[SceneData]:
        """Parse script content into scenes."""
        # First try regex parsing
        scenes = self._regex_parse(script_content)
        
        # If regex fails or produces too few scenes, use GPT
        if not scenes or len(scenes) < 2:
            scenes = self._gpt_parse(script_content)
        else:
            # Regex succeeded but didn't extract location/time, so extract them separately
            scenes = self._extract_location_time_from_scenes(scenes)
        
        # Extract entities from scenes if not already populated
        scenes = self._extract_entities_from_scenes(scenes)
        
        return scenes
    
    def _regex_parse(self, script_content: str) -> List[SceneData]:
        """Try to parse script using regex patterns."""
        scenes = []
        
        # Common scene heading patterns (more specific to avoid false positives)
        patterns = [
            r'^#+\s*Scene\s+(\d+)[:\s-]*(.*)$',     # # Scene 1: Description
            r'^Scene\s+(\d+)[:\s-]*(.*)$',          # Scene 1: Description
            r'^\[Scene\s+(\d+)\][:\s-]*(.*)$',      # [Scene 1]: Description
            r'^(\d+)\.\s*Scene[:\s-]*(.*)$',        # 1. Scene: Description
            r'^(\d+)\.\s*([A-Z][^.]*(?:\.|$))',     # 1. DESCRIPTION (only if uppercase start)
        ]
        
        lines = script_content.split('\n')
        current_scene = None
        scene_text = []
        
        for line in lines:
            # Check if line matches any scene pattern
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE | re.MULTILINE)
                if match:
                    # Save previous scene if exists
                    if current_scene is not None:
                        current_scene.raw_text = '\n'.join(scene_text).strip()
                        scenes.append(current_scene)
                    
                    # Start new scene
                    scene_id = int(match.group(1))
                    description = match.group(2).strip() if len(match.groups()) > 1 else ""
                    
                    current_scene = SceneData(
                        scene_id=scene_id,
                        raw_text="",
                        description=description
                    )
                    scene_text = []
                    break
            else:
                # Not a scene header, add to current scene text
                if current_scene is not None:
                    scene_text.append(line)
        
        # Save last scene
        if current_scene is not None:
            current_scene.raw_text = '\n'.join(scene_text).strip()
            scenes.append(current_scene)
        
        return scenes
    
    def _gpt_parse(self, script_content: str) -> List[SceneData]:
        """Use GPT to parse script into scenes."""
        model = self.state.config["models"]["script_parser"]
        
        prompt = f"""Parse the following script into individual scenes. 
        Extract for each scene:
        - scene_id (integer)
        - description (brief description)
        - location (if mentioned)
        - time (if mentioned)
        - entities (character/prop names mentioned)
        - raw_text (the full text of that scene)
        
        Return as JSON array.
        
        Script:
        {script_content[:self.state.config['preprocess']['max_tokens_script']]}
        """
        
        try:
            response = call_openai_with_retry(
                self.client,
                model=model,
                messages=[
                    {"role": "system", "content": "You are a script parser. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens
            cost = calculate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
            
            self.state.total_tokens += tokens
            self.state.total_cost += cost
            
            log_entry(self.state, "preprocess_script", "success",
                     model=model, tokens=tokens, cost_usd=cost)
            
            # Parse response
            data = parse_json_response(content)
            scenes_data = data if isinstance(data, list) else data.get("scenes", [])
            
            # Convert to SceneData objects
            scenes = []
            for scene_dict in scenes_data:
                scene = SceneData(
                    scene_id=scene_dict.get("scene_id", len(scenes) + 1),
                    raw_text=scene_dict.get("raw_text", ""),
                    description=scene_dict.get("description"),
                    location=scene_dict.get("location"),
                    time=scene_dict.get("time"),
                    entities=scene_dict.get("entities", [])
                )
                scenes.append(scene)
            
            return scenes
            
        except Exception as e:
            log_entry(self.state, "preprocess_script", "error", 
                     model=model, error=str(e))
            # Return empty list on error
            return []
    
    def _extract_location_time_from_scenes(self, scenes: List[SceneData]) -> List[SceneData]:
        """Use GPT to extract location and time from scene text if not already present."""
        if not scenes:
            return scenes
        
        model = self.state.config["models"].get("script_parser", "gpt-4o")
        
        for scene in scenes:
            if scene.location and scene.time:
                continue  # Skip if already populated
                
            prompt = f"""Extract location and time from the following scene text.

Scene {scene.scene_id}:
{scene.raw_text[:500]}

Return a JSON object with keys "location" and "time". Use null if not specified.
Example: {{"location": "Inside the battleship", "time": "night"}}
"""
            
            try:
                response = call_openai_with_retry(
                    self.client,
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a location and time extractor. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=200
                )
                
                content = response.choices[0].message.content.strip()
                tokens = response.usage.total_tokens
                cost = calculate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
                
                self.state.total_tokens += tokens
                self.state.total_cost += cost
                
                log_entry(self.state, "extract_location_time", "success",
                         model=model, tokens=tokens, cost_usd=cost,
                         extra={"scene_id": scene.scene_id})
                
                data = parse_json_response(content)
                scene.location = data.get("location")
                scene.time = data.get("time")
                
            except Exception as e:
                log_entry(self.state, "extract_location_time", "error",
                         model=model, error=str(e),
                         extra={"scene_id": scene.scene_id})
        
        return scenes
    
    def _extract_entities_from_scenes(self, scenes: List[SceneData]) -> List[SceneData]:
        """Extract entities from scene text and populate entities field."""
        if not scenes:
            return scenes
        
        # Get known entities from state
        known_entities = set()
        if self.state.entities_dict:
            known_entities.update(self.state.entities_dict.keys())
        
        # Common character names and entities to look for (fallback if entities.md not loaded)
        common_entities = {
            'Helena', 'Joy', 'Tanaka', 'Mr. Tanaka', 'Urmutter', 
            'Silicate', 'battleship', 'infantry', 'mech', 'mechs'
        }
        all_entities = known_entities.union(common_entities)
        
        # Process each scene
        for scene in scenes:
            # Skip if entities already populated (from GPT parsing)
            if scene.entities:
                continue
                
            # Extract entities mentioned in the scene text
            scene_entities = []
            scene_text_lower = scene.raw_text.lower()
            
            for entity in all_entities:
                # Check for entity mentions (case-insensitive)
                entity_lower = entity.lower()
                if (entity_lower in scene_text_lower or 
                    entity.lower() in scene_text_lower or
                    any(variant in scene_text_lower for variant in [
                        f" {entity_lower} ", f" {entity_lower}'s ", f" {entity_lower}.",
                        f" {entity_lower},", f" {entity_lower}!", f" {entity_lower}?",
                        f"({entity_lower}", f"{entity_lower})", f"\"{entity_lower}\""
                    ])):
                    scene_entities.append(entity)
            
            # Also use GPT for more accurate entity extraction if we have a budget
            if self.state.config.get('preprocess', {}).get('script') == 'auto':
                gpt_entities = self._gpt_extract_entities(scene.raw_text, scene.scene_id)
                # Merge with rule-based entities
                all_found = set(scene_entities + gpt_entities)
                scene_entities = list(all_found)
            
            scene.entities = scene_entities
            
        log_entry(self.state, "extract_entities", "success", 
                 extra={"scenes_processed": len(scenes), 
                       "entities_found": sum(len(s.entities) for s in scenes)})
        
        return scenes
    
    def _gpt_extract_entities(self, scene_text: str, scene_id: int) -> List[str]:
        """Use GPT to extract entities from a single scene."""
        if not scene_text.strip():
            return []
            
        # Get known entities for context
        known_entities_list = list(self.state.entities_dict.keys()) if self.state.entities_dict else []
        known_entities_str = ", ".join(known_entities_list) if known_entities_list else "Helena, Joy, Tanaka, Urmutter, Silicate"
        
        model = self.state.config["models"].get("script_parser", "gpt-4o")
        
        prompt = f"""Extract character names, important objects, and entities mentioned in this scene.

Known entities to look for: {known_entities_str}

Scene {scene_id} text:
{scene_text[:500]}

Return ONLY a JSON array of entity names mentioned in this scene.
Example: ["Helena", "Joy", "Silicate infantry"]
"""
        
        try:
            response = call_openai_with_retry(
                self.client,
                model=model,
                messages=[
                    {"role": "system", "content": "You extract entities from text. Return only JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens
            cost = calculate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
            
            self.state.total_tokens += tokens
            self.state.total_cost += cost
            
            # Parse the JSON response
            entities = parse_json_response(content)
            if isinstance(entities, list):
                return [str(e) for e in entities if e]
            else:
                return []
                
        except Exception as e:
            log_entry(self.state, "gpt_extract_entities", "error",
                     extra={"scene_id": scene_id, "error": str(e)})
            return []


class ReferencePreprocessor:
    """Tags reference images using GPT-4o vision."""
    
    def __init__(self, state: WorkflowState):
        self.state = state
        self.client = get_openai_client()

        # Configurable flags for using directory / file names as hints
        pp_cfg = self.state.config.get("preprocess", {})
        self.use_dir_names: bool = pp_cfg.get("refs_use_dir_names", True)
        self.use_file_names: bool = pp_cfg.get("refs_use_file_names", False)

        self.cache_dir = Path(".cache") / "thumbs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def process_references(self, refs_dir: str) -> List[RefMeta]:
        """Process all reference images in directory (recursively)."""
        refs_path = Path(refs_dir)
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp'}

        ref_metas: List[RefMeta] = []

        # Walk recursively so that sub-folder images are also picked up
        for image_file in refs_path.rglob('*'):
            if not image_file.is_file():
                continue

            if image_file.suffix.lower() in valid_extensions:
                # Gather optional hints from directory / filename
                dir_hint: Optional[str] = None
                if self.use_dir_names and image_file.parent != refs_path:
                    dir_hint = image_file.parent.name

                file_hint: Optional[str] = image_file.stem if self.use_file_names else None

                # Debug output to console
                try:
                    rel_path = image_file.relative_to(refs_path)
                except ValueError:
                    rel_path = image_file
                print(f"[PreprocessRefs] Reading reference image: {rel_path}")

                try:
                    ref_meta = self._process_single_image(
                        str(image_file), dir_hint=dir_hint, file_hint=file_hint
                    )
                    ref_metas.append(ref_meta)
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    log_entry(
                        self.state,
                        "preprocess_refs",
                        "error",
                        extra={"file": str(image_file), "error": str(e)},
                    )

        return ref_metas
    
    def _process_single_image(
        self,
        image_path: str,
        dir_hint: Optional[str] = None,
        file_hint: Optional[str] = None,
    ) -> RefMeta:
        """Process a single reference image with optional textual hints."""
        # Create thumbnail
        image_hash = get_image_hash(image_path)
        thumb_path = self.cache_dir / f"{image_hash}_thumb.jpg"
        
        if not thumb_path.exists():
            create_thumbnail(image_path, str(thumb_path))
        
        # Load image for GPT vision
        image_b64 = load_image_as_base64(image_path)
        
        # Tag with GPT-4o vision (pass hints)
        tags_data = self._tag_image_with_gpt(
            image_b64, image_path, dir_hint=dir_hint, file_hint=file_hint
        )
        
        # Generate text embedding
        embedding = self._generate_embedding(tags_data['tags'])
        
        # Create RefMeta
        ref_meta = RefMeta(
            category=tags_data['category'],
            entity=tags_data['entity'],
            tags=tags_data['tags'],
            confidence=tags_data['confidence'],
            clip_embedding=embedding,
            thumb_path=str(thumb_path),
            source="user_upload",
            original_path=image_path
        )
        
        return ref_meta
    
    def _tag_image_with_gpt(
        self,
        image_b64: str,
        image_path: str,
        dir_hint: Optional[str] = None,
        file_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Use GPT-4o vision to tag an image, incorporating optional hints."""
        model = self.state.config["models"]["ref_tagger"]
        
        # Build context from entities
        entities_context = json.dumps(self.state.entities_dict) if self.state.entities_dict else ""
        
        # Build optional hint text
        hint_lines: List[str] = []
        if dir_hint:
            hint_lines.append(f"Folder hint: {dir_hint}")
        if file_hint:
            hint_lines.append(f"Filename hint: {file_hint}")

        hint_block = "\n".join(hint_lines)

        # Pre-compute fallback entity so it's always defined (also used in error path)
        fallback_entity: str = dir_hint or (file_hint if file_hint else Path(image_path).stem)

        prompt = f"""Analyze this reference image for a storyboard generation system.

Known entities: {entities_context}
{hint_block}

Provide:
1. category: \"character\", \"environment\", \"props\", or \"other\"
2. entity: main entity name (match to known entities if possible)
3. tags: list of descriptive tags (visual features, colors, poses, etc.)
4. confidence: 0.0-1.0 confidence score

Return as JSON."""
        
        try:
            response = call_openai_with_retry(
                self.client,
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            # Note: Vision models don't return token counts reliably
            cost = 0.01  # Approximate cost per image
            
            self.state.total_cost += cost
            
            log_entry(self.state, "preprocess_refs", "success",
                     model=model, cost_usd=cost,
                     extra={"image": Path(image_path).name})
            
            # Parse response
            data = parse_json_response(content)
            
            # Use GPT-identified entity first, fallback_entity only if GPT didn't identify one
            return {
                "category": data.get("category", "other"),
                "entity": data.get("entity", fallback_entity),
                "tags": data.get("tags", []),
                "confidence": float(data.get("confidence", 0.5))
            }
            
        except Exception as e:
            log_entry(self.state, "preprocess_refs", "error",
                     model=model, error=str(e))
            # Return default values
            return {
                "category": "other",
                "entity": fallback_entity,
                "tags": ["untagged"],
                "confidence": 0.0,
            }
    
    def _generate_embedding(self, tags: List[str]) -> List[float]:
        """Generate text embedding for tags."""
        model = self.state.config["models"]["embedding_text"]
        text = " ".join(tags)
        
        try:
            # text-embedding-3-large supports dimensions parameter
            embedding_dim = self.state.config.get("embedding_dimension", 1536)
            response = call_openai_with_retry(
                self.client,
                model=model,
                input=text,
                dimensions=embedding_dim  # Use configurable dimension
            )
            
            embedding = response.data[0].embedding
            tokens = response.usage.total_tokens
            cost = calculate_cost(model, tokens, 0)
            
            self.state.total_tokens += tokens
            self.state.total_cost += cost
            
            return embedding
            
        except Exception as e:
            # Return zero vector on error
            return [0.0] * self.state.config.get("embedding_dimension", 1536) 


# ------------------------------------------------------------
# NEW: EntitiesPreprocessor
# ------------------------------------------------------------

class EntitiesPreprocessor:
    """Extracts structured entity data from entities.md using GPT if no valid JSON block was found."""

    def __init__(self, state: WorkflowState):
        self.state = state
        self.client = get_openai_client()

    def parse_entities(self, entities_markdown: str) -> Dict[str, Any]:
        """Return a mapping of entity_name -> {description: str, features: str | None}."""

        # Token limit (approx chars)
        max_tokens = self.state.config.get("preprocess", {}).get("max_tokens_refs", 2000)
        char_budget = max_tokens * 4  # rough 4 chars per token
        prompt = (
            "You are a knowledgeable storyboard assistant. "
            "Extract EVERY entity (characters, props, environments) that has a heading or bullet list in the following design document. "
            "Return JSON with each entity name as a key. For each, include at minimum `description` (1-2 sentences). "
            "Include other keys like `features` if present. Do NOT wrap the JSON in markdown fences; output raw JSON only.\n\n" +
            entities_markdown[:char_budget]
        )

        model_map = self.state.config.get("models", {})
        # Use dedicated key if defined, else fall back to script_parser or planner
        model = (
            model_map.get("entities_parser")
            or model_map.get("script_parser")
            or model_map.get("planner", "gpt-4o")
        )

        try:
            response = call_openai_with_retry(
                self.client,
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert information extractor."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1500,
            )

            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
            cost = (
                calculate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
                if hasattr(response, "usage")
                else 0.0
            )

            self.state.total_tokens += tokens
            self.state.total_cost += cost

            data = parse_json_response(content)
            if not isinstance(data, dict):
                raise ValueError("Parsed entities is not a dict")

            log_entry(
                self.state,
                "preprocess_entities",
                "success",
                model=model,
                tokens=tokens,
                cost_usd=cost,
                extra={"entities": len(data)},
            )

            return data

        except Exception as e:
            # On failure just return empty dict and log error
            log_entry(
                self.state,
                "preprocess_entities",
                "error",
                model=model,
                error=str(e),
            )
            return {} 


# ------------------------------------------------------------
# NEW: EntitiesEnricher – reconcile textual entities with visual refs
# ------------------------------------------------------------

class EntitiesEnricher:
    """Merge entity descriptions with visual reference metadata via GPT."""

    def __init__(self, state: WorkflowState):
        self.state = state
        self.client = get_openai_client()

    def enrich(self) -> Dict[str, Any]:
        """Return updated entities_dict with visual info merged."""

        if not self.state.entities_dict or not self.state.ref_index:
            return self.state.entities_dict  # Nothing to do

        # Build a compact JSON of visual refs grouped by entity
        refs_by_entity: Dict[str, List[str]] = {}
        for ref in self.state.ref_index:
            ent = ref.entity or "unknown"
            refs_by_entity.setdefault(ent, []).append(", ".join(ref.tags[:6]))

        # Prepare prompt
        prompt_blocks = []
        prompt_blocks.append("### Textual Entity Descriptions (JSON)\n" + json.dumps(self.state.entities_dict, indent=2)[:6000])
        prompt_blocks.append("\n### Visual Reference Tags per Entity (from images)\n" + json.dumps(refs_by_entity, indent=2)[:4000])

        full_prompt = (
            "Merge the textual entity descriptions with the visual reference tags. "
            "For each entity, produce a CONSOLIDATED JSON object with keys: description (text), visual_traits (comma list), canonical_colors (comma list if identifiable). "
            "If there are conflicts, choose the version best supported by image tags. Return raw JSON only.\n\n" +
            "\n\n".join(prompt_blocks)
        )

        model = self.state.config["models"].get("entities_parser", self.state.config["models"].get("planner","gpt-4o"))

        try:
            resp = call_openai_with_retry(
                self.client,
                model=model,
                messages=[
                    {"role": "system", "content": "You are a meticulous story bible editor."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.2,
                max_tokens=1200,
            )

            merged = parse_json_response(resp.choices[0].message.content)
            if isinstance(merged, dict):
                log_entry(
                    self.state,
                    "enrich_entities",
                    "success",
                    model=model,
                    tokens=getattr(resp.usage, "total_tokens", None),
                )
                return merged
        except Exception as e:
            log_entry(self.state, "enrich_entities", "error", model=model, error=str(e))

        return self.state.entities_dict 