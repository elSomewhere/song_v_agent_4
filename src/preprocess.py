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
        
        return scenes
    
    def _regex_parse(self, script_content: str) -> List[SceneData]:
        """Try to parse script using regex patterns."""
        scenes = []
        
        # Common scene heading patterns
        patterns = [
            r'^#+\s*Scene\s+(\d+)[:\s-]*(.*)$',  # # Scene 1: Description
            r'^Scene\s+(\d+)[:\s-]*(.*)$',        # Scene 1: Description
            r'^\[Scene\s+(\d+)\][:\s-]*(.*)$',    # [Scene 1]: Description
            r'^(\d+)\.\s+(.*)$',                  # 1. Description
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
            
            # Determine a best-effort entity fallback
            fallback_entity = dir_hint or (file_hint if file_hint else Path(image_path).stem)

            return {
                "category": data.get("category", "other"),
                "entity": fallback_entity,
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