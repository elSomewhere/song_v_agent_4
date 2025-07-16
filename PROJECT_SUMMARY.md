# VC-RAG-SBG Project Summary & Current Issue Analysis

## Project Overview

**VC-RAG-SBG (Visual-Context-Aware Retrieval-Augmented-Generation Storyboard Generator)** is a sophisticated system that generates consistent, high-quality storyboard frames from scripts using AI models with visual memory and context awareness.

### Core Architecture

```
Script → Parsing → Memory System → Planning → Review → Image Generation → QA → Accept/Retry
           ↓           ↓            ↓         ↓           ↓               ↓
      Entities    Reference     Shot Plan  Enhanced   gpt-image-1    Quality
      Extraction  Images        Creation   Context    + References   Assessment
```

### Key Models Used
- **GPT-4o**: Script parsing, planning, reviewing, QA
- **gpt-image-1**: Image generation via Responses API
- **text-embedding-3-large**: Vector embeddings for memory retrieval
- **LanceDB**: Vector database for visual memory storage

### Data Structure
- **Script**: `data/script.md` (29KB, 437 lines, 60 scenes)
- **Entities**: `data/entities.md` (14KB, 244 lines) 
- **Style Guide**: `data/style.md` (1.9KB)
- **References**: `data/refs/` (33 reference images: Helena, Urmutter, Silicate Army, etc.)

## Enhanced Context System

The system includes three configuration modes:
- **Basic** ($25-35, 1-10 shots): Simple context window
- **Enhanced** ($40-60, 10-50 shots): Advanced memory retrieval + global consistency  
- **Global** ($75-100, 50+ shots): Maximum context tracking

### Enhanced Features Implemented
- **Global context tracking**: Character consistency across 50+ scenes
- **Enhanced visual context**: `get_enhanced_visual_context()` method
- **Character appearance tracking**: `_get_character_appearances()`
- **Environment continuity**: `_get_environment_continuity()`
- **Location/time extraction**: GPT-based extraction from scene narratives

## Current Issue: Reference Images Not Reaching gpt-image-1

### Problem Statement
The system loads 33 reference images during preprocessing but **none are being sent to gpt-image-1** for visual consistency during image generation, resulting in inconsistent character/environment appearance across frames.

### Root Cause Analysis

#### ✅ FIXED: API Implementation Issue
**Problem**: The renderer was using incorrect API endpoints for gpt-image-1
- **Before**: Used `client.images.edit()` with file objects (broken)
- **After**: Uses `client.responses.create()` with Responses API (working)

**Solution Implemented**:
```python
# Fixed implementation in src/nodes/renderer.py
response = call_openai_with_retry(
    client,
    model="gpt-4.1",
    input=[{
        "role": "user", 
        "content": [
            {"type": "input_text", "text": full_prompt},
            {"type": "input_image", "image_url": f"data:image/png;base64,{ref['base64']}"},
            # ... up to 4 reference images
        ]
    }],
    tools=[{"type": "image_generation"}]
)
```

#### ❌ REMAINING: Memory Retrieval Issue
**Problem**: The reviewer node cannot find relevant reference images to send to the renderer

**Debug Output**:
```
[Reviewer] No relevant references found for Scene 1 Shot 1
[Renderer] No visual_context in reviewed_plan
[Renderer] No reference images found for Scene 1 Shot 1
[Renderer] Using Responses API without reference images
```

### Technical Flow Analysis

1. **Preprocessing** ✅ Working
   - 33 reference images loaded successfully
   - Images indexed in LanceDB with embeddings
   - Entities extracted from scenes

2. **Planning** ✅ Working
   - Scene plans created with entity lists
   - Camera configurations generated

3. **Review** ❌ **FAILING HERE**
   - `get_enhanced_visual_context()` returns empty `relevant_refs`
   - `hybrid_retrieve()` not finding matches
   - No reference IDs stored in `reviewed_plan.visual_context`

4. **Rendering** ✅ Working (but gets no references)
   - Responses API correctly implemented
   - Would send references if they were provided
   - Falls back to text-only generation

### Key Files and Functions

#### Memory System (`src/memory.py`)
- `get_enhanced_visual_context()`: Should find relevant references but returns empty
- `hybrid_retrieve()`: Vector search that should match scene content to references
- `search_references()`: Searches by frame_id

#### Reviewer Node (`src/nodes/reviewer.py`)
- Gets references via `memory.get_enhanced_visual_context()`
- Stores reference IDs in `reviewed_plan.visual_context`
- Currently: `visual_context` is always empty

#### Renderer Node (`src/nodes/renderer.py`) 
- `_get_reference_images()`: Looks up references by ID from `reviewed_plan.visual_context`
- **Fixed**: Now uses correct Responses API for gpt-image-1
- Currently: Gets empty reference list

### Data Evidence

#### Successful Preprocessing
```
[PreprocessRefs] Reading reference image: Helena/My ChatGPT image (5).png
[PreprocessRefs] Reading reference image: Urmutter/ChatGPT Image Jun 13, 2025, 02_18_33 PM.png
[PreprocessRefs] Reading reference image: Silicate Army/ChatGPT Image Jun 13, 2025, 02_05_28 PM.png
# ... 33 images total
```

#### Failed Reference Lookup
```
[Reviewer] No relevant references found for Scene 1 Shot 1
[Renderer] No visual_context in reviewed_plan
```

#### Working API
```
[Renderer] Using Responses API without reference images  # API works, just no refs
```

## Next Steps to Fix Reference System

### 1. Debug Memory Indexing
Check if reference images are properly indexed in LanceDB:
- Verify `visual_ctx.lance` table has data
- Check if entity embeddings match scene content
- Validate `frame_id` generation and storage

### 2. Fix Hybrid Retrieval
Investigate why `hybrid_retrieve()` finds no matches:
- Check embedding similarity thresholds
- Verify entity name matching logic
- Test with manual queries

### 3. Validate Reference Search
Ensure `search_references(frame_id)` can find indexed images:
- Check `frame_id` format consistency
- Verify `original_path` field population

## Current System Status

### Working Components ✅
- Script parsing with location/time extraction
- Entity extraction (Helena, Joy, Tanaka, Urmutter, Silicate, etc.)
- Reference image loading (33 images)
- Planning and variation generation
- **Responses API implementation for gpt-image-1** 
- Quality assessment and policy decisions
- Memory export functionality

### Broken Components ❌
- **Memory retrieval system** (can't find relevant references)
- Visual context population in reviewer
- Reference image delivery to image generation

### Impact
Without reference images, the system generates visually inconsistent storyboards despite having all the infrastructure in place for consistency. Characters may look different between scenes, environments may not match, and the overall visual narrative coherence is compromised.

## Configuration

**Current Test Config**: `config_enhanced.yaml`
```yaml
context_mode: enhanced
global_context_enabled: true
ctx_window: 20
character_consistency_window: 50
environment_consistency_window: 30
```

**Models**:
```yaml
renderer_new: gpt-image-1  # Now using correct Responses API
renderer_edit: gpt-image-1
planner: gpt-4o
reviewer: gpt-4o  
```

## File Locations

- **Main Issue**: Memory retrieval in `src/memory.py` 
- **Fixed Component**: Responses API in `src/nodes/renderer.py`
- **Debug Point**: Reference lookup in `src/nodes/reviewer.py`
- **Test Data**: 33 reference images in `data/refs/`
- **Output**: Debug logs show API working but no references found

## Success Criteria

The issue will be resolved when:
1. `[Reviewer] Found X relevant references for Scene Y Shot Z` appears in logs
2. `[Renderer] Using Responses API with X reference images` appears
3. Generated images show visual consistency with reference materials
4. Character appearances remain consistent across scenes

The API infrastructure is now correct - we just need to fix the memory system to actually find and deliver the reference images that are already loaded and available. 