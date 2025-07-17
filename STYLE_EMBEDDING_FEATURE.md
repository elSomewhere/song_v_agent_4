# Visual Style Embedding Feature

This document describes the newly implemented **visual style embedding** feature that enhances reference image retrieval by matching visual style characteristics in addition to content similarity.

## Overview

The style embedding feature uses OpenAI's `image-embed-1` model to extract low-level perceptual style vectors (color palette, texture, stroke weight, etc.) from images and uses them alongside content embeddings for improved visual consistency in storyboard generation.

## Usage

### Command Line Flag

Enable the feature using the new command line flag:

```bash
python run.py --data ./data --out ./output --enable-style-embedding
```

### Configuration

The feature can also be enabled via configuration files:

```yaml
# In your config.yaml
style_embedding_enabled: true
style_embedding_dimension: 1024
retrieval:
  style_weight: 0.45  # Weight for style vs content (0.0 = content only, 1.0 = style only)

models:
  embedding_style: "image-embed-1"
```

## How It Works

1. **Preprocessing**: When reference images are processed, style embeddings are computed using `image-embed-1` and cached locally
2. **Generated Frames**: Style embeddings are also computed for generated frames when the feature is enabled
3. **Retrieval**: During reference search, both content embeddings (from tags) and style embeddings are used
4. **Fusion**: Results are combined using weighted rank fusion based on the `style_weight` parameter

## Technical Details

### Database Schema
- Adds `style_embedding` column to the `visual_ctx` table
- Backward compatible with existing databases
- Uses 1024-dimensional vectors (matching `image-embed-1`)

### Cost and Performance
- Approximately $0.0005 per image embedding
- Results are cached in `.cache/style_emb/` to avoid recomputation
- Gracefully degrades to content-only search if disabled or on errors

### Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `style_embedding_enabled` | `false` | Master switch for the feature |
| `style_embedding_dimension` | `1024` | Dimension of style vectors |
| `retrieval.style_weight` | `0.45` | Weight for style vs content similarity |
| `models.embedding_style` | `"image-embed-1"` | Model for style embeddings |

## Benefits

- **Improved Visual Consistency**: Better matching of visual style across reference images
- **Enhanced Retrieval**: More relevant reference images for style-sensitive scenes
- **Backward Compatible**: Existing workflows continue to work unchanged
- **Cost Effective**: Caching prevents repeated API calls for the same images
- **Optional**: Can be enabled/disabled per run without affecting the core system

## Examples

### Basic Usage
```bash
# Standard run (content-only retrieval)
python run.py --data ./storyboard --out ./output

# With style embedding enabled
python run.py --data ./storyboard --out ./output --enable-style-embedding
```

### Custom Style Weight
```yaml
# config.yaml - emphasize style over content
style_embedding_enabled: true
retrieval:
  style_weight: 0.7  # 70% style, 30% content
```

## Implementation Notes

- Uses OpenAI's beta `image-embed-1` model
- Implements rank fusion algorithm for combining content and style results
- Handles missing or corrupted style embeddings gracefully
- Supports migration of existing databases
- Includes comprehensive error handling and logging

This feature provides a significant enhancement to visual consistency while maintaining full backward compatibility with existing workflows. 