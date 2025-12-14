"""
Maze Mode Skin Generation Prompts
"""

# Style consistency prompt for image-to-image generation
STYLE_CONSISTENT_PROMPT = """Above is the style reference image. Generate a new asset matching this exact visual style.

{base_prompt}

CRITICAL: Match the art style, color palette, and rendering technique of the reference image.
The new asset MUST look like it comes from the SAME GAME as the reference.
"""