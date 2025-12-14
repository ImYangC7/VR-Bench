"""
Pipeline Nodes for Maze Mode Skin Generation

Flow:
1. AnalyzerNode: Generate analysis.json from maze configuration
2. StrategistNode: Generate strategy.json from maze configuration
3. AssetGeneratorNode: Generate image assets based on strategy
4. BackgroundRemovalNode: Remove background and crop assets
"""

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

from pydantic import Field

from autoenv.pipeline.visual.prompt import STYLE_CONSISTENT_PROMPT
from base.engine.async_llm import AsyncLLM
from base.pipeline.base_node import BaseNode, NodeContext
from base.utils.image import save_base64_image


class AutoEnvContext(NodeContext):
    """AutoEnv Pipeline context for maze mode skin generation."""

    # Maze mode input
    maze_type: str | None = None  # maze, pathfinder, sokoban, trapfield
    theme: str | None = None  # 视觉主题，如 "cyberpunk neon city"
    output_dir: Path = Field(default_factory=lambda: Path("."))

    # AnalyzerNode output
    analysis: dict[str, Any] | None = None
    analysis_file: Path | None = None

    # StrategistNode output
    strategy: dict[str, Any] | None = None
    strategy_file: Path | None = None

    # AssetGeneratorNode output
    generated_assets: dict[str, str] = Field(default_factory=dict)
    style_anchor_image: str | None = None

    # Error tracking
    success: bool = False
    error: str | None = None





class AnalyzerNode(BaseNode):
    """Generate analysis.json from maze configuration."""

    async def execute(self, ctx: AutoEnvContext) -> None:
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = ctx.output_dir / "analysis.json"

        if not ctx.maze_type or not ctx.theme:
            ctx.error = "AnalyzerNode requires maze_type and theme"
            return

        from autoenv.pipeline.visual.maze_assets_config import get_maze_config

        try:
            config = get_maze_config(ctx.maze_type)
            ctx.analysis = {
                "mode": "maze",
                "maze_type": ctx.maze_type,
                "theme": ctx.theme,
                "description": config["description"],
            }

            # 保存 analysis.json
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(ctx.analysis, f, indent=2, ensure_ascii=False)
            ctx.analysis_file = output_file

            print(f"[AnalyzerNode] Maze mode: {ctx.maze_type}, theme: {ctx.theme}")
        except ValueError as e:
            ctx.error = str(e)


class StrategistNode(BaseNode):
    """Generate strategy.json from maze configuration."""

    async def execute(self, ctx: AutoEnvContext) -> None:
        if not ctx.maze_type or not ctx.theme:
            ctx.error = "StrategistNode requires maze_type and theme"
            return

        from autoenv.pipeline.visual.maze_assets_config import (
            generate_strategy_for_maze,
        )

        try:
            ctx.strategy = generate_strategy_for_maze(ctx.maze_type, ctx.theme)

            # 保存 strategy.json
            output_file = ctx.output_dir / "strategy.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(ctx.strategy, f, indent=2, ensure_ascii=False)
            ctx.strategy_file = output_file

            print(
                f"[StrategistNode] Generated strategy for {ctx.maze_type} "
                f"with theme: {ctx.theme}"
            )
        except ValueError as e:
            ctx.error = str(e)


class AssetGeneratorNode(BaseNode):
    """Generate game assets based on strategy."""

    image_llm: AsyncLLM | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, ctx: AutoEnvContext) -> None:
        if not ctx.strategy:
            ctx.error = "AssetGeneratorNode requires strategy from StrategistNode"
            return

        if not self.image_llm:
            ctx.error = "AssetGeneratorNode requires image_llm"
            return

        assets = ctx.strategy.get("assets", [])
        if not assets:
            ctx.error = "No assets defined in strategy"
            return

        skins_dir = ctx.output_dir / "skins"
        skins_dir.mkdir(parents=True, exist_ok=True)
        print(f"[AssetGenerator] Starting generation for {len(assets)} assets → {skins_dir}")

        # Sort by priority, style_anchor first
        sorted_assets = sorted(assets, key=lambda x: -x.get("priority", 0))

        # 1. Generate style_anchor (text-to-image) - must complete first, other assets depend on it
        style_anchor_id = ctx.strategy.get("style_anchor")
        for asset in sorted_assets:
            if asset.get("id") == style_anchor_id:
                print(f"[AssetGenerator] Generating style anchor: {style_anchor_id}")
                prompt = self._get_asset_prompt(asset)
                result = await self.image_llm.generate_text_to_image(prompt)
                if result["success"]:
                    ctx.generated_assets[asset["id"]] = result["image_base64"]
                    ctx.style_anchor_image = result["image_base64"]
                    save_base64_image(result["image_base64"], skins_dir / f"{style_anchor_id}.png")
                    print(f"[AssetGenerator] ✓ Style anchor saved: {style_anchor_id}.png")
                else:
                    print(f"[AssetGenerator] ✗ Style anchor failed: {result.get('error')}")
                break

        # 2. Generate other assets in parallel (image-to-image, using style_anchor as reference)
        other_assets = [a for a in sorted_assets if a.get("id") != style_anchor_id]
        if other_assets:
            print(f"[AssetGenerator] Generating {len(other_assets)} assets in parallel...")
            tasks = [self._generate_asset(asset, ctx, skins_dir) for asset in other_assets]
            await asyncio.gather(*tasks)

        print(f"[AssetGenerator] Done. Total: {len(ctx.generated_assets)} assets")

    async def _generate_asset(
        self, asset: dict[str, Any], ctx: AutoEnvContext, skins_dir: Path
    ) -> None:
        """Generate a single asset and save immediately."""
        asset_id = asset.get("id", "unknown")
        print(f"[AssetGenerator] → Generating: {asset_id}")

        prompt = STYLE_CONSISTENT_PROMPT.format(base_prompt=self._get_asset_prompt(asset))
        if ctx.style_anchor_image:
            result = await self.image_llm.generate_image_to_image(
                prompt, [ctx.style_anchor_image]
            )
        else:
            result = await self.image_llm.generate_text_to_image(prompt)

        if result["success"]:
            ctx.generated_assets[asset_id] = result["image_base64"]
            save_base64_image(result["image_base64"], skins_dir / f"{asset_id}.png")
            print(f"[AssetGenerator] ✓ Saved: {asset_id}.png")
        else:
            print(f"[AssetGenerator] ✗ Failed: {asset_id} - {result.get('error')}")

    def _get_asset_prompt(self, asset: dict[str, Any]) -> str:
        """Get asset generation prompt."""
        prompt = asset.get("prompt_strategy", {}).get("base_prompt", "")
        if not prompt:
            prompt = asset.get("description", asset.get("name", "game asset"))
        return prompt


class BackgroundRemovalNode(BaseNode):
    """Remove image background, crop to subject, and generate description.json."""

    vision_llm: AsyncLLM | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, ctx: AutoEnvContext) -> None:
        if not ctx.generated_assets:
            ctx.error = "BackgroundRemovalNode requires generated_assets"
            return

        skins_dir = ctx.output_dir / "skins"
        if not skins_dir.exists():
            ctx.error = "Skins directory not found"
            return

        print(f"[BackgroundRemoval] Processing {len(ctx.generated_assets)} assets...")

        tasks = []
        for asset_id in ctx.generated_assets:
            image_path = skins_dir / f"{asset_id}.png"
            if image_path.exists():
                tasks.append(self._process_image(image_path, asset_id))

        await asyncio.gather(*tasks)
        print("[BackgroundRemoval] Done.")

        # Generate description.json with visual analysis
        await self._generate_description(ctx, skins_dir)

    async def _process_image(self, image_path: Path, asset_id: str) -> None:
        """Remove background and crop to subject."""
        from PIL import Image
        from rembg import remove

        def _process() -> None:
            img = Image.open(image_path)
            output = remove(img)
            bbox = output.getbbox()
            if bbox:
                output = output.crop(bbox)
            output.save(image_path)

        await asyncio.to_thread(_process)
        print(f"[BackgroundRemoval] ✓ Processed: {asset_id}.png")

    async def _generate_description(self, ctx: AutoEnvContext, skins_dir: Path) -> None:
        """Generate description.json with visual descriptions from image analysis."""
        if not ctx.strategy or not ctx.maze_type:
            print(
                "[BackgroundRemoval] ⚠️  Cannot generate description: "
                "missing strategy or maze_type"
            )
            return

        if not self.vision_llm:
            print(
                "[BackgroundRemoval] ⚠️  Cannot generate description: "
                "vision_llm not configured"
            )
            return

        # Extract skin_id from output directory name
        # e.g., "maze_20231214_130944" -> "20231214_130944"
        skin_id = (
            ctx.output_dir.name.split("_", 1)[-1]
            if "_" in ctx.output_dir.name
            else "1"
        )

        print("[BackgroundRemoval] Analyzing images for visual descriptions...")

        # Analyze each asset image in parallel
        tasks = []
        asset_ids = []
        for asset_id in ctx.generated_assets:
            image_path = skins_dir / f"{asset_id}.png"
            if image_path.exists():
                tasks.append(self._analyze_image(image_path))
                asset_ids.append(asset_id)

        descriptions = await asyncio.gather(*tasks)

        # Build visual_description dict
        visual_description = {}
        for asset_id, desc in zip(asset_ids, descriptions):
            if desc:
                visual_description[asset_id] = desc
            else:
                visual_description[asset_id] = asset_id  # Fallback

        # Map target to goal for consistency with VR-Bench naming
        if "target" in visual_description and ctx.maze_type in ["maze", "sokoban"]:
            visual_description["goal"] = visual_description.pop("target")

        # Generate description.json (format matches VR-Bench skins)
        description = {
            "game_type": ctx.maze_type,
            "skin_id": skin_id,
            "visual_description": visual_description
        }

        description_file = skins_dir / "description.json"
        with open(description_file, "w", encoding="utf-8") as f:
            json.dump(description, f, indent=2, ensure_ascii=False)

        print("[BackgroundRemoval] ✓ Generated: description.json")

    async def _analyze_image(self, image_path: Path) -> str | None:
        """Analyze a single image and return a short visual description."""
        try:
            # Read image and convert to base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Prepare multimodal prompt with system message
            system_prompt = (
                "You are an expert at describing game sprite images.\n"
                "Your task is to describe what this sprite image shows "
                "in a concise phrase.\n"
                "Focus on:\n"
                "1. The object/character type (e.g., ball, rabbit, "
                "treasure chest, stone wall, wooden crate)\n"
                "2. The primary color (e.g., red, blue, golden, gray)\n"
                "3. The shape if relevant (e.g., circle, square, star)\n\n"
                "Return ONLY a short descriptive phrase, nothing else.\n"
                "Examples: \"red ball\", \"golden treasure chest\", "
                "\"gray stone bricks\", \"white rabbit\", \"green circle\", "
                "\"wooden floor tiles\""
            )

            user_prompt = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                },
                {
                    "type": "text",
                    "text": (
                        "Describe this game sprite image in a few words. "
                        "What object is it and what color?"
                    )
                }
            ]

            # Create a temporary LLM instance with system prompt
            temp_llm = AsyncLLM(
                self.vision_llm.config,
                system_msg=system_prompt,
                max_completion_tokens=50
            )

            # Call vision LLM
            response = await temp_llm(user_prompt)

            # Clean up response (remove quotes, extra whitespace)
            description = response.strip().strip('"').strip("'").strip()

            print(f"[BackgroundRemoval]   → {image_path.name}: {description}")

            return description

        except Exception as e:
            print(f"[BackgroundRemoval] ✗ Failed to analyze {image_path.name}: {e}")
            return None



