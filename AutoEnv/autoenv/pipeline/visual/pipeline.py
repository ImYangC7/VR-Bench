"""
Maze Mode Skin Generation Pipeline
Simplified DAG-based visual asset generation pipeline
"""

from pathlib import Path

from autoenv.pipeline.visual.nodes import (
    AnalyzerNode,
    AssetGeneratorNode,
    AutoEnvContext,
    BackgroundRemovalNode,
    StrategistNode,
)
from base.engine.async_llm import AsyncLLM
from base.pipeline.base_pipeline import BasePipeline


class VisualPipeline(BasePipeline):
    """
    Visualization pipeline for maze mode.

    DAG structure:
        Analyzer → Strategist → AssetGenerator → BackgroundRemoval
    """

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def create_default(
        cls,
        image_model: str,
    ) -> "VisualPipeline":
        """
        Factory method: Create default visualization pipeline.

        Args:
            image_model: Image generation model name (required)

        Usage:
            pipeline = VisualPipeline.create_default(
                image_model="gemini-2.5-flash-image"
            )
            ctx = await pipeline.run(
                maze_type="maze",
                theme="cyberpunk neon city",
                output_dir=Path("workspace/envs/maze_001")
            )
        """
        image_llm = AsyncLLM(image_model)

        analyzer = AnalyzerNode()
        strategist = StrategistNode()
        asset_generator = AssetGeneratorNode(image_llm=image_llm)
        bg_removal = BackgroundRemovalNode(vision_llm=image_llm)

        analyzer >> strategist >> asset_generator >> bg_removal

        return cls(root=analyzer)

    async def run(
        self,
        maze_type: str,
        theme: str,
        output_dir: Path = Path("."),
    ) -> AutoEnvContext:
        """
        Execute pipeline.

        Args:
            maze_type: 迷宫类型（maze, pathfinder, sokoban, trapfield）
            theme: 视觉主题（如 "cyberpunk neon city"）
            output_dir: 输出目录
        """
        ctx = AutoEnvContext(
            maze_type=maze_type,
            theme=theme,
            output_dir=output_dir,
        )
        return await super().run(ctx)
