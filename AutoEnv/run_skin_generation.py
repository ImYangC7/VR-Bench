"""
Maze Mode Skin Generation Entry Point

Generates visual assets for maze-type games using VisualPipeline.

Usage:
    python run_skin_generation.py --maze-type maze --theme "cyberpunk neon city"
    python run_skin_generation.py --maze-type sokoban --theme "medieval castle"
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from autoenv.pipeline import VisualPipeline
from base.engine.cost_monitor import CostMonitor

# Load environment variables from parent directory .env file (VR-Bench/.env)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

DEFAULT_CONFIG = "config/env_skin_gen.yaml"


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


async def run_skin_gen(
    image_model: str,
    output_dir: Path,
    maze_type: str,
    theme: str,
):
    """
    Run skin generation pipeline for maze mode.

    Args:
        image_model: Image generation model name
        output_dir: Output directory
        maze_type: Maze type (maze, pathfinder, sokoban, trapfield)
        theme: Visual theme (e.g., "cyberpunk neon city")
    """
    # Determine output location with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    visual_output = output_dir / f"{maze_type}_{ts}"
    visual_output.mkdir(parents=True, exist_ok=True)

    label = f"{maze_type}_{theme[:20]}"
    print(f"🎨 [{label}] Generating visuals...")
    print(f"   Mode: Maze ({maze_type})")
    print(f"   Theme: {theme}")

    pipeline = VisualPipeline.create_default(image_model=image_model)

    ctx = await pipeline.run(
        maze_type=maze_type,
        theme=theme,
        output_dir=visual_output,
    )

    if ctx.generated_assets:
        print(f"✅ [{label}] Visuals generated → {visual_output}")
        print(f"   Assets: {len(ctx.generated_assets)} files")
    else:
        print(f"❌ [{label}] Visual generation failed: {ctx.error}")


async def main():
    parser = argparse.ArgumentParser(
        description="Generate visual skins for maze-type games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_skin_generation.py --maze-type maze --theme "cyberpunk neon city"
  python run_skin_generation.py --maze-type sokoban --theme "medieval castle"
  python run_skin_generation.py --maze-type pathfinder --theme "candy land"
  python run_skin_generation.py --maze-type trapfield --theme "sci-fi space"
        """,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")

    # Maze mode arguments (required)
    parser.add_argument(
        "--maze-type",
        choices=["maze", "pathfinder", "sokoban", "trapfield"],
        help="Maze type (maze/pathfinder/sokoban/trapfield)",
    )
    parser.add_argument(
        "--theme", help="Visual theme (e.g., 'cyberpunk neon city')"
    )

    # Model arguments
    parser.add_argument("--image-model", help="Override: image model name")
    parser.add_argument("--output", help="Override: output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI args override config
    image_model = args.image_model or cfg.get("image_model")
    output = args.output or cfg.get("envs_root_path") or "workspace/envs"
    maze_type = args.maze_type or cfg.get("maze_type")
    theme = args.theme or cfg.get("theme")

    if not image_model:
        print(
            "❌ No image_model configured. "
            "Set 'image_model' in config or --image-model"
        )
        return

    # Validate maze mode
    if not maze_type or not theme:
        print("❌ Both --maze-type and --theme are required")
        print("   Example: --maze-type maze --theme 'cyberpunk neon city'")
        return

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Config: {args.config}")
    print(f"🎨 Image Model: {image_model}")
    print(f"📁 Output: {output}")
    print(f"🎮 Maze Type: {maze_type}")
    print(f"🎨 Theme: {theme}")

    with CostMonitor() as monitor:
        await run_skin_gen(
            image_model=image_model,
            output_dir=output_dir,
            maze_type=maze_type,
            theme=theme,
        )

        # Print and save cost summary
        summary = monitor.summary()
        print("\n" + "=" * 50)
        print("💰 Cost Summary")
        print("=" * 50)
        print(f"Total Cost: ${summary['total_cost']:.4f}")
        print(f"Total Calls: {summary['call_count']}")
        print(f"Input Tokens: {summary['total_input_tokens']:,}")
        print(f"Output Tokens: {summary['total_output_tokens']:,}")

        if summary["by_model"]:
            print("\nBy Model:")
            for model_name, stats in summary["by_model"].items():
                print(
                    f"  {model_name}: ${stats['cost']:.4f} ({stats['calls']} calls)"
                )

        cost_file = monitor.save()
        print(f"\n📊 Cost saved: {cost_file}")


if __name__ == "__main__":
    asyncio.run(main())
