import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
)

# Load LoRA weights
pipe.load_lora(pipe.dit, f"path/to/your/lora/checkpoint.safetensors", alpha=1)

pipe.enable_vram_management()


input_image = Image.open("path/to/your/input_image.png").resize((512, 512))
video = pipe(
    prompt = """Your prompt here"""
,
    negative_prompt="",
    seed=0, tiled=True,
    height=512, width=512,
    input_image=input_image,
    num_frames=193,
)
save_video(video, "path/to/your/output_video.mp4", fps=15, quality=5)
