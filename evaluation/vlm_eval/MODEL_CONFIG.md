# VLM 模型配置说明

## 模型类型

系统支持两种类型的 VLM 模型：

### 1. API 模型 (type: api)

通过 API 调用的远程模型，如 GPT-4、Gemini 等。

**配置示例：**
```yaml
models:
  - name: gpt-5
    type: api
    base_url: https://newapi.deepwisdom.ai/v1
    max_tokens: 60000
    temperature: 1.0
```

**参数说明：**
- `name`: 模型名称
- `type`: 必须为 `api`
- `base_url`: API 端点地址
- `max_tokens`: 最大生成 token 数
- `temperature`: 生成温度（0.0-2.0）

**环境变量：**
需要设置 `OPENAI_API_KEY` 环境变量（在 `.env` 文件中）

### 2. Local 模型 (type: local)

本地加载的 HuggingFace 模型。

**配置示例：**
```yaml
models:
  - name: Qwen/Qwen2-VL-7B-Instruct
    type: local
    device: cuda:0
    max_tokens: 10000
    temperature: 0.0
  
  - name: llava-hf/llava-v1.6-mistral-7b-hf
    type: local
    device: cuda:1
    max_tokens: 10000
    temperature: 0.0
```

**参数说明：**
- `name`: HuggingFace 模型名称或路径
- `type`: 必须为 `local`
- `device`: 运行设备（如 `cuda:0`, `cuda:1`, `cpu`）
- `max_tokens`: 最大生成 token 数
- `temperature`: 生成温度（0.0-2.0）

**模型加载逻辑：**
1. 首先尝试从本地缓存加载模型
2. 如果失败，自动从 HuggingFace 下载到 `/huggingface_model` 目录
3. 模型会被加载到指定的 GPU 设备上

**依赖安装：**
```bash
pip install transformers torch pillow accelerate

# 如果使用 Qwen2.5-VL 模型，还需要安装：
pip install qwen-vl-utils
```

## 完整配置示例

```yaml
game: maze
dataset: dataset/maze/1
output: vlm_eval_results/maze

models:
  # API 模型
  - name: gpt-5
    type: api
    base_url: https://newapi.deepwisdom.ai/v1
    max_tokens: 60000
    temperature: 1.0
  
  # Local 模型 - GPU 0
  - name: Qwen/Qwen2-VL-7B-Instruct
    type: local
    device: cuda:0
    max_tokens: 10000
    temperature: 0.0
  
  # Local 模型 - GPU 1
  - name: llava-hf/llava-v1.6-mistral-7b-hf
    type: local
    device: cuda:1
    max_tokens: 10000
    temperature: 0.0

workers: 10
max_levels: -1
assets_folder: skins/maze/1
```

## 多 GPU 使用

可以配置多个 local 模型在不同的 GPU 上运行：

```yaml
models:
  - name: model-1
    type: local
    device: cuda:0  # 第一张 GPU
    
  - name: model-2
    type: local
    device: cuda:1  # 第二张 GPU
    
  - name: model-3
    type: local
    device: cuda:2  # 第三张 GPU
```

## 注意事项

1. **API 模型**：需要确保网络连接正常，API key 有效
2. **Local 模型**：
   - 首次运行会下载模型，可能需要较长时间
   - 确保有足够的磁盘空间（`/huggingface_model` 目录）
   - 确保 GPU 显存足够（7B 模型约需 14GB 显存）
   - 可以使用 `device: cpu` 在 CPU 上运行（速度较慢）
3. **并行执行**：不同模型会并行评估，注意资源分配
4. **每个难度只测试后 24 个 case**

## 支持的模型示例

### Local 模型
- `Qwen/Qwen2-VL-7B-Instruct`
- `Qwen/Qwen2-VL-2B-Instruct`
- `llava-hf/llava-v1.6-mistral-7b-hf`
- `llava-hf/llava-1.5-7b-hf`
- 其他支持 `AutoModelForVision2Seq` 的模型

### API 模型
- GPT-4o, GPT-4V
- Gemini Pro Vision
- Claude 3 Vision
- 其他兼容 OpenAI API 的模型

