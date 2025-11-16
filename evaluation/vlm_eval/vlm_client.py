import base64
import os
import logging
from openai import OpenAI


class VLMClient:
    """API模型客户端"""
    def __init__(self, model: str = "gpt-4o", api_key: str = None, base_url: str = None, max_tokens: int = 10000, temperature: float = 0.0):
        self.model = model
        self.maxtokens = max_tokens
        self.temperature = temperature
        self.model_type = "api"
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
    
    def query(self, system_prompt: str, user_prompt: str, image_path: str = None) -> str:
        try:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]

            if image_path:
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")

                user_content = [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            else:
                user_content = user_prompt

            messages.append({
                "role": "user",
                "content": user_content
            })

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.maxtokens,
                temperature=self.temperature
            )

            print(response.choices[0].message.content)

            return response.choices[0].message.content

        except FileNotFoundError as e:
            logging.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logging.error(f"VLM API call failed: {type(e).__name__}: {e}")
            raise


