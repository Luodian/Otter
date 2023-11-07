import requests
import base64
from .base_model import BaseModel
from PIL import Image
import io
import time


def get_pil_image(raw_image_data) -> Image.Image:
    if isinstance(raw_image_data, Image.Image):
        return raw_image_data

    elif isinstance(raw_image_data, dict) and "bytes" in raw_image_data:
        return Image.open(io.BytesIO(raw_image_data["bytes"]))

    elif isinstance(raw_image_data, str):  # Assuming this is a base64 encoded string
        image_bytes = base64.b64decode(raw_image_data)
        return Image.open(io.BytesIO(image_bytes))

    else:
        raise ValueError("Unsupported image data format")


class OpenAIGPT4Vision(BaseModel):
    def __init__(self, api_key: str, max_new_tokens: int = 256):
        super().__init__("openai-gpt4", "gpt-4-vision-preview")
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def encode_image_to_base64(raw_image_data) -> str:
        if isinstance(raw_image_data, Image.Image):
            buffered = io.BytesIO()
            raw_image_data.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        raise ValueError("The input image data must be a PIL.Image.Image")

    def generate(self, text_prompt: str, raw_image_data):
        raw_image_data = get_pil_image(raw_image_data).convert("RGB")
        base64_image = self.encode_image_to_base64(raw_image_data)

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
        }

        retry = True
        retry_times = 0
        while retry and retry_times < 5:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
            if response.status_code == 200:
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
            else:
                print(f"Failed to connect to OpenAI API: {response.status_code} - {response.text}. Retrying...")
                time.sleep(10)
                retry_times += 1
        return "Failed to connect to OpenAI GPT4V API"

    def eval_forward(self, **kwargs):
        return super().eval_forward(**kwargs)


if __name__ == "__main__":
    # Use your own API key here
    api_key = "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"
    model = OpenAIGPT4Vision(api_key)
    image = Image.open("/home/luodian/projects/Otter/archived/data/G4_IMG_00001.png").convert("RGB")
    print(model.generate("What’s in this image?", image))
