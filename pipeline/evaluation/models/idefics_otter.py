from PIL import Image
import torch
from transformers import AutoProcessor
from transformers import IdeficsForVisionText2Text

from pipeline.train.train_utils import get_image_attention_mask


class OtterIdefics(Model):
    def __init__(
        self,
        checkpoint: str = "/data/pufanyi/training_data/checkpoints/otter_idefics9b_0830",
        processor: str = "/data/pufanyi/training_data/checkpoints/idefics-80b-instruct",
    ) -> None:
        super().__init__("idefics_otter", checkpoint)
        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        self.model = IdeficsForVisionText2Text.from_pretrained(checkpoint, **kwargs)
        self.processor = AutoProcessor.from_pretrained(processor)
        self.image_processor = self.processor.image_processor
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.model.eval()

    def move_to_device(self, device):
        pass

    def generate(self, question, raw_image, device=None, keep_in_device=False):
        prompts = [
            [
                "User:",
                raw_image,
                question + "<end_of_utterance>\n",
                "Assistant:",
            ],
        ]

        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        inputs = self.processor(prompts, return_tensors="pt", add_end_of_utterance_token=False, debug=True).to(self.model.device)
        generated_ids = self.model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_new_tokens=128)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text[0]

    def otter_idefics_generate(self, question, raw_image, no_image_flag=False):
        input_data = raw_image
        if isinstance(input_data, Image.Image):
            if no_image_flag:
                vision_x = torch.zeros(1, 1, 3, 224, 224, dtype=next(self.model.parameters()).dtype)
            else:
                vision_x = self.image_processor.preprocess([input_data], return_tensors="pt").unsqueeze(0)
        else:
            raise ValueError("Invalid input data. Expected PIL Image.")

        lang_x = self.tokenizer(
            [
                self.get_formatted_prompt(question, no_image_flag=no_image_flag),
            ],
            return_tensors="pt",
        )

        model_dtype = next(self.model.parameters()).dtype
        vision_x = vision_x.to(dtype=model_dtype)
        # vision_x = self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(0)
        lang_x = self.tokenizer(
            [
                self.get_formatted_prompt(question, no_image_flag=no_image_flag),
            ],
            return_tensors="pt",
        )
        print("------------------------------------------------------------")
        print(self.get_formatted_prompt(question, no_image_flag=no_image_flag))
        image_attention_mask = get_image_attention_mask(lang_x["input_ids"], 1, self.tokenizer)
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        generated_text = self.model.generate(
            pixel_values=vision_x.to(self.model.device),
            input_ids=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            image_attention_mask=image_attention_mask.to(self.model.device),
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            max_new_tokens=512,
        )
        output = self.tokenizer.decode(generated_text[0])

        return output

    def get_formatted_prompt(self, question: str, no_image_flag: str) -> str:
        if no_image_flag:
            return f"User:{question}<end_of_utterance>\nAssistant:<answer>"
        else:
            return f"User:<fake_token_around_image><image><fake_token_around_image>{question}<end_of_utterance>\nAssistant:<answer>\n"


if __name__ == "__main__":
    model = OtterIdefics()
    image = Image.open("/data/pufanyi/project/Otter-2/pipeline/evaluation/test_data/test.jpg")
    response = model.generate("What is this?", image)
    print(response)
    response = model.generate("What is this?", image)
    print(response)
