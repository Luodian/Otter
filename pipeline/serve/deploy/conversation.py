import dataclasses
from enum import auto, Enum
from typing import List, Tuple

import io
import base64
import os
from PIL import Image
import copy

IMG_FLAG = "<image>"


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


def decode_image(encoded_image: str) -> Image:
    decoded_bytes = base64.b64decode(encoded_image.encode("utf-8"))
    buffer = io.BytesIO(decoded_bytes)
    image = Image.open(buffer)
    return image


def encode_image(image: Image.Image, format: str = "PNG") -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format=format)
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded_image


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[dict]  # multi-turn -> user & assistant -> {'images': [PIL.Image,], 'text': str}
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = copy.deepcopy(self.messages)
        if self.sep_style == SeparatorStyle.SINGLE:
            if self.system is None or self.system == "":
                text = ""
            else:
                text = self.system + self.sep
            images = []
            for message in messages:
                text += message["role"] + ": " + message["message"]["text"] + self.sep
                for image_path, image_ids in zip(message["message"]["images"], message["message"]["images_ids"]):
                    if image_ids is not None:
                        images.append(image_ids)
                    else:
                        image = Image.open(image_path).resize((256, 256))
                        image_base64 = encode_image(image)
                        images.append(image_base64)

            text += self.roles[1] + ":"
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            # b_token = "[INST] "
            b_token = "[USER] "
            # e_token = " [/INST]"
            e_token = " [ASSISTANT]"
            if self.system is None or self.system == "":
                text = ""
            else:
                text = f"<<SYS>>\n{self.system}\n<</SYS>>\n\n"
            images = []
            for idx, message in enumerate(messages):
                # text += message['role'] + ": " + message['message']['text'] + self.sep
                if idx % 2 == 0:
                    text += b_token + message["message"]["text"] + e_token + self.sep
                else:
                    text += message["message"]["text"] + self.sep

                for image_path, image_ids in zip(message["message"]["images"], message["message"]["images_ids"]):
                    if image_ids is not None:
                        images.append(image_ids)
                    else:
                        image = Image.open(image_path).resize((256, 256))
                        image_base64 = encode_image(image)
                        images.append(image_base64)
        else:
            raise NotImplementedError

        return {"text": text, "images": images}

    def update_image_ids(self, images_ids):
        image_count = 0
        for message in self.messages:
            for idx in range(len(message["message"]["images_ids"])):
                if message["message"]["images_ids"][idx] is None:
                    message["message"]["images_ids"][idx] = images_ids[image_count]
                image_count += 1

        assert len(images_ids) == image_count, print(len(images_ids), image_count)

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        dialog = []
        for i, single_turn in enumerate(self.messages[self.offset :]):
            single_turn = single_turn["message"]
            text_list = single_turn["text"].split(IMG_FLAG)
            assert len(text_list) == len(single_turn["images"]) + 1, print(text_list, len(single_turn["images"]))
            message = ""
            for image_idx in range(len(single_turn["images"])):
                # image = single_turn['images'][image_idx]
                # image_base64 = encode_image(image)
                # image_str = f'<img src="data:image/png;base64,{image_base64}" alt="user upload image" />'
                image_path = single_turn["images"][image_idx]
                if image_path == "":
                    message += text_list[image_idx] + "<corrupt_image>"
                else:
                    message += text_list[image_idx] + f"![](file={image_path})"
            message += text_list[-1]

            if i % 2 == 0:
                dialog.append([message, None])
            else:
                dialog[-1][-1] = message

        return dialog

    def copy(self):
        return Conversation(system=self.system, roles=self.roles, messages=copy.deepcopy(self.messages), offset=self.offset, sep_style=self.sep_style, sep=self.sep, sep2=self.sep2, version=self.version)

    def dict(self):
        messages = copy.deepcopy(self.messages)
        for message in messages:
            if "images_ids" in message:
                message.pop("images_ids")
            for i in range(len(message["message"]["images"])):
                message["message"]["images"][i] = os.path.basename(message["message"]["images"][i])
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


model = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="\n",
)
