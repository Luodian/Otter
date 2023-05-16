import dataclasses
from enum import auto, Enum
from typing import List, Any
import base64
from io import BytesIO


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                role = None  # hard-coded to remove role labels for Apr 29 model

                if message:
                    if type(message) is tuple:
                        message = message[0]

                    if role is None:
                        ret += message + seps[i % 2]
                    else:
                        ret += role + ": " + message + seps[i % 2]
                else:
                    if role is not None:
                        ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                msg = list(msg)
                msg, image_list = msg[0], msg[1:]
                for image in image_list:
                    if image is not None:
                        max_hw, min_hw = max(image.size), min(image.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 800, 400
                        shortest_edge = int(
                            min(max_len / aspect_ratio, min_len, min_hw)
                        )
                        longest_edge = int(shortest_edge * aspect_ratio)
                        H, W = image.size
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((H, W))
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                msg = list(msg)
                msg, images = msg[0], msg[1:]
                for image in images:
                    if image is not None:
                        max_hw, min_hw = max(image.size), min(image.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 800, 400
                        shortest_edge = int(
                            min(max_len / aspect_ratio, min_len, min_hw)
                        )
                        longest_edge = int(shortest_edge * aspect_ratio)
                        H, W = image.size
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((H, W))
                        # image = image.resize((224, 224))
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                        msg = msg.replace("<image>", img_str, 1)
                # hard-coded post processing for Otter
                if msg.rfind("GPT:") != -1:
                    msg = msg[: msg.rfind("GPT:")]  # remove the last GPT label
                msg = msg.replace("GPT:", "\nOtter:")
                msg = msg.replace("<|endofchunk|>", "")
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id,
        )

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [
                    [x, y[0] if type(y) is tuple else y] for x, y in self.messages
                ],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Give three tips for staying healthy."),
        (
            "Assistant",
            "Sure, here are three tips for staying healthy:\n"
            "1. Exercise regularly: Regular physical activity can help improve your overall health and wellbeing. "
            "It can also help reduce your risk of chronic conditions such as obesity, diabetes, heart disease, "
            "and certain cancers. Aim for at least 150 minutes of moderate-intensity aerobic exercise or "
            "75 minutes of vigorous-intensity aerobic exercise per week, along with muscle-strengthening "
            "activities at least two days per week.\n"
            "2. Eat a balanced diet: Eating a balanced diet that is rich in fruits, "
            "vegetables, whole grains, lean proteins, and healthy fats can help support "
            "your overall health. Try to limit your intake of processed and high-sugar foods, "
            "and aim to drink plenty of water throughout the day.\n"
            "3. Get enough sleep: Getting enough quality sleep is essential for your physical "
            "and mental health. Adults should aim for seven to nine hours of sleep per night. "
            "Establish a regular sleep schedule and try to create a relaxing bedtime routine to "
            "help improve the quality of your sleep.",
        ),
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_v1_2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        (
            "Human",
            "What are the key differences between renewable and non-renewable energy sources?",
        ),
        (
            "Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n",
        ),
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_bair_v1 = Conversation(
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

otter_v1 = Conversation(
    system="",
    roles=("User", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

open_flamingo_v1 = Conversation(
    system="",
    roles=(None, None),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="",
    sep2="</s>",
)


default_conversation = otter_v1
conv_templates = {
    "v1": conv_v1_2,
    "bair_v1": conv_bair_v1,
    "otter": otter_v1,
    "open_flamingo": open_flamingo_v1,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
