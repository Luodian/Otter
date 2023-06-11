import json
import random
from typing import List, Dict
import re
from abstract_dataset import AbstractDataset


class SceneNavigation(AbstractDataset):
    """
    Implementation of the SceneNavigation dataset.
    """

    def __init__(
        self,
        name: str = "SceneNavigation",
        in_context_path: str = "prompts/scene_navigation.json",
        query_inputs_path: str = "annotations/scene_navigation/scan_info.json",
    ):
        super().__init__(name, in_context_path, query_inputs_path)

    def _load_query_inputs(self, path: str) -> List[Dict[str, str]]:
        """
        Load the query inputs from the given path and return them as a list of dictionaries.
        """
        with open(path, "r") as f:
            query_inputs = json.load(f)

        results = []
        for scene_id, inner_dict in query_inputs.items():
            descriptions = inner_dict["description"]
            random.shuffle(descriptions)
            formatted_descriptions = [cur_description[1] for cur_description in descriptions[:50]]
            results.append(
                {
                    "id": scene_id,
                    "sentences": "\n".join(formatted_descriptions),
                }
            )

        return results


class EGO4D(AbstractDataset):
    """
    Implementation of the EGO4D dataset.
    """

    def __init__(
        self,
        name: str = "EGO4D",
        in_context_path: str = "prompts/ego4d.json",
        query_inputs_path: str = "annotations/ego4d/processed_all_anns.json",
    ):
        super().__init__(name, in_context_path, query_inputs_path)

    def _get_restrict_words(self):
        # sample = 'You are not allowed to include exact timestamps. For example, timestamp 107.7 is not allowed in your outputs. Do not mention "Based on the description". You could say "According what I observed".\nYou are not allowed to include exact timestamps in your outputs. For example, timestamp 107.7 is not allowed in your outputs. Do not mention "Based on the descriptions". You could say "According what I observed".\nYou are not allowed to mention woman X, woman Y, man X, man Y, Person A, Person B, in your outputs, STRICTLY! You are only allowed to refer them as "the person" or "a person". Also remember, person C or C both refer to the cameraman, or the user, so you cannot use the word cameraman in your questions or answers. Since cameraman is user, you should refer cameraman as "I" in the questions, and "you" in the answers.\nPlease ask at least 6 questions. Questions should be the first-person view (I or me). Questions that are valuable and suggestive are preferable, get rid of simple questions like "what do I see in the video?". Also remember what you see is the real world, so do not refer what you see as "videos". For example, it is unnatural to have the question "Do you see any flotation devices in the video that I could use for support?", and the better question is "Do you see any flotation devices around that I could use for support?". Also, make sure the generated questions and answers are concise.'

        sample = "Remember, in your responses, avoid directly referencing specific timestamps, such as 'timestamp 107.7'. Instead, refer to events or objects observed in the sequence of events. Rather than stating 'Based on the descriptions', you should phrase it as 'According to what I observed' to emphasize the perspective of an observing AI. You are not allowed to mention woman X, woman Y, man X, man Y, Person A, Person B, in your outputs, STRICTLY! You are only allowed to refer them as 'the person' or 'a person'. Additionally, 'Person C' or 'C' are synonymous with the user or the individual wearing the AR glasses. Avoid using the term 'cameraman'; instead, refer to the user as 'I' in questions and 'you' in answers. Strive to produce at least six questions in your output, maintaining a first-person perspective (using 'I' or 'me'). Prioritize generating valuable and suggestive questions over simplistic ones like 'what do I see?'. Keep in mind that what you are observing is the real world, not a video, so avoid referring to it as 'footage' or 'video', but use some words like 'my observation' or 'what I saw'. For instance, a question like 'Do you see any flotation devices in the video that I could use for support?' should be replaced with 'Do you see any flotation devices around that I could use for support?'. Lastly, ensure the questions and answers you generate are concise and succinct for clear communication."
        return sample

    def _load_query_inputs(self, path: str) -> List[Dict[str, str]]:
        """
        Load the query inputs from the given path and return them as a list of dictionaries.
        """
        with open(path, "r") as f:
            query_inputs = json.load(f)

        results = []
        for video_name, video_data in query_inputs.items():
            processed_timestamps = set()
            for clip_id, clip_data in enumerate(video_data["clips"]):
                narrations = clip_data.get("narrations", [])
                formatted_descriptions = []
                for narration in narrations:
                    timestamp = narration.get("time", 0)
                    rounded_timestamp = round(timestamp)
                    description = narration["text"]
                    dense_caption = narration["object_description"]
                    dense_caption = "; ".join(dense_caption)
                    if rounded_timestamp in processed_timestamps:
                        formatted_descriptions.append(f"description: {description}")
                    else:
                        processed_timestamps.add(rounded_timestamp)
                        formatted_descriptions.append(f"timestamp: {rounded_timestamp}\ndescription: {description}\nobjects: {dense_caption}")

                formatted_descriptions = "\n".join(formatted_descriptions)
                formatted_descriptions = formatted_descriptions + "\n" + self._get_restrict_words()
                filled_clip_id = str(clip_id).zfill(6)
                results.append(
                    {
                        "id": f"{video_name}_clip{filled_clip_id}",
                        "sentences": formatted_descriptions,
                    }
                )

        return results
