"""
This file contains the implementation of the DenseCaptions and TVCaptions datasets.
"""

import json

from abstract_dataset import AbstractDataset


class DenseCaptions(AbstractDataset):
    def __init__(
        self,
        name: str = "DenseCaptions",
        prompt_path: str = "prompts/dense_captions.json",
        query_inputs_path: str = "annotations/dense_captions/train.json",
    ):
        super().__init__(name, prompt_path, query_inputs_path)

    def _load_query_inputs(self, path: str) -> list[dict[str, str]]:
        with open(path, "r") as f:
            json_data = json.load(f)
        query_inputs = []
        for item in json_data:
            now_video = {}
            now_video["id"] = item
            now_query_input = ""
            now_time_stamps = json_data[item]["timestamps"]
            for i in range(len(now_time_stamps)):
                now_time_stamps[i][0] = round(float(now_time_stamps[i][0]))
                now_time_stamps[i][1] = round(float(now_time_stamps[i][1]))
            now_query_input += "timestamps: " + str(now_time_stamps) + "\n"
            now_query_input += "sentences: " + json.dumps(json_data[item]["sentences"])
            query_inputs.append(
                {
                    "id": item,
                    "sentences": now_query_input,
                }
            )
        return query_inputs


class TVCaptions(AbstractDataset):
    def __init__(
        self,
        name: str = "TVCaptions",
        in_context_path: str = "prompts/tv_captions.json",
        annotation_path: str = "annotations/tvc_annotations/tvc_train_release.jsonl",
    ):
        super().__init__(name, in_context_path, annotation_path)

    def _load_query_inputs(self, path: str) -> list[dict[str]]:
        query_inputs = []
        with open(path, "r") as json_file:
            for json_str in json_file:
                video = json.loads(json_str)
                video_id = video["vid_name"]
                now_query_input = []
                for disc_id, desc in enumerate(video["descs"], 1):
                    now_query_input.append(str(disc_id) + ". " + desc["desc"])
                query_inputs.append({"id": video_id, "sentences": "\n".join(now_query_input)})
        return query_inputs


class VisualStoryTelling(AbstractDataset):
    def __init__(
        self,
        name: str = "VisualStoryTelling",
        in_context_path: str = "prompts/visual_story_telling.json",
        annotation_path: str = "annotations/story_in_sequence/train.story-in-sequence.json",
    ):
        super().__init__(name, in_context_path, annotation_path)

    def generate_single_query_input(self, album: dict):
        query_input = ""
        query_input += "title: " + album["title"] + "\n"
        query_input += "description: " + album["description"] + "\n"
        for image in album["images"]:
            query_input += "\n"
            query_input += "image: " + image["title"] + "\n"
            query_input += "tags: " + image["tags"] + "\n"
            query_input += "annotations: " + json.dumps(image["annotations"]) + "\n"
        return query_input

    def _load_query_inputs(self, path: str) -> list[dict[str]]:
        query_inputs = []
        with open(path) as f:
            json_data = json.load(f)

            # create images dictionary and add basic information
            images = {}
            for image in json_data["images"]:
                # url = ""
                # if "url_o" not in image:
                #     url = image["url_m"]
                # else:
                #     url = image["url_o"]
                images[image["id"]] = {
                    "title": image["title"],
                    # "url": url,
                    "tags": image["tags"],
                    "annotations": [],
                }

            # add annotations to images
            for annotation_list in json_data["annotations"]:
                for annotation in annotation_list:
                    images[annotation["photo_flickr_id"]]["annotations"].append(annotation["text"])

            # create albums dictionary and add basic information
            albums = {}
            for album in json_data["albums"]:
                albums[album["id"]] = {
                    "description": album["description"],
                    "title": album["title"],
                    "images": [],
                }

            # add images to albums
            for image in json_data["images"]:
                albums[image["album_id"]]["images"].append(images[image["id"]])

            # create query inputs
            for album in albums:
                query_inputs.append(
                    {
                        "id": album,
                        "sentences": self.generate_single_query_input(albums[album]),
                    }
                )

        return query_inputs
