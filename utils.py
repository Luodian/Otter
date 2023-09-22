import csv
from enum import IntEnum
import cv2
import numpy as np
import re
from word2number import w2n

FISH_TYPES = 5


class FishNames(IntEnum):
    HIMEDAKA = 1
    NEON_TETRA = 2
    GLASS_CATFISH = 3
    GUPPY = 4
    RYUKIN = 5

    @staticmethod
    def from_name(name):
        for v in FishNames:
            if name == v.name:
                return int(v)
        raise ValueError(f"{name} is not a valid FishNames!")


def accuracy(result, ground_truth):
    count = 0
    for v in FishNames:
        if result[int(v)] == ground_truth[int(v)]:
            count += 1
    accuracy = 100 * (count / FISH_TYPES)
    return accuracy


def output_csv(result, input_video_path, output_csv_path):
    with open(output_csv_path, "w", encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([input_video_path])
        csv_writer.writerow([FISH_TYPES])
        for key in sorted(result.keys()):
            value = result[key]
            csv_writer.writerow([key, value])
    return


def convert_fish_name_to_number(result):
    converted_result = {}
    for k, v in result.items():
        key = FishNames.from_name(k)
        converted_result[key] = v
    return converted_result


def extract_number_from_response(response):
    try:
        result = w2n.word_to_num(response)
    except:
        result = 0
    return result
