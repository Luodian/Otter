import cv2
import numpy as np
import re
from word2number import w2n


# TODO: 出力csvと正解csvを比較して、検出結果の精度を出す
def accuracy():
    return


def extract_number_from_response(response):
    try:
        result = w2n.word_to_num(response)
    except:
        result = 0
    return result
