import cv2
import numpy as np


# 出力csvと正解csvを比較して、検出結果の精度を出す
def accuracy():
    return


# TODO：出力ファイルの指定をなくす
def remove_background(input_video_path):
    output_video_path = "output_temp.MP4"
    cap = cv2.VideoCapture(input_video_path)

    # 動画のフレームサイズを取得
    # TODO: マジックナンバーを消す
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    video_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_len_sec = video_frame_count / video_fps

    # 出力動画ファイルの設定
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height), isColor=True)

    # 背景差分法の初期化
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 背景差分法を適用して前景を取得
        fg_mask = fgbg.apply(frame)

        # 前景と背景の差分を反転させて背景を取得
        bg_mask = cv2.bitwise_not(fg_mask)

        # 背景をマスクする
        result = cv2.bitwise_and(frame, frame, mask=fg_mask)

        # 出力動画にフレームを書き込み
        out.write(result)

        if cv2.waitKey(int(video_len_sec)):
            break

    # リソースを解放
    cap.release()
    out.release()
