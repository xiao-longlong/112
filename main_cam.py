import os
import cv2
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

# 加载模型组件库
from components.hand_detect.yolo_v3_hand import yolo_v3_hand_model
from components.hand_keypoints.handpose_x import handpose_x_model
# from components.hand_track.hand_track import Tracker
from components.hand_track.hand_track import pred_gesture
from lib.hand_lib.utils.utils import parse_data_cfg

from components.hand_gesture.resnet import resnet18

import torch
import argparse


def handpose_x_process(config, is_videos, video_path):
    # 模型初始化
    print("load model component  ...")
    # yolo v3 手部检测模型初始化
    if config["detect_model_arch"] == "yolo_v3":
        hand_detect_model = yolo_v3_hand_model(
            conf_thres=float(config["detect_conf_thres"]), nms_thres=float(config["detect_nms_thres"]), model_arch=config["detect_model_arch"], model_path=config["detect_model_path"]
        )
    else:
        print("error : 无效检测模型输入")
        return None
    # handpose_x 21 关键点回归模型初始化
    handpose_model = handpose_x_model(model_arch=config["handpose_x_model_arch"], model_path=config["handpose_x_model_path"])

    # 识别手势
    gesture_model = resnet18()
    gesture_model = gesture_model.cuda()
    gesture_model.load_state_dict(torch.load(config["gesture_model_path"]))
    gesture_model.eval()

    print("start handpose process ~")
    cap = cv2.VideoCapture(video_path if video_path != "0" else 0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        algo_img = frame.copy()
        hand_bbox = hand_detect_model.predict(frame, vis=False)  # 修改vis参数为False，因为我们会在下面手动绘制

        # 遍历检测到的每个手部边界框
        for h_box in hand_bbox:
            x_min, y_min, x_max, y_max, score = h_box

            # 以下代码使用x_min, y_min, x_max, y_max变量，而不是img.shape
            w_ = max(x_max - x_min, y_max - y_min)
            if w_ < 60:
                continue
            w_ = w_ * 1.26

            x_mid = (x_max + x_min) / 2
            y_mid = (y_max + y_min) / 2

            x1, y1, x2, y2 = int(x_mid - w_ / 2), int(y_mid - w_ / 2), int(x_mid + w_ / 2), int(y_mid + w_ / 2)

            # 确保裁剪区域不超出图像边界
            x1, y1, x2, y2 = x1, max(y1, 0), min(x2, frame.shape[1] - 1), min(y2, frame.shape[0] - 1)

            box = [x1, y1, x2, y2]

            # 预测手指关键点
            pts_ = handpose_model.predict(algo_img[y1:y2, x1:x2, :])
            gesture_name = pred_gesture(box, pts_, frame, gesture_model)  # 确保传入frame而不是algo_img
            print(gesture_name)

            # 在原始帧上绘制矩形和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "LABEL {}".format(gesture_name), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # 显示结果
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # 按'q'退出
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=" Project Hand Pose Inference")
    parser.add_argument("--cfg_file", type=str, default="lib/hand_lib/cfg/handpose.cfg", help="model_path")  # 模型路径
    # parser.add_argument(
    #     "--test_path", type=str, default="/mnt/data/RecogniseSIgn/simple-handpose-recognition/test_set", help="test_path"
    # )  # 测试图片路径 'weights/handpose_x_gesture_v1/handpose_x_gesture_v1/000-one' camera_id
    parser.add_argument("--is_video", type=bool, default=True, help="if test_path is video")  # 是否视频
    parser.add_argument("--video_path", type=str, default="0", help="0 for cam / path ")  # 是否视频

    print("\n/******************* {} ******************/\n".format(parser.description))
    args = parser.parse_args()  # 解析添加参数

    config = parse_data_cfg(args.cfg_file)
    is_videos = args.is_video
    # test_path = args.test_path
    video_path = args.video_path

    handpose_x_process(config, is_videos, video_path)
