import cv2
import torch
import argparse

from components.hand_detect.yolo_v3_hand import yolo_v3_hand_model
from components.hand_keypoints.handpose_x import handpose_x_model
from components.hand_track.hand_track import pred_gesture
from lib.hand_lib.utils.utils import parse_data_cfg
from components.hand_gesture.resnet import resnet18


def handpose_x_process(config, video_path):
    # 模型初始化
    print("load model component  ...")
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

    gesture_to_number = {
        "fist": 0,
        "one": 1,
        "yearh": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "heartSingle": 7,
        "gun": 8,
        "thunbUp": 9,
    }

    gesture_history = []
    output_flag = []
    four_same_gestures_confirmed = 0  # 用于记录确认的四次相同十帧手势

    while True:
        # if four_same_gestures_confirmed < 4:
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

            print(gesture_name)  # 这个地方需要修改一下，首先打印的内容需要修改改为某种编码，其次需要将这个编码传递给其他模块
            gesture_history.append(gesture_name)

            if len(gesture_history) >= 20 and gesture_history[-20:].count(gesture_history[-1]) == 20:  # 映射手势名称到数字

                gesture_number = gesture_to_number.get(gesture_history[-1], 0)
                print("连续10帧相同，手势数字:", gesture_number)
                output_flag.append(gesture_number)
                # 重置手势历史记录
                gesture_history = []
                four_same_gestures_confirmed += 1
                print("第x次:", four_same_gestures_confirmed)

                # 检查是否达到四次连续三帧相同手势的识别结果
                if len(output_flag) >= 4:
                    recent_four = output_flag[:4]
                    result = "".join(str(num) for num in recent_four)  # 将数字列表转换为字符串
                    print("四次连续结果:", result)
                    break

        #     # 在原始帧上绘制矩形和标签
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(frame, "LABEL {}".format(gesture_name), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        #     break

        # # 显示结果
        # cv2.imshow("image", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):  # 按'q'退出
        #     break

        if four_same_gestures_confirmed >= 4:
            print("已确认四次连续十帧相同手势，退出程序。")
            break  # 退出循环


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=" Project Hand Pose Inference")
    parser.add_argument("--cfg_file", type=str, default="lib/hand_lib/cfg/handpose.cfg", help="model_path")  # 模型路径
    parser.add_argument("--video_path", type=str, default="0", help="0 for cam / path ")  # 是否视频

    print("\n/******************* {} ******************/\n".format(parser.description))
    args = parser.parse_args()  # 解析添加参数
    config = parse_data_cfg(args.cfg_file)
    video_path = args.video_path

    handpose_x_process(config, video_path)
