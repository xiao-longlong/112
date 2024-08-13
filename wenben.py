# coding: utf-8
import tr
import codecs
import sys, cv2, time, os
from PIL import Image
import numpy as np

# 获取当前脚本所在的目录，并将工作目录切换到该目录
_BASEDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASEDIR)

# 读取文件并检测关键字
def detect_keywords_in_file(file_path):
    # 关键字组列表，每个组的关键字对应特定的编码值
    keyword_groups = [
        {"轴突": "1", "树突": "2", "神经元": "3"},  # 第一个关键字组
        {"筋膜": "1", "肌肉": "2", "肌纤维": "3"},  # 第二个关键字组
        {"神经元": "01", "胶质细胞": "02", "突触": "03"},  # 第三个关键字组
    ]

    # keyword_groups = [
    #     {"检测": "1", "跟踪": "2", "巡航": "3"},  # 任务
    #     {"静止": "1", "运动": "2", "机动": "3"},  # 目标
    #     {"样条曲线": "01", "昏暗条件": "02", "光照条件": "03"},  # 数据集
    # ]
    
    # 以带有自动检测编码的方式打开文件
    with codecs.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # 初始化一个长度为4的列表来存储每个位的编码，初始值为 "0"
    code_positions = ["0", "0", "00"]

    # 遍历文件的每一行
    for line_num, line in enumerate(lines, 1):
        for i, group in enumerate(keyword_groups):
            for keyword, code in group.items():
                if keyword in line:
                    code_positions[i] = code
    
    # 将结果合并为一个四位的编码
    final_code = ''.join(code_positions)
    
    return final_code

# 读取文件进行文字识别
def text_detect(img_path):
    
    # 打开图像文件
    img_pil = Image.open(img_path)
    
    try:
        # 检查图像是否包含EXIF信息（用于获取图像的旋转信息）
        if hasattr(img_pil, '_getexif'):
            # 定义EXIF中的方向键
            orientation = 274
            # 获取EXIF信息并转换为字典
            exif = dict(img_pil._getexif().items())
            # 根据EXIF的方向信息调整图像的方向
            if exif[orientation] == 3:
                img_pil = img_pil.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img_pil = img_pil.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img_pil = img_pil.rotate(90, expand=True)
    except:
        # 如果处理EXIF信息时发生异常，则忽略
        pass

    # 定义图像的最大尺寸，如果图像超过此尺寸，则进行缩放
    MAX_SIZE = 1600
    if img_pil.height > MAX_SIZE or img_pil.width > MAX_SIZE:
        # 计算缩放比例，确保图像的尺寸不超过最大值
        scale = max(img_pil.height / MAX_SIZE, img_pil.width / MAX_SIZE)
        new_width = int(img_pil.width / scale + 0.5)
        new_height = int(img_pil.height / scale + 0.5)
        # 按比例缩放图像
        img_pil = img_pil.resize((new_width, new_height), Image.ANTIALIAS)

    # 将图像转换为灰度图像，用于文字检测
    gray_pil = img_pil.convert("L")

    # 记录当前时间，用于计算文字检测的时间
    t = time.time()
    n = 1  # 循环次数，这里设置为1
    for _ in range(n):
        # 执行文字检测，结果以矩形区域表示
        tr.detect(gray_pil, flag=tr.FLAG_RECT)
    # 计算并打印检测所用的平均时间
    print("time", (time.time() - t) / n)

    # 使用旋转矩形进行文字检测，获取检测结果
    results = tr.run(gray_pil, flag=tr.FLAG_ROTATED_RECT)

    # 指定保存检测结果的文件路径和名称
    file_path = "recognized_text.txt"

    # 将检测到的文字写入到TXT文件中
    with open(file_path, 'w', encoding='utf-8') as file:
        for rect in results:
            file.write(rect[1] + '\n')  # 写入检测到的文字内容，并换行

    # 打印提示信息，表示文本已保存
    print(f"文本已保存到 {file_path}")

    return file_path   

if __name__ == "__main__":
    img_path = "/workspace/wenbenjiance/屏幕截图 2024-08-08 175654.png"
    # 调用文本检测函数，开始进行文字检测
    file_path = text_detect(img_path)
    # 调用关键字检测函数
    result_code = detect_keywords_in_file(file_path)
    # 输出检测结果
    print(f"最终编码为: {result_code}")
    
