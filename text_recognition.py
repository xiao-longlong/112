import cv2
import numpy as np
import onnxruntime
import json
import os
import argparse
import statistics
from scipy.special import softmax
import fitz  # PyMuPDF
import tr
import codecs
import sys, time
from PIL import Image


# 获取当前脚本所在的目录，并将工作目录切换到该目录
_BASEDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASEDIR)

# 读取文件并检测关键字
def detect_keywords_in_file(file_path):
    # 关键字组列表，每个组的关键字对应特定的编码值
    keyword_groups = [
        {"集群侦查": "1", "单对单侦查": "2", "态势构建": "3"},
        {"飞机": "1", "坦克": "2", "装甲车": "3", "爱国者": "4"},
        {"复杂背景": "01", "有遮挡": "02", "光照多样": "03", "尺度变换": "04", "尺度差异": "05",
        "跨场景": "06", "迷彩": "07" , "夜视": "08",},
    ]
    
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
def text_detect(path):
    for img_path in path:
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
        # print("time", (time.time() - t) / n)

        # 使用旋转矩形进行文字检测，获取检测结果
        results = tr.run(gray_pil, flag=tr.FLAG_ROTATED_RECT)

        # 指定保存检测结果的文件路径和名称
        file_path = "recognized_text.txt"

        # 将检测到的文字写入到TXT文件中
        with open(file_path, 'a', encoding='utf-8') as file:
            for rect in results:
                file.write(rect[1] + '\n')  # 写入检测到的文字内容，并换行

    # 打印提示信息，表示文本已保存
    # print(f"文本已保存到 {file_path}")

    return file_path   

# 打开 PDF 文件
def pdf2img(pdf_path):
    doc = fitz.open(pdf_path)
    out_path = []

    # 遍历每一页，将其转换为图像
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # 加载页面
        pix = page.get_pixmap()  # 将页面渲染为像素图
        
        # 保存图像
        output_image_path = f'page_{page_num + 1}.png'
        pix.save(output_image_path)

        # print(f'Page {page_num + 1} saved as {output_image_path}')

        out_path.append(output_image_path)

    return out_path

def read_vocab(path):
    """
    加载词典
    """
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab

def do_norm(x):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    x = x / 255.0
    x[0, :, :] -= mean[0]
    x[1, :, :] -= mean[1]
    x[2, :, :] -= mean[2]
    x[0, :, :] /= std[0]
    x[1, :, :] /= std[1]
    x[2, :, :] /= std[2]
    return x

def decode_text(tokens, vocab, vocab_inp):
    """
    decode trocr
    """
    s_start = vocab.get('<s>')
    s_end = vocab.get('</s>')
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')
    text = ''
    for tk in tokens:
        if tk == s_end:
            break
        if tk not in [s_end, s_start, pad, unk]:
            text += vocab_inp[tk]
    return text

class OnnxEncoder(object):
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())
    
    def __call__(self, image):
        onnx_inputs = {self.model.get_inputs()[0].name: np.asarray(image, dtype='float32')}
        onnx_output = self.model.run(None, onnx_inputs)[0]
        return onnx_output

class OnnxDecoder(object):
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.model.get_inputs())}
    
    def __call__(self, input_ids, encoder_hidden_states, attention_mask):
        input_info = {"input_ids": input_ids,
                      "attention_mask": attention_mask,
                      "encoder_hidden_states": encoder_hidden_states}
        onnx_inputs = {key: input_info[key] for key in self.input_names}
        onnx_output = self.model.run(['logits'], onnx_inputs)
        return onnx_output

class OnnxEncoderDecoder(object):
    def __init__(self, model_path):
        self.encoder = OnnxEncoder(os.path.join(model_path, "encoder_model.onnx"))
        self.decoder = OnnxDecoder(os.path.join(model_path, "decoder_model.onnx"))
        self.vocab = read_vocab(os.path.join(model_path, "vocab.json"))
        self.vocab_inp = {self.vocab[key]: key for key in self.vocab}
        self.threshold = 0.95  # 置信度阈值
        self.max_len = 50  # 最长文本长度

    def run(self, image):
        """
        处理裁剪后的图像
        """
        image = cv2.resize(image, (384, 384))
        pixel_values = cv2.split(np.array(image))
        pixel_values = do_norm(np.array(pixel_values))
        pixel_values = np.array([pixel_values])
        encoder_output = self.encoder(pixel_values)
        ids = [self.vocab["<s>"], ]
        mask = [1, ]
        scores = []
        for i in range(self.max_len):
            input_ids = np.array([ids]).astype('int64')
            attention_mask = np.array([mask]).astype('int64')
            decoder_output = self.decoder(input_ids=input_ids,
                                          encoder_hidden_states=encoder_output,
                                          attention_mask=attention_mask)
            pred = decoder_output[0][0]
            pred = softmax(pred, axis=1)
            max_index = pred.argmax(axis=1)
            if max_index[-1] == self.vocab["</s>"]:
                break
            scores.append(pred[max_index.shape[0] - 1, max_index[-1]])
            ids.append(max_index[-1])
            mask.append(1)
        avg_score = statistics.mean(scores) if scores else 0
        # print("解码单字评分：{}".format(scores))
        # print("解码平均评分：{}".format(avg_score))
        text = ""
        if avg_score >= self.threshold:
            text = decode_text(ids, self.vocab, self.vocab_inp)
        return text, avg_score

class StampRecognizer:
    def __init__(self, model_path):
        self.encoder_decoder = OnnxEncoderDecoder(model_path)
    
    def run(self, image):
        # 裁剪右下角的384x384区域
        height, width = image.shape[:2]
        if height >= 128 and width >= 128:
            cropped_img = image[height-128:height, width-128:width]
        else:
            # 如果图像小于384x384，则调整为适合的大小
            cropped_img = cv2.resize(image, (384, 384))
        
        # 识别裁剪后的区域
        text, avg_score = self.encoder_decoder.run(cropped_img)
        
        # 检查识别结果
        if avg_score >= 0.95 and text == "北京理工大学团队":
            # 进行下一步操作
            # print("检测到印章，进行下一步操作")
            return True
        
        # print("未检测到印章或识别文字不符合要求")
        return False

if __name__ == '__main__':
    pdf_path = '/workspace/wenbenjiance/测试文档.pdf'
    # outpath为一个列表，要遍历列表中的所有地址，判断是否进行下一步操作
    out_path = pdf2img(pdf_path)

    parser = argparse.ArgumentParser(description='ONNX model test')
    parser.add_argument('--model', type=str, help="ONNX 模型地址", default='/workspace/wenbenjiance/seal')
    # parser.add_argument('--test_img', type=str, help="测试图像")
    
    # args = parser.parse_args()
    # recognizer = StampRecognizer(args.model)
    # img = cv2.imread(args.test_img)
    # img = img[..., ::-1]  # BGR to RGB
    # result = recognizer.run(img)
    # print("检测结果:", result)
    args = parser.parse_args()
    recognizer = StampRecognizer(args.model)

    # 遍历所有图片路径并进行处理
    for img_path in out_path:
        img = cv2.imread(img_path)
        img = img[..., ::-1]  # BGR to RGB
        result = recognizer.run(img)
        # print(f"检测结果 ({img_path}):", result)
        
        # 如果检测结果为 False，立即停止循环
        if not result:
            break
    if result:
        print(f"检测到印章，进行下一步操作")
        
        # 调用文本检测函数，开始进行文字检测
        file_path = text_detect(out_path)
        # 调用关键字检测函数
        result_code = detect_keywords_in_file(file_path)
        # 输出检测结果
        print(f"最终编码为: {result_code}")
        # 使用后删除文本文件
        os.remove(file_path)
    else:
        print("未检测到印章或识别文字不符合要求")