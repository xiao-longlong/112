# coding: utf-8
import tr
import sys, cv2, time, os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

_BASEDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASEDIR)


def test():
    img_path = "/workspace/wenbenjiance/屏幕截图 2024-08-08 175654.png"
    # img_path = "imgs/name_card.jpg"

    img_pil = Image.open(img_path)
    try:
        if hasattr(img_pil, '_getexif'):
            # from PIL import ExifTags
            # for orientation in ExifTags.TAGS.keys():
            #     if ExifTags.TAGS[orientation] == 'Orientation':
            #         break
            orientation = 274
            exif = dict(img_pil._getexif().items())
            if exif[orientation] == 3:
                img_pil = img_pil.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img_pil = img_pil.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img_pil = img_pil.rotate(90, expand=True)
    except:
        pass

    MAX_SIZE = 1600
    if img_pil.height > MAX_SIZE or img_pil.width > MAX_SIZE:
        scale = max(img_pil.height / MAX_SIZE, img_pil.width / MAX_SIZE)

        new_width = int(img_pil.width / scale + 0.5)
        new_height = int(img_pil.height / scale + 0.5)
        img_pil = img_pil.resize((new_width, new_height), Image.ANTIALIAS)

    color_pil = img_pil.convert("RGB")
    gray_pil = img_pil.convert("L")

    img_draw = ImageDraw.Draw(color_pil)
    colors = ['red', 'green', 'blue', "purple"]

    t = time.time()
    n = 1
    for _ in range(n):
        tr.detect(gray_pil, flag=tr.FLAG_RECT)
    print("time", (time.time() - t) / n)

    results = tr.run(gray_pil, flag=tr.FLAG_ROTATED_RECT)

    # 指定保存的文件路径和名称
    file_path = "recognized_text.txt"

    # 将文字写入到TXT文件中
    with open(file_path, 'w', encoding='utf-8') as file:
        for i, rect in enumerate(results):
            file.write(rect[1]+'\n')

    print(f"文本已保存到 {file_path}")


    for i, rect in enumerate(results):
        cx, cy, w, h, a = tuple(rect[0])
        print(i, "\t", rect[1], rect[2])
        box = cv2.boxPoints(((cx, cy), (w, h), a))
        box = np.int0(np.round(box))

        for p1, p2 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            img_draw.line(xy=(box[p1][0], box[p1][1], box[p2][0], box[p2][1]), fill=colors[i % len(colors)], width=2)

    color_pil.show()


if __name__ == "__main__":
    test()
