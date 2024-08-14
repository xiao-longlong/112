import cv2
import os

# 定义原始文件夹和目标文件夹路径
original_folder = "/media/wxl/Elements/BIT/实验室/比赛/solo_12/sequence.0/"  # 原始文件夹路径
new_folder = "/home/wxl/wxlcode/112/data/tym_output/exp3/imgs/view3"  # 新文件夹路径

# 如果新文件夹不存在，则创建它
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# 遍历原始文件夹中的图片文件
for filename in os.listdir(original_folder):
    if filename.endswith(".png"):
        # 读取图片
        img = cv2.imread(os.path.join(original_folder, filename))
        
        # 提取数字部分
        step_num = filename.replace('step', '').replace('.camera.png', '')
        
        # 定义新文件名
        new_filename = f"{step_num}.jpg"
        
        # 保存到新文件夹
        cv2.imwrite(os.path.join(new_folder, new_filename), img)

print("图片重命名并保存成功。")
