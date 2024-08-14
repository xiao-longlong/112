import json
import os

# 定义JSON文件所在文件夹路径
json_folder = "/media/wxl/Elements/BIT/实验室/比赛/solo_10/sequence.0"  # 替换为你的JSON文件夹路径
output_folder = "/home/wxl/wxlcode/112/data/tym_output/exp3/annotations/view2"  # 输出文件路径

# 遍历文件夹中的所有JSON文件
for idx,filename in enumerate(os.listdir(json_folder)):
    if filename.endswith(".json"):
        # 构建JSON文件的完整路径
        file_path = os.path.join(json_folder, filename)
        # 读取并解析JSON文件
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        # 提取 "captures" 中的图片文件名
        captures = data.get("captures", [])
        annotations = captures[0]["annotations"][0]["values"]
        yolo_annotations = []
        for annotation in annotations:
            number_id = annotation["instanceId"]
            class_id = annotation["labelId"]
            cx = (annotation["origin"][0] + annotation["dimension"][0] / 2) / 1920
            cy = (annotation["origin"][1] + annotation["dimension"][1] / 2) / 1080
            w = annotation["dimension"][0] / 1920
            h = annotation["dimension"][1] / 1080
            yolo_annotations.append([class_id,cx,cy,w,h,number_id])
        output_file = output_folder + f"/{int((idx-1)/2)}.txt"
        with open(output_file, "w") as file:
            for item in yolo_annotations:
                line = ','.join(map(str, item))
            # 将每个列表转换为字符串并写入文件
                file.write(line + "\n")
print("标注文件转换成功。")