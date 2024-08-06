import torch
if torch.cuda.is_available():
    print("GPU可用")
else:
    print("GPU不可用")
import torch

print("当前可用的GPU数量: ", torch.cuda.device_count())
import torch  
print(torch.__version__)

# #####################
# import os
# import random
# from tqdm import tqdm
# # 指定 images 文件夹路径，images文件夹里面是图片，这里的路径换为自己的，一个是图片路径
# image_dir = "/workspace/output/coco128/images/train2017"
# # 指定 labels 文件夹路径，一个是标注文件路径

# label_dir = "/workspace/output/coco128/labels/train2017"

# # 创建一个空列表来存储有效图片的路径

# valid_images = []

# # 创建一个空列表来存储有效 label 的路径

# valid_labels = []

# # 遍历 images 文件夹下的所有图片

# for image_name in os.listdir(image_dir):

#     # 获取图片的完整路径

#     image_path = os.path.join(image_dir, image_name)

#     # 获取图片文件的扩展名

#     # ext = os.path.splitext(image_name)[-1]

#     # 根据扩展名替换成对应的 label 文件名
#     label_name=image_name.replace(".jpg",".txt")
#     # label_name = image_name.replace(ext, ".txt")

#     # 获取对应 label 的完整路径

#     label_path = os.path.join(label_dir, label_name)

#     # 判断 label 是否存在

#     if not os.path.exists(label_path):

#         # 删除图片
        
#         os.remove(image_path)

#         print("deleted:", image_path)

#     else :
    
#         # 将图片路径添加到列表中

#         valid_images.append(image_path)

#         # 将label路径添加到列表中

#         valid_labels.append(label_path)

#         # print("valid:", image_path, label_path)

# # 遍历每个有效图片路径

# for i in tqdm(range(len(valid_images))):

#     image_path = valid_images[i]

#     label_path = valid_labels[i]

#     # 随机生成一个概率

#     r = random.random()

#     # 判断图片应该移动到哪个文件夹

#     # train：valid：test = 7:3:1

#     if r < 0.1:

#         # 移动到 test 文件夹，这里的三个路径换为自己的

#         destination = "/workspace/output/coco128/test2017"

#     elif r < 0.2:

#         # 移动到 valid 文件夹

#         destination = "/workspace/output/coco128/val2017"

#     else:

#         # 移动到 train 文件夹

#         destination = "/workspace/output/coco128/train2017"

#     # 生成目标文件夹中图片的新路径

#     image_destination_path = os.path.join(destination, "images", os.path.basename(image_path))

#     # 移动图片到目标文件夹

#     os.rename(image_path, image_destination_path)

#     # 生成目标文件夹中 label 的新路径

#     label_destination_path = os.path.join(destination, "labels", os.path.basename(label_path))

#     # 移动 label 到目标文件夹

#     os.rename(label_path, label_destination_path)

# print("valid images:", valid_images)

# #输出有效label路径列表

# print("valid labels:", valid_labels)





# ############################
# import os
# import json
# import cv2
# import random
# import time
# from PIL import Image

# coco_format_save_path='/workspace/output/coco128/annotations/'   #要生成的标准coco格式标签所在文件夹
# yolo_format_classes_path='/workspace/output/coco128/train2017/classes.txt'     #类别文件，一行一个类
# yolo_format_annotation_path='/workspace/output/coco128//train/labels/'  #yolo格式标签所在文件夹
# img_pathDir='/workspace/output/coco128/train/images/'    #图片所在文件夹

# with open(yolo_format_classes_path,'r') as fr:                               #打开并读取类别文件
#     lines1=fr.readlines()
# # print(lines1)
# categories=[]                                                                 #存储类别的列表
# for j,label in enumerate(lines1):
#     label=label.strip()
#     categories.append({'id':j+1,'name':label,'supercategory':'None'})         #将类别信息添加到categories中
# # print(categories)

# write_json_context=dict()                                                      #写入.json文件的大字典
# write_json_context['info']= {'description': '', 'url': '', 'version': '', 'year': 2024, 'contributor': '纯粹ss', 'date_created': '2024-01-12'}
# write_json_context['licenses']=[{'id':1,'name':None,'url':None}]
# write_json_context['categories']=categories
# write_json_context['images']=[]
# write_json_context['annotations']=[]

# #接下来的代码主要添加'images'和'annotations'的key值
# imageFileList=os.listdir(img_pathDir)                                           #遍历该文件夹下的所有文件，并将所有文件名添加到列表中
# for i,imageFile in enumerate(imageFileList):
#     imagePath = os.path.join(img_pathDir,imageFile)                             #获取图片的绝对路径
#     image = Image.open(imagePath)                                               #读取图片，然后获取图片的宽和高
#     W, H = image.size

#     img_context={}                                                              #使用一个字典存储该图片信息
#     #img_name=os.path.basename(imagePath)                                       #返回path最后的文件名。如果path以/或\结尾，那么就会返回空值
#     img_context['file_name']=imageFile
#     img_context['height']=H
#     img_context['width']=W
#     img_context['date_captured']='2024.1.12'
#     img_context['id']=i                                                         #该图片的id
#     img_context['license']=1
#     img_context['color_url']=''
#     img_context['flickr_url']=''
#     write_json_context['images'].append(img_context)                            #将该图片信息添加到'image'列表中


#     txtFile=imageFile[:12]+'.txt'                                               #获取该图片获取的txt文件
#     with open(os.path.join(yolo_format_annotation_path,txtFile),'r') as fr:
#         lines=fr.readlines()                                                   #读取txt文件的每一行数据，lines2是一个列表，包含了一个图片的所有标注信息
#     for j,line in enumerate(lines):

#         bbox_dict = {}                                                          #将每一个bounding box信息存储在该字典中
#         # line = line.strip().split()
#         # print(line.strip().split(' '))

#         class_id,x,y,w,h=line.strip().split(' ')                                          #获取每一个标注框的详细信息
#         class_id,x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)       #将字符串类型转为可计算的int和float类型

#         xmin=(x-w/2)*W                                                                    #坐标转换
#         ymin=(y-h/2)*H
#         xmax=(x+w/2)*W
#         ymax=(y+h/2)*H
#         w=w*W
#         h=h*H

#         bbox_dict['id']=i*10000+j                                                         #bounding box的坐标信息
#         bbox_dict['image_id']=i
#         bbox_dict['category_id']=class_id+1                                               #注意目标类别要加一
#         bbox_dict['iscrowd']=0
#         height,width=abs(ymax-ymin),abs(xmax-xmin)
#         bbox_dict['area']=height*width
#         bbox_dict['bbox']=[xmin,ymin,w,h]
#         bbox_dict['segmentation']=[[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]]
#         write_json_context['annotations'].append(bbox_dict)                               #将每一个由字典存储的bounding box信息添加到'annotations'列表中

# name = os.path.join(coco_format_save_path,"train"+ '.json')
# with open(name,'w') as fw:                                                                #将字典信息写入.json文件中
#     json.dump(write_json_context,fw,indent=2)


