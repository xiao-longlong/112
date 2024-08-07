import numpy as np
import cv2
import sys
from matchers import matchers
import time

import cv2
import numpy as np

def draw_yolo_annotations_on_image(image, annotations, output_path):

    # 获取图像的宽度和高度
    height, width, _ = image.shape

    # 绘制标注框
    for ann in annotations:
        class_id, center_x, center_y, bbox_width, bbox_height = ann
        # 计算实际的边界框坐标
        x1 = int((center_x - bbox_width / 2) * width)
        y1 = int((center_y - bbox_height / 2) * height)
        x2 = int((center_x + bbox_width / 2) * width)
        y2 = int((center_y + bbox_height / 2) * height)

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 可以选择在框上标注类别 ID
        cv2.putText(image, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # 保存绘制后的图像
    cv2.imwrite(output_path, image)
    # print(f"Annotated image saved to {output_path}")

def convert_string_list_to_numeric(nested_list):
    numeric_list = []
    for sublist in nested_list:
        numeric_sublist = []
        for item in sublist:
            # 将字符串转换为数字
            numeric_item = list(map(float, item[0].split(',')))
            numeric_sublist.append(numeric_item)
        numeric_list.append(numeric_sublist)
    return numeric_list

def add_black_border(image, top_ratio, bottom_ratio, left_ratio, right_ratio):
    # 获取原始图像的高度和宽度
    height, width = image.shape[:2]
    # 计算要添加的边框的像素数
    top_border = int(height * top_ratio)
    bottom_border = int(height * bottom_ratio)
    left_border = int(width * left_ratio)
    right_border = int(width * right_ratio)
    # 添加黑边
    bordered_image = cv2.copyMakeBorder(
        image, 
        top_border, bottom_border, left_border, right_border, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]  # 黑色边框
    )
    return bordered_image


def remove_black_border_and_get_ratios(image):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 找到非零元素的行和列
    rows = np.any(gray != 0, axis=1)
    cols = np.any(gray != 0, axis=0)
    # 找到非零行和列的起始和结束位置
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    # 裁剪图像
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]
    # 计算各方向去除的比例
    height, width = image.shape[:2]
    top_ratio = y_min / height
    bottom_ratio = (height - 1 - y_max) / height
    left_ratio = x_min / width
    right_ratio = (width - 1 - x_max) / width
    return cropped_image, top_ratio, bottom_ratio, left_ratio, right_ratio

def change_extension_to_txt(file_list):
    return [file_path.replace('.jpg', '.txt') for file_path in file_list]

def read_yolo_annotations(txt_file_list):
    annotations = []
    for txt_file in txt_file_list:
        with open(txt_file, 'r') as file:
            file_annotations = [line.strip().split() for line in file]
            annotations.append(file_annotations)
    return annotations

class Stitch:
	def __init__(self, args):
		self.path = args
		fp = open(self.path, 'r')
		filenames = [each.rstrip('\r\n') for each in  fp.readlines()]
		# print(filenames)
		self.images = [cv2.resize(cv2.imread(each),(480, 320)) for each in filenames]
		filenames = change_extension_to_txt(filenames)
		self.annotations = read_yolo_annotations(filenames)
		# print(self.annotations)
		self.annotations = convert_string_list_to_numeric(self.annotations)
		# print(self.annotations)
		self.count = len(self.images)
		self.left_list, self.right_list, self.center_im = [], [],None
		self.left_annos, self.right_annos = [], []
		self.leftanno, self.rightanno = None, None
		self.matcher_obj = matchers()
		self.prepare_lists()
		self.flag_left = True

	def prepare_lists(self):
		# print("Number of images : %d"%self.count)
		self.centerIdx = self.count/2 -1
		# print("Center index image : %d"%self.centerIdx)
		self.center_im = self.images[int(self.centerIdx)]
		for i in range(self.count):
			if(i<self.centerIdx):
				self.left_list.append(self.images[i])
				self.left_annos.append(self.annotations[i])
			else:
				self.right_list.append(self.images[i])
				self.right_annos.append(self.annotations[i])
		# print("Image lists prepared")

	def leftshift(self):
		# self.left_list = reversed(self.left_list)
		a = self.left_list[0]
		a_annos = self.left_annos[0]
		tmp = a
		tmp_annos = a_annos
		for idx,b in enumerate(self.left_list[1:]):
			b_annos = self.left_annos[1+idx]
			H = self.matcher_obj.match(a, b, 'left')
			# print("Homography is : ", H)
			xh = np.linalg.inv(H)
			# print("Inverse Homography :", xh)
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			ds = ds/ds[-1]
			# print("final ds=>", ds)
			f1 = np.dot(xh, np.array([0,0,1]))
			f1 = f1/f1[-1]
			xh[0][-1] += abs(f1[0])
			xh[1][-1] += abs(f1[1])
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			offsety = abs(int(f1[1]))
			offsetx = abs(int(f1[0]))
			dsize = (int(ds[0])+offsetx + 1000, int(ds[1]) + offsety)
			# print("image dsize =>", dsize)
			tmp = cv2.warpPerspective(a, xh, dsize)
			tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b

			xywh_annos = [[x[0], x[1]*b.shape[1], x[2]*b.shape[0], x[3], x[4]] for x in a_annos]
			tmp_annos  = [[x[0], int((x[1]*xh[0][0]+x[2]*xh[0][1]+xh[0][2])/(x[1]*xh[2][0]+x[2]*xh[2][1]+xh[2][2])), 
				  				int((x[1]*xh[1][0]+x[2]*xh[1][1]+xh[1][2])/(x[1]*xh[2][0]+x[2]*xh[2][1]+xh[2][2])), x[3], x[4]] for x in xywh_annos]
			tmp_annos = [[x[0], x[1]/tmp.shape[1], x[2]/tmp.shape[0], x[3], x[4]] for x in tmp_annos]
			
			xywh_annos = [[x[0], x[1]*b.shape[1]+offsetx, x[2]*b.shape[0]+offsety, x[3], x[4]] for x in b_annos]
			b_annos = [[x[0], x[1]/tmp.shape[1], x[2]/tmp.shape[0], x[3], x[4]] for x in xywh_annos]
			
			tmp_annos = tmp_annos + b_annos
			a = tmp
			a_annos = tmp_annos

		self.leftImage = tmp
		self.leftanno = tmp_annos
		print("self.leftanno",self.leftanno)
		self.leftImage, top_ratio, bottom_ratio, left_ratio, right_ratio= remove_black_border_and_get_ratios(self.leftImage)
		self.leftanno = [[x[0], (x[1] - left_ratio)/(1 - left_ratio - right_ratio), (x[2] - top_ratio)/(1 - top_ratio - bottom_ratio), x[3], x[4]] for x in self.leftanno]
		self.leftImage = cv2.resize(self.leftImage,(480, 320))
		# if(self.flag_left):
		# 	# cv2.imwrite("test_left.jpg",self.leftImage)
		# 	self.flag_left = False
		# else:
		# 	# cv2.imwrite("test_final.jpg",self.leftImage)
		self.left_list = []
		self.left_list.append(self.leftImage)
		self.left_annos = []
		self.left_annos.append(self.leftanno)


	def rightshifttest(self):
		a = self.right_list[-1]
		a_annos = self.right_annos[-1]
		tmp = a
		tmp_annos = a_annos
		for idx,b in enumerate(reversed(self.right_list[:-1])):
			b_annos = self.right_annos[len(self.right_annos)-2-idx]
			H = self.matcher_obj.match(a, b, 'left')
			# print("Homography is : ", H)
			xh = np.linalg.inv(H)
			# print("Inverse Homography :", xh)
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			ds = ds/ds[-1]
			# print("final ds=>", ds)
			f1 = np.dot(xh, np.array([0,0,1]))
			f1 = f1/f1[-1]
			xh[0][-1] += abs(f1[0])
			xh[1][-1] += abs(f1[1])
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			ds1 = np.dot(xh,np.array([0,a.shape[0],1]))
			ds1 = ds1/ds1[-1]
			ds2 = np.dot(xh,np.array([a.shape[1],0,1]))
			ds2 = ds2/ds2[-1]
			offsety = abs(int(f1[1]))
			offsetx = abs(int(f1[0]))
			dsize = (int(ds2[0]) + offsetx, max((int(ds1[1]) + offsety),(b.shape[1]+ offsety)))
			# print("image dsize =>", dsize)

			tmp = cv2.warpPerspective(a, xh, dsize)
			tmp[int(xh[1][2]/xh[2][2])-offsety:int(xh[1][2]/xh[2][2])+b.shape[0]-offsety, int(xh[0][2]/xh[2][2])-offsetx:int(xh[0][2]/xh[2][2])+b.shape[1]-offsetx] = b
			xywh_annos = [[x[0], x[1]*b.shape[1], x[2]*b.shape[0], x[3], x[4]] for x in a_annos]
			tmp_annos  = [[x[0], (x[1]*xh[0][0]+x[2]*xh[0][1]+xh[0][2])/(x[1]*xh[2][0]+x[2]*xh[2][1]+xh[2][2]), 
				  				(x[1]*xh[1][0]+x[2]*xh[1][1]+xh[1][2])/(x[1]*xh[2][0]+x[2]*xh[2][1]+xh[2][2]), x[3], x[4]] for x in xywh_annos]
			tmp_annos = [[x[0], x[1]/tmp.shape[1], x[2]/tmp.shape[0], x[3], x[4]] for x in tmp_annos]
			
			xywh_annos = [[x[0], x[1]*b.shape[1]+int(xh[0][2]/xh[2][2])-offsetx, x[2]*b.shape[0]+int(xh[1][2]/xh[2][2])-offsety, x[3], x[4]] for x in b_annos]
			b_annos = [[x[0], x[1]/tmp.shape[1], x[2]/tmp.shape[0], x[3], x[4]] for x in xywh_annos]

			tmp_annos = tmp_annos + b_annos
			a = tmp
			a_annos = tmp_annos

		self.rightImage = tmp
		self.rightanno = tmp_annos
		self.rightImage, top_ratio, bottom_ratio, left_ratio, right_ratio  = remove_black_border_and_get_ratios(self.rightImage)
		self.rightanno = [[x[0], (x[1] - left_ratio)/(1 - left_ratio - right_ratio), (x[2] - top_ratio)/(1 - top_ratio - bottom_ratio), x[3], x[4]] for x in self.rightanno]
		self.rightImage = cv2.resize(self.rightImage,(480, 320))
		self.left_list.append(self.rightImage)
		self.left_annos.append(self.rightanno)
		cv2.imwrite("test_right.jpg",self.rightImage)


if __name__ == '__main__':
	start_time = time.time()
	try:
		args = sys.argv[1]
	except:
		args = "/home/wxl/wxlcode/Python-Multiple-Image-Stitching/code/file1.txt"
	# finally:
	# 	print("Parameters : ", args)
	s = Stitch(args)
	s.leftshift()
	s.rightshifttest()
	s.leftshift()
	print("s.leftanno",s.leftanno)
	draw_yolo_annotations_on_image(s.leftImage, s.leftanno, "visual.jpg")
	# cv2.imwrite("test_time1111.jpg", s.leftImage)

	print(time.time() - start_time)