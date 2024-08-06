import numpy as np
import cv2
import sys
from matchers import matchers
import time

def remove_black_border(image):
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
	
	return cropped_image

class Stitch:
	def __init__(self, args):
		self.path = args
		fp = open(self.path, 'r')
		filenames = [each.rstrip('\r\n') for each in  fp.readlines()]
		print(filenames)
		self.images = [cv2.resize(cv2.imread(each),(480, 320)) for each in filenames]
		self.count = len(self.images)
		self.left_list, self.right_list, self.center_im = [], [],None
		self.matcher_obj = matchers()
		self.prepare_lists()
		self.flag_left = True

	def prepare_lists(self):
		print("Number of images : %d"%self.count)
		self.centerIdx = self.count/2 -1
		print("Center index image : %d"%self.centerIdx)
		self.center_im = self.images[int(self.centerIdx)]
		for i in range(self.count):
			if(i<self.centerIdx):
				self.left_list.append(self.images[i])
			else:
				self.right_list.append(self.images[i])
		print("Image lists prepared")

	def leftshift(self):
		# self.left_list = reversed(self.left_list)
		a = self.left_list[0]
		tmp = a
		for b in self.left_list[1:]:
			H = self.matcher_obj.match(a, b, 'left')
			print("Homography is : ", H)
			xh = np.linalg.inv(H)
			print("Inverse Homography :", xh)
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			ds = ds/ds[-1]
			print("final ds=>", ds)
			f1 = np.dot(xh, np.array([0,0,1]))
			f1 = f1/f1[-1]
			xh[0][-1] += abs(f1[0])
			xh[1][-1] += abs(f1[1])
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			offsety = abs(int(f1[1]))
			offsetx = abs(int(f1[0]))
			dsize = (int(ds[0])+offsetx + 1000, int(ds[1]) + offsety)
			print("image dsize =>", dsize)
			tmp = cv2.warpPerspective(a, xh, dsize)
			# cv2.imshow("warped", tmp)
			# cv2.waitKey(0)
			tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
			a = tmp

		self.leftImage = tmp
		self.leftImage = remove_black_border(self.leftImage)
		self.leftImage = cv2.resize(self.leftImage,(480, 320))
		if(self.flag_left):
			cv2.imwrite("test_left.jpg",self.leftImage)
			self.flag_left = False
		else:
			cv2.imwrite("test_final.jpg",self.leftImage)
		self.left_list = []
		self.left_list.append(self.leftImage)


	def rightshifttest(self):
		a = self.right_list[-1]
		tmp = a
		for b in reversed(self.right_list[:-1]):
			# xh = self.matcher_obj.match(a, b, 'left')
			# print("Homography is : ", xh)

			H = self.matcher_obj.match(a, b, 'left')
			print("Homography is : ", H)
			xh = np.linalg.inv(H)
			print("Inverse Homography :", xh)
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			ds = ds/ds[-1]
			print("final ds=>", ds)
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
			print("image dsize =>", dsize)
			# dsize = (1500,500)
			# 对图像 a 进行仿射变换
			tmp = cv2.warpPerspective(a, xh, dsize)
			# cv2.imshow("tmp",tmp)
			# cv2.waitKey(0)
			# cv2.imwrite("tmp.jpg", tmp)
			
			# 将图像 b 放到变换后的 tmp 的右侧
			tmp[int(xh[1][2]/xh[2][2])-offsety:int(xh[1][2]/xh[2][2])+b.shape[0]-offsety, int(xh[0][2]/xh[2][2])-offsetx:int(xh[0][2]/xh[2][2])+b.shape[1]-offsetx] = b

			a = tmp

		self.rightImage = tmp
		self.rightImage = remove_black_border(self.rightImage)
		self.rightImage = cv2.resize(self.rightImage,(480, 320))
		self.left_list.append(self.rightImage)
		cv2.imwrite("test_right.jpg",self.rightImage)


		
	# def rightshift(self):
	# 	for each in self.right_list:
	# 		H = self.matcher_obj.match(self.leftImage, each, 'right')
	# 		print("Homography :", H)
	# 		txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
	# 		txyz = txyz/txyz[-1]
	# 		dsize = (int(txyz[0])+self.leftImage.shape[1], int(txyz[1])+self.leftImage.shape[0])
	# 		tmp = cv2.warpPerspective(each, H, dsize)
	# 		# cv2.imshow("tp", tmp)
	# 		# cv2.waitKey()
	# 		if(tmp.shape[0] < self.leftImage.shape[0]):
	# 			tmp1 = np.zeros((self.leftImage.shape[0], tmp.shape[1], 3), dtype=np.uint8)
	# 			tmp1[:tmp.shape[0],:]=tmp
	# 			tmp = tmp1
	# 		tmp[:self.leftImage.shape[0], :self.leftImage.shape[1]]=self.leftImage
	# 		# tmp = self.mix_and_match(self.leftImage, tmp)
	# 		print("tmp shape",tmp.shape)
	# 		print("self.leftimage shape=", self.leftImage.shape)
	# 		self.leftImage = tmp
	# 	# self.showImage('left')



	# def mix_and_match(self, leftImage, warpedImage):
	# 	i1y, i1x = leftImage.shape[:2]
	# 	i2y, i2x = warpedImage.shape[:2]
	# 	print(leftImage[-1,-1])

	# 	t = time.time()
	# 	black_l = np.where(leftImage == np.array([0,0,0]))
	# 	black_wi = np.where(warpedImage == np.array([0,0,0]))
	# 	print(time.time() - t)
	# 	print(black_l[-1])

	# 	for i in range(0, i1x):
	# 		for j in range(0, i1y):
	# 			try:
	# 				if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
	# 					# print "BLACK"
	# 					# instead of just putting it with black, 
	# 					# take average of all nearby values and avg it.
	# 					warpedImage[j,i] = [0, 0, 0]
	# 				else:
	# 					if(np.array_equal(warpedImage[j,i],[0,0,0])):
	# 						# print "PIXEL"
	# 						warpedImage[j,i] = leftImage[j,i]
	# 					else:
	# 						if not np.array_equal(leftImage[j,i], [0,0,0]):
	# 							bw, gw, rw = warpedImage[j,i]
	# 							bl,gl,rl = leftImage[j,i]
	# 							# b = (bl+bw)/2
	# 							# g = (gl+gw)/2
	# 							# r = (rl+rw)/2
	# 							warpedImage[j, i] = [bl,gl,rl]
	# 			except:
	# 				pass
	# 	# cv2.imshow("waRPED mix", warpedImage)
	# 	# cv2.waitKey()
	# 	return warpedImage




	# def trim_left(self):
	# 	pass

	# def showImage(self, string=None):
	# 	if string == 'left':
	# 		cv2.imshow("left image", self.leftImage)
	# 		# cv2.imshow("left image", cv2.resize(self.leftImage, (400,400)))
	# 	elif string == "right":
	# 		cv2.imshow("right Image", self.rightImage)
	# 	cv2.waitKey()


if __name__ == '__main__':
	start_time = time.time()
	try:
		args = sys.argv[1]
	except:
		args = "/home/wxl/wxlcode/Python-Multiple-Image-Stitching/code/file1.txt"
	finally:
		print("Parameters : ", args)
	s = Stitch(args)
	s.leftshift()
	s.rightshifttest()
	s.leftshift()
	print("done")
	cv2.imwrite("test_time1111.jpg", s.leftImage)
	print("image written")
	cv2.destroyAllWindows()
	print(time.time() - start_time)