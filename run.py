import cv2 as cv
import numpy as np
from os import environ
from os.path import join
from sys import argv
from glob import glob
from numpy import zeros
from os.path import basename, join
from skimage.io import imread
from glob import glob
from re import sub
from time import time
from traceback import format_exc
from os import makedirs
from sklearn.svm import LinearSVC
from math import pi
from skimage.transform import resize
from skimage.color import rgb2gray
from scipy.misc import imresize



def find_contours_mask(mask, image, gauss=3, color=(255, 0, 0), find_const=0.0006):
	square = mask.shape[0] * mask.shape[1]
	gray = cv.GaussianBlur(mask, (gauss, gauss), 0)
	edged = cv.Canny(gray, 10, 250)

	contours, hierarchy = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

	contour_list = []
	for cnt in contours:
		perim = cv.arcLength(cnt, True)
		area = cv.contourArea(cnt)
		if area > 0.04 * perim * perim and area < 0.9 * perim * perim and area > square * find_const:
			cv.drawContours(image, cnt, -1, color, 2)
			contour_list.append(cnt)

	return contour_list, image


def find_contours(image, gauss=5, color=(255, 0, 0), find_const=0.0006):
	square = image.shape[0] * image.shape[1]

	gray = cv.convertScaleAbs(image)
	gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
	gray = cv.GaussianBlur(gray, (gauss, gauss), 0)

	edged = cv.Canny(gray, 10, 250)

	contours, hierarchy = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

	contour_list = []
	for cnt in contours:
		perim = cv.arcLength(cnt, True)
		area = cv.contourArea(cnt)
		if area > 0.04 * perim * perim and area < 0.9 * perim * perim and area > square * find_const:
			cv.drawContours(image, cnt, -1, color, 2)
			contour_list.append(cnt)
			

	return contour_list, image



def red_filter(image, low_l=[0,150,100], high_l=[10,255,255], low_r=[145, 100, 50], high_r=[179, 255, 255]):	
	hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	lower_blue = np.array(low_l)
	upper_blue = np.array(high_l)
	lower_blue_minus = np.array(low_r)
	upper_blue_minus = np.array(high_r)
	mask = cv.inRange(hsv, lower_blue, upper_blue)
	mask_minus = cv.inRange(hsv, lower_blue_minus, upper_blue_minus)
	mask += mask_minus
	return mask


def blue_filter(image, low=[90,50,50], high=[145,255,255]):
	hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	lower_blue = np.array(low)
	upper_blue = np.array(high)
	mask = cv.inRange(hsv, lower_blue, upper_blue)
	return mask


def white_filter(image, low=[0,0,130], high=[179,40,255]):
	hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	lower_blue = np.array(low)
	upper_blue = np.array(high)
	mask = cv.inRange(hsv, lower_blue, upper_blue)
	return mask



def contour_img(cnt, image):
	minx, maxx, miny, maxy = image.shape[0], 0, image.shape[1], 0
	for j in cnt:
		i = j[0]
		if i[1] < minx:
			minx = i[1]
		if i[0] < miny:
			miny = i[0]
		if i[1] > maxx:
			maxx = i[1]
		if i[0] > maxy:
			maxy = i[0]
	if ((maxx - minx) * (maxy - miny) < image.shape[0] * image.shape[1] * 0.0005):
		minx = max(0, int(minx - (maxx - minx)))
		maxx = min(image.shape[0], int(maxx + (maxx - minx)))
		miny = max(0, int(miny - (maxy - miny)))
		maxy = min(image.shape[1], int(maxy + (maxy - miny)))
	else:
		minx = max(0, int(minx - (maxx - minx) / 3))
		maxx = min(image.shape[0], int(maxx + (maxx - minx) / 3))
		miny = max(0, int(miny - (maxy - miny) / 3))
		maxy = min(image.shape[1], int(maxy + (maxy - miny) / 3))
	res = image[minx:maxx, miny:maxy, :]
	print(minx, maxx, miny, maxy)
	return res, ((minx, maxx, miny, maxy))


def remove_same(arr, percent=0.95):
	change = True
	while (change):
		if (len(arr) <= 1):
			return arr
		change = False
		for i in range(1, len(arr)):
			for j in range(i):
				xi1, xi2, yi1, yi2 = arr[i][1]
				xj1, xj2, yj1, yj2 = arr[j][1]
				if (xj2 <= xi1 or xi2 <= xj1 or yi2 <= yj1 or yj2 <= yi1):
					continue
				sq1 = (xi2 - xi1) * (yi2 - yi1)
				sq2 = (xj2 - xj1) * (yj2 - yj1)
				x1 = max(xi1, xj1)
				y1 = max(yi1, yj1)
				x2 = min(xi2, xj2)
				y2 = min(yi2, yj2)
				sq = (x2 - x1) * (y2 - y1)
				if sq > sq1 * percent:
					arr.pop(i)
					change = True
					break
				elif sq > sq2 * percent:
					arr.pop(j)
					change = True
					break
			if change:
				break
	return arr



def find_all_contours(image):
	mask_red = red_filter(image)
	mask_blue = blue_filter(image)
	mask_white = white_filter(image)
	image_print = image.copy()
	c_contours, image_print = find_contours(image_print)
	r_contours, image_print = find_contours_mask(mask_red, image_print)
	b_contours, image_print = find_contours_mask(mask_blue, image_print)
	w_contours, image_print = find_contours_mask(mask_white, image_print)
	contours = r_contours + b_contours + w_contours + c_contours
	images = []
	for elem in contours:
		images.append(contour_img(elem, image))
	print(len(images))

	images = remove_same(images)
	print(len(images))

	return images, image_print


def find_one_cnt(image, has_cnt=False, retr=cv.RETR_TREE):
	center = (image.shape[1] // 2, image.shape[0] // 2)
	contours, hierarchy = cv.findContours(image.copy(), retr, cv.CHAIN_APPROX_NONE)
	if len(contours) <= 1 and has_cnt:
		return []

	contour_list = []
	cur_area = 0
	cur_cnt = []
	for cnt in contours:
		perim = cv.arcLength(cnt, True)
		area = cv.contourArea(cnt)
		if cv.pointPolygonTest(cnt, center, True) > 0 and area > cur_area:
			cur_area = area
			cur_cnt = cnt
	return cur_cnt


def make_frame(image, color):
	for i in range(image.shape[0]):
		image[i, 0] = color
		image[i, image.shape[1] - 1] = color
	for j in range(image.shape[1]):
		image[0, j] = color
		image[image.shape[0] - 1, j] = color
	return image


def to_find_cnts(image, gauss, frame=False, color=(0, 0, 0)):
	if (frame):
		image_fr = make_frame(image, color)
		gray = cv.GaussianBlur(image_fr, (gauss, gauss), 0)
	else:
		gray = cv.GaussianBlur(image, (gauss, gauss), 0)
	gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
	edged = cv.Canny(gray, 10, 250)
	return edged


def gauss_const(image):
	if (image.shape[0] < 50 or image.shape[1] < 50):
		return 5
	return 7


def clear_image(image, gauss=7, colors=[(255, 0, 0), (0, 0, 255), (0, 255, 0)], beta=-30):
	if gauss == 7:
		gauss = gauss_const(image)

	image = cv.convertScaleAbs(image, alpha=1.5, beta=beta)

	cnrs = to_find_cnts(image, gauss)
	cur_cnt = find_one_cnt(cnrs)
	
	if (len(cur_cnt) == 0):
		kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
		closed = cv.morphologyEx(cnrs, cv.MORPH_CLOSE, kernel)
		cur_cnt = find_one_cnt(closed)

		if (len(cur_cnt) == 0):
			cnrs = to_find_cnts(image, gauss, True, colors[0])
			cur_cnt = find_one_cnt(cnrs, has_cnt=True, retr=cv.RETR_EXTERNAL)
			if (len(cur_cnt) == 0):
				cnrs = to_find_cnts(image, gauss, True, colors[1])
				cur_cnt = find_one_cnt(cnrs, has_cnt=True, retr=cv.RETR_EXTERNAL)
				if (len(cur_cnt) == 0):
					cnrs = to_find_cnts(image, gauss, True, colors[2])
					cur_cnt = find_one_cnt(cnrs, has_cnt=True, retr=cv.RETR_EXTERNAL)
					if (len(cur_cnt) == 0):
						return False, []

	mask = np.zeros(image.shape[:2], np.uint8)
	white = np.full(image.shape, 255, dtype=np.uint8)

	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if cv.pointPolygonTest(cur_cnt, (j,i), True) >= 0:
				mask[i, j] = 255
				
	cv.drawContours(mask, cur_cnt, -1, 255, -1)
	res = cv.bitwise_and(image, image, mask=mask)
	return True, res


def cut_signs(image, i=0):
	images, image_print = find_all_contours(image)
	signs = []
	for elem in images:
		res, res_img = clear_image(elem)
		if (res):
			signs.append(res_img)
	return signs, image_print



def extract_features(path, files):
	hog_length = len(extract_hog(files[0]))
	data = zeros((len(files), hog_length))
	for i in range(0, len(files)):
		data[i, :] = extract_hog(files[i])
	return data


def Histogram(G, angle, cell_num, cell_size, binCount):
	histogram = np.zeros((cell_num, cell_num, binCount))
	cur_bin = np.zeros(binCount)
	for i in range(cell_num):
		for j in range(cell_num):

			cur_bin = np.zeros(binCount)
			for k in range(cell_size):
				for l in range(cell_size):
					cur_angle = angle[i * cell_size + k][j * cell_size + l]
					sector = int(cur_angle / (pi / binCount)) % binCount
					weight = (cur_angle - sector * (pi / binCount)) / (pi / binCount)
					
					cur_bin[sector] += weight * G[i * cell_size + k][j * cell_size + l]
					if (sector + 1 == binCount):
						cur_bin[0] += (1 - weight) * G[i * cell_size + k][j * cell_size + l] 
					else:
						cur_bin[sector + 1] += (1 - weight) * G[i * cell_size + k][j * cell_size + l]

			histogram[i][j] = cur_bin
	return histogram


def Block(histogram, block_size, block_num, binCount, eps):
	vectors = np.zeros((block_num, block_num, block_size, block_size, binCount))

	for i in range(block_num):
		for j in range(block_num):
			vector = histogram[i : i + 2, j : j + 2, :]
			vectors[i, j, :] = vector / np.sqrt(np.sum(np.square(vector)) + eps ** 2)
	return np.ravel(vectors)


def extract_hog(image):
	size = (64, 64)
#	image = resize(rgb2gray(image), size, anti_aliasing=False)
	image = imresize(rgb2gray(image), size)
	gradient_x, gradient_y = np.gradient(image, 0.5)
	g_module = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
	angle = np.absolute(np.arctan2(gradient_y, gradient_x))

	cell_num, binCount, eps = 8, 8, 0.0000000000001
	cell_size = size[0] // cell_num
	block_cells_num, block_cells_size = cell_num - 1, 2
	histogram = Histogram(g_module, angle, cell_num, cell_size, binCount)
	return Block(histogram, block_cells_size, block_cells_num, binCount, eps)


def fit_and_classify(train_features, train_labels, test_features):
	clf = LinearSVC()
	clf.fit(train_features, train_labels)
	return clf.predict(test_features)



def train_features(n, file):
	features = [] * n
	for i in range(1, n+1):
		features[i-1] = file[i].split(' ')
	return features


def train_labels(n, file):
	return file[n+1].split(' ')


def find_signs(name, image_format, par):
	image = cv.imread(name+image_format)
	if not (image is not None):
		return [], [], -1

	signs, image_print = cut_signs(image)
	num = len(signs)

	f = open('train.txt', 'r')
	i = f[0].split(' ')
	y = fit_and_classify(train_features(i, f), train_labels(i, f), signs)

	f.close()
	return image_print, y, num



def input_image():
	print("- Input image")
	name = input()
	new_name = join(name, '*.png')
	format_name = ".png"
	if name == new_name:
		format_name = ".jpg"
		new_name = join(name, '*.jpg')
	if new_name == name:
		return "", ""
	return new_name, format_name


def input_pars():
	print("- You can input parameters. Use one of this commands:")
	str0 = "nothing"
	str1 = "only names to csv"
	str2 = "only image with found signs"
	str3 = "return image with found signs and names to csv"
	print("-- "+str0)
	print("-- "+str1)
	print("-- "+str2)
	print("-- "+str3)
	str = input()
	if str == str0:
		return 0
	if str == str1:
		return 1
	elif str == str2:
		return 2
	elif str == str3:
		return 3
	print("----- Please try again")
	return -1


def print_names(arr, file=0):
	names = signs_names()
	if (file == 0):
		for elem in arr:
			print(names[elem])
		return

	with open(join(output_dir, 'output.csv'), 'w') as fout:
		for elem in arr:
			print('%s' % (names[elem]), file=fout)


def start():
	print("- Here you can input image and choose enought parameters to fing signs in image")
	print("- You can use one of this commands:")
	print("-- input")
	print("-- info")
	print("-- exit")

	str = input()
	if (str.find("input") >= 0):
		image_name, image_format = input_image()
		if image_name != "":
			return 1, image_name, image_format
		print("----- Wrong image or format, try again")
		return 0
	elif (str.find("info") >= 0):
		print("- The program was made by Alexandra Latysheva, group CS AMI 172")
		return 0
	elif (str.find("exit") >= 0):
		return -1;
	print("----- Please try again")
	return 0


def input_program():
	res = 1
	while res != -1:
		res = start()
		if res != 0 and res != -1:
			par = -1
			while par == -1:
				par = input_pars()
			res_img, res_list, num = find_signs(res[1], res[2], par)
			if num == -1:
				print("----- Image can't be open")
				continue
			print("- found", num, "signs")
			if par > 1:
				cv.imwrite(res[1]+"_result.png", res_img)
				print("- File "+res[1]+"_result.png"+" was added in dir")
			if par % 2 == 1:
				print_names(res_list, 1)
				print("- File output.csv was added in dir")
			print_names(res_list)


input_program()			
