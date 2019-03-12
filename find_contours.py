import cv2 as cv
import numpy as np

def find_contours_mask(mask, image):
	square = len(mask) * len(mask[0])
	gray = cv.GaussianBlur(mask, (3, 3), 0) #2?
	edged = cv.Canny(gray, 10, 250)
	kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
	closed = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)

	contours, hierarchy = cv.findContours(closed.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

	contour_list = []
	for cnt in contours:
		perim = cv.arcLength(cnt, True)
		area = cv.contourArea(cnt)
		if area > 0.04 * perim * perim and area < 0.9 * perim * perim and area > square * 0.0005:
			cv.drawContours(image, cnt, -1, (255, 0, 0), 2)
			contour_list.append(cnt)

	return contour_list, image

def find_contours(image):
	square = len(image) * len(image[0])


	gray = cv.convertScaleAbs(image)
	gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
	gray = cv.GaussianBlur(gray, (5, 5), 0) #2?

	edged = cv.Canny(gray, 10, 250)

	kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
	closed = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)
	
	contours, hierarchy = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

	contour_list = []
	for cnt in contours:
		perim = cv.arcLength(cnt, True)
		area = cv.contourArea(cnt)
		if area > 0.040 * perim * perim and area < 0.9 * perim * perim and area > square * 0.0005:
			cv.drawContours(image, cnt, -1, (255, 0, 0), 2)
			contour_list.append(cnt)
			

	return contour_list, image

def red_filter(image):
	hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	lower_blue = np.array([0,150,100])
	upper_blue = np.array([10,255,255])
	lower_blue_minus = np.array([145, 100, 50])
	upper_blue_minus = np.array([179, 255, 255])
	mask = cv.inRange(hsv, lower_blue, upper_blue)
	mask_minus = cv.inRange(hsv, lower_blue_minus, upper_blue_minus)
	mask += mask_minus
	cv.imwrite("output_red.jpg", mask)
	return mask


def blue_filter(image):
	hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	lower_blue = np.array([90,50,50])
	upper_blue = np.array([145,255,255])
	mask = cv.inRange(hsv, lower_blue, upper_blue)
	cv.imwrite("output_blue.jpg", mask)
	return mask


def white_filter(image):
	hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	lower_blue = np.array([0,0,130])
	upper_blue = np.array([179,40,255])
	mask = cv.inRange(hsv, lower_blue, upper_blue)
	cv.imwrite("output_write.jpg", mask)
	return mask



def contour_img(cnt, image):
	minx, maxx, miny, maxy = len(image[:, 0, 0]), 0, len(image[0, :, 0]), 0
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
	if ((maxx - minx) * (maxy - miny) < len(image[:, 0, 0]) * len(image[0, :, 0]) * 0.0005):
		minx = max(0, int(minx - (maxx - minx)))
		maxx = min(len(image[:, 0, 0]), int(maxx + (maxx - minx)))
		miny = max(0, int(miny - (maxy - miny)))
		maxy = min(len(image[0, :, 0]), int(maxy + (maxy - miny)))
	else:
		minx = max(0, int(minx - (maxx - minx) / 3))
		maxx = min(len(image[:, 0, 0]), int(maxx + (maxx - minx) / 3))
		miny = max(0, int(miny - (maxy - miny) / 3))
		maxy = min(len(image[0, :, 0]), int(maxy + (maxy - miny) / 3))
	res = image[minx:maxx, miny:maxy, :]
	print(minx, maxx, miny, maxy)
	return res, ((minx, maxx, miny, maxy))


def remove_same(arr):
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
				if sq > sq1 * 0.95:
					arr.pop(i)
					change = True
					break
				elif sq > sq2 * 0.95:
					arr.pop(j)
					change = True
					break
			if change:
				break
	return arr

def find_all_contours(image, i):
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
	for elem in images:
		i += 1
		cv.imwrite("{0}.png".format(i), elem[0])

	return images, image_print, i




j = 0
for i in range(6):
	image = cv.imread("example{0}.jpg".format(i))
	contours, image_p, j = find_all_contours(image, j)
	cv.imwrite("output{0}.jpg".format(i), image_p)

