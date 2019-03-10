import cv2 as cv 

def find_contours(image, i):
#	cv.imwrite("enother.jpg", gray)
	square = len(image) * len(image[0])


	gray = cv.convertScaleAbs(image)
	gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
	gray = cv.GaussianBlur(gray, (5, 5), 0)

	edged = cv.Canny(gray, 10, 250)

	kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
	closed = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)
#	cv.imwrite("edged.jpg", closed)
	
	contours, hierarchy = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

	contour_list = []
	for cnt in contours:
		eps = cv.arcLength(cnt, True)
		approx = cv.approxPolyDP(cnt, 0.01*eps, True)
		area = cv.contourArea(cnt)
		if len(approx) > 8 and area > square * 0.0005:
			cv.drawContours(image, cnt, -1, (0, 255, 0), 4)
			contour_list.append(('circle', cnt))
		elif len(approx) == 5 and area > square * 0.0005:
			cv.drawContours(image, cnt, -1, (0, 255, 255), 2)
			contour_list.append(('quad', cnt))
		elif len(approx) == 3 and area > square * 0.0005:
			cv.drawContours(image, cnt, -1, (0, 0, 255), 2)
			contour_list.append(('triangle', cnt))
		elif len(approx) == 8 and area > square * 0.0005:
			cv.drawContours(image, cnt, -1, (255, 255, 0), 2)
			contour_list.append(('octagon', cnt))
		elif area > square * 0.0005:
			cv.drawContours(image, cnt, -1, (255, 0, 225), 2)
			contour_list.append(('another', cnt))
			

	cv.imwrite("output{0}.jpg".format(i), image)
	return contour_list

def contour_img(cnt, image):
	minx, maxx, miny, maxy = len(image[:, 0, 0]), 0, len(image[0, :, 0]), 0
	for i in cnt:
		print(i)
#	ball = img[280:340, 330:390]


for i in range(5):
	image = cv.imread("example{0}.jpg".format(i))
	contours = find_contours(image, i)

