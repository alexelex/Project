import numpy as np
from sklearn.svm import LinearSVC
from math import pi
from skimage.transform import resize
from skimage.color import rgb2gray


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
	image = resize(rgb2gray(image), size)
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
