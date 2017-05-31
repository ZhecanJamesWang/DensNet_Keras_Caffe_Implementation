import cv2
import numpy as np
import os
import random
from random import shuffle
import shutil
from decimal import Decimal
from PIL import Image

from keras import optimizers
from keras import backend as K
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, merge, ZeroPadding2D
from keras.models import Model

# from keras.callbacks import ModelCheckpoint
# from keras import callbacks
from sklearn.preprocessing import LabelEncoder

from densenet121 import DenseNet
import utility as ut
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="0"



class trainDSN(object):
	def __init__(self):

		# self.init = True
		self.debug = True
		self.outputDir = "./output/05282017_01_output/"
		self.modelDir = "./output/05282017_01_model/"
		self.imSize = 224
		self.derivateNum = 2
		self.classNum = 46936
		self.weights_path = 'imagenet_models/densenet121_weights_tf.h5'

		# with open('dummy_labels.txt', 'r') as f:
		# 	self.labels = f.readlines()
		# print "len(self.labels): ", len(self.labels)

		TrainPath = "MScleanTrainFilterNovelBase.txt"
		# TestPath = 'MScleanTrainFilterNovelBase.txt'

		FTr = open(TrainPath,'r')
		self.DataTr = FTr.readlines()
		TrNum = len(self.DataTr)

		self.DataTe = self.DataTr[:int(len(self.DataTr)*0.1)]
		self.DataTr = self.DataTr[int(len(self.DataTr)*0.1):]
		# self.DataTe = self.DataTr

		# FTe = open(TestPath,'r')
		# self.DataTe = FTe.readlines()
		TeNum = len(self.DataTe)

		print "len(self.DataTr): ", len(self.DataTr)
		print "len(self.DataTe): ", len(self.DataTe)


		shuffle(self.DataTr)

		# print "type(DataTr): ", type(self.DataTr)
		# print "len(DataTr): ", len(self.DataTr)

		self.batch_size = 32
		self.MaxIters = int(TrNum/float(self.batch_size))
		self.MaxTestIters = int(TeNum/float(self.batch_size))

		print "train data length:", TrNum
		print "test data length:", TeNum
		print "self.MaxIters: ", self.MaxIters
		print "self.MaxTestIters: ", self.MaxTestIters

		self.labelList = pickle.load( open("uniqueLabelList.p", "rb"))

		print "len(self.labelList): ", len(self.labelList)

	def transformOneHot(self):
		print "transfer labels to one hot representation"
		labels = []
		counter = 0
		for line in self.DataTr:
			split = line.split("\t")
			label = split[1].replace("\n", "")
			# print "label: ", label
			labels.append(label)
			counter += 1
			# if counter >= 130:
			# 	break
			if counter%1000000 == 0:
				print "counter: ", counter

		encoder = LabelEncoder()
		encoder.fit(labels)
		encoded_labels = encoder.transform(labels)
		print encoded_labels
		self.dummy_labels = np_utils.to_categorical(encoded_labels)
		print "self.dummy_labels.shape:", self.dummy_labels.shape
		print "self.dummy_labels[0]: ", self.dummy_labels[0]

	def final_pred(self, y_true, y_pred):
		# y_cont=np.concatenate(y_pred,axis=1)
		return y_pred


	def DataGenBB(self, DataStrs, train_start,train_end):
		# generateFunc = ["original", "mirror", "scale", "rotate", "translate", "brightnessAndContrast"]
		generateFunc = ["original", "mirror", "rotate", "translate"]

		InputData = np.zeros([self.batch_size * self.derivateNum, self.imSize, self.imSize, 3], dtype = np.float32)
		# InputLabel = np.zeros([self.batch_size * len(generateFunc), 7], dtype = np.float32)
		InputLabel = np.zeros([self.batch_size * self.derivateNum, self.classNum], dtype = np.float32)

		InputNames = []
		count = 0
		for i in range(train_start,train_end):
			strLine = DataStrs[i]
			split = strLine.split("\t")
			imgPath = split[0].replace(".jpg", ".png")
			label = split[1].replace("\n", "")
			print "self.labelList[10:]: ", self.labelList[:10]
			labelOneHot = np.zeros(len(self.labelList))
			index = self.labelList.index(label)
			print "index: ", index
			labelOneHot[index] = 1
			label = labelOneHot

			print "label: ", label


			# label = self.dummy_labels[i]
			# label = label.replace(" \n", "")
			# label = map(int, label.split(" "))

			if self.debug:
				print "imgPath: ", imgPath

			img = cv2.imread(imgPath)
			x, y = [], []

			if img != None:
				img, x, y = ut.scale(img, x, y, imSize = self.imSize)
				print "find image: ", imgPath
				print "img.shape: ", img.shape

				for index in range(self.derivateNum):
					(w, h, _) = img.shape
					method = random.choice(generateFunc)

					if index == 0:
						method = "original"

					# if method == "resize":
					#     newImg, newX, newY = ut.resize(img, x, y, xMaxBound = w, yMaxBound = h, random = True)
					if method == "rotate":
						newImg, newX, newY = ut.rotate(img, x, y, w = w, h = h)
					elif method == "mirror":
						newImg, newX, newY = ut.mirror(img, x, y, w = w, h = h)
					elif method == "translate":
						newImg, newX, newY = ut.translate(img, x, y, w = w, h = h)
					elif method == "brightnessAndContrast":
						newImg, newX, newY = ut.contrastBrightess(img, x, y)
					elif method == "original":
						newImg, newX, newY = img, x, y
						# newImg, newX, newY = ut.scale(img, x, y, imSize = self.imSize)
					# elif method == "scale":
					# 	newImg, newX, newY = ut.scale(img, x, y, imSize = self.imSize)
					else:
						raise "not existing function"

					# if self.debug:
					# 	print "newImg.shape: ", newImg.shape
					# 	print "method: ", method
					# 	cv2.imshow("img", img)
					# 	cv2.imshow("newImg", newImg)
					# 	cv2.waitKey(0)




					# print "imgName: ", imgName.split("/")[-2] + imgName.split("/")[-1].split(".")[0]
					# print "inputCheck/"  + imgName.split("/")[-2] + imgName.split("/")[-1].split(".")[0] + str(method) + str(count) + '.jpg'
					# cv2.imwrite("inputCheck/" + imgName.split("/")[-2] + imgName.split("/")[-1].split(".")[0] + str(method) + str(count) + '.jpg', newImg)



					# print "len(InputData): ", len(InputData)
					InputData[count,...] = newImg
					InputLabel[count,...] = label
					InputNames.append(imgPath)

					# print "count: ", count
					count += 1
			else:
				print "cannot : ", imgPath


		return InputData, InputLabel, np.asarray(InputNames)


	def train_on_batch(self, nb_epoch, MaxIters):
		if os.path.exists(self.modelDir)==False:
			os.mkdir(self.modelDir)
		if os.path.exists(self.outputDir)==False:
			os.mkdir(self.outputDir)

		for e in range(nb_epoch):
			shuffle(self.DataTr)
			iterTest=0
			for iter in range (self.MaxIters):
				train_start=iter*self.batch_size
				train_end = (iter+1)*self.batch_size
				X_batch, label_BB, Z_Names = self.DataGenBB(self.DataTr, train_start=train_start, train_end=train_end)

				loss, tras, pred = self.model.train_on_batch(X_batch,label_BB)

				# print "*****"
				print "loss, train: ", loss


				if iter%1000 == 0:
				    logInfo = ""
				    if os.path.exists(self.outputDir + 'log.txt') and self.init == False:
				        f = open(self.outputDir + 'log.txt', 'a')
				    else:
				        f = open(self.outputDir + 'log.txt','w')
				        self.init = False
				
				    iterationInfo = ("^^^^^" + "\n" + 'iteration: ' + str(iter))
				    logInfo += iterationInfo
				    print iterationInfo
				
				
				    test_start = iterTest * self.batch_size
				    test_end = (iterTest + 1) * self.batch_size
				    X_batch_T, label_BB_T, Z_Names_T= self.DataGenBB(self.DataTe, train_start=test_start, train_end=test_end)
				    loss, tras, pred = self.model.evaluate(X_batch_T,label_BB_T)
				
								
				    testInfo = ("====" + "\n" + "loss, TEST: " + str(loss))
				    logInfo += testInfo
				    print testInfo
				
				
				    iterTest += self.batch_size
				    iterTest %= self.MaxTestIters
				
				    f.write(logInfo)
				    f.close()

				if iter%3000==0:
				    self.model.save(self.modelDir + '/model%d.h5'%iter)

	# def pop_layer(self, model):
	#     if not model.outputs:
	#         raise Exception('Sequential model cannot be popped: model is empty.')

	#     popped_layer = model.layers.pop()
	#     if not model.layers:
	#         model.outputs = []
	#         model.inbound_nodes = []
	#         model.outbound_nodes = []
	#     else:
	#         model.layers[-1].outbound_nodes = []
	#         model.outputs = [model.layers[-1].output]
	#     model.built = False
	#     return popped_layer

	def run(self):

		self.model = DenseNet(reduction = 0.5, classes = self.classNum, weights_path = self.weights_path)
		sgd = SGD(lr = 0.01, momentum = 0.9, nesterov = True)

		self.model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics=['accuracy', self.final_pred])

		self.model.summary()
		# self.transformOneHot()
		self.train_on_batch(1, MaxIters = 20000)


		# self.model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics=['accuracy', self.final_pred])
	   

		# print "self.model.layers[-1]: ", self.model.layers[-1]
		# self.model.layers.pop()
		# self.model.layers.pop()
		# # model.outputs = [self.model.layers[-1].output]
		# # self.model.layers[-1].outbound_nodes = []

		# print "self.model.layers[-1]: ", self.model.layers[-1]

		# last = self.model.output
		# x = Dense(1000, name='fc6')(last)
		# x = Dense(self.classNum, name='fc7')(x)
		# preds = Activation('softmax', name='prob')(x)

		# img_input = Input(shape=(224, 224, 3), name='data')

		# self.model = Model(img_input, preds)



	def main(self):
		self.run()

if __name__ == '__main__':
	trainDSN().main()
