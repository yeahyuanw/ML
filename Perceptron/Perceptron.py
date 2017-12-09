#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import codecs
import random
import numpy as np

def readData(path) :
	if not os.path.exists(path):
		print(path + u'not exists...')
		return None, None

	labels = []
	data = []
	with codecs.open(path,'r') as f:
		for line in f.readlines():
			temp = line.split(',')
			labels.append(float(temp[-1]))
			data.append([ float(t) for t in temp[0:-1]])
	return data,labels

'''
	#train model#
	datas : train datas
	labels : train labels
'''

def train(datas, labels) :

	global alpha, total_study, con_right_limit

	train_size = len(datas)
	feature_len = len(datas[0])

	#initialize model parameter w and b
	w = np.zeros((feature_len, 1))
	b = 0

	continous_right = 0	# the consecutive correct number of classifications

	while total_study >= 0 :

		if continous_right > con_right_limit :
			break

		#Stochastic gradient descent
		index = random.choice(np.arange(train_size))
		example = datas[index]
		label = labels[index]

		res = label*(np.dot(example, w) + b)

		if res <= 0 :
			continous_right = 0
			example = np.reshape(example, (feature_len, 1))
			w += example*label*alpha
			b += label*alpha

			total_study -= 1
		else :
			continous_right += 1
	return w, b

def prediction(datas, labels, w, b) :
	test_size = len(datas)

	predict_positive = 0
	label_positive = 0
	pos_predict_pos = 0
	for i in np.arange(test_size):
		res = np.dot(datas[i], w) + b
		if res >= 0 :
			predict_positive += 1
		if labels[i] > 0 :
			label_positive += 1
		if res >= 0 and labels[i] > 0 :
			pos_predict_pos += 1

	precision = pos_predict_pos/predict_positive
	recall = pos_predict_pos/label_positive
	print(str(pos_predict_pos) + " " + str(predict_positive) + " " + str(label_positive))
	return precision, recall, 2*precision*recall/(precision + recall)


alpha = 0.001	#study step
total_study = 1000	#while study_nums <= total_study, while loop
con_right_limit = 100	# the upper number of consecutive classification correct
k = 10

if __name__ == '__main__' :
	datas, labels = readData('E:\\datasetsForPractice\\Dataset\\german.txt')
	data_size = len(datas)

	k_size = data_size//k

	for n in np.arange(1, k + 1):
		test_index = [i for i in np.arange((n-1)*k_size, n*k_size)]
		w, b = train(\
			[datas[i] for i in np.arange(data_size) if i not in test_index], \
			[labels[i] for i in np.arange(data_size) if i not in test_index])

		pre, recall , f1 = prediction(\
			[datas[i] for i in test_index], \
			[labels[i] for i in test_index], \
			w, \
			b)

		print("alpha : " + str(alpha) + \
			" pre : " + str(pre) + \
			" recall : " + str(recall) + \
			" f score : " + str(f1))
