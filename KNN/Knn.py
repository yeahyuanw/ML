#!/usr/bin/python3
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

def kdtree(train_data):

def search(test_data, root):

if '__main__' == __name__ :
	datas, labels = readData('E:\\datasetsForPractice\\Dataset\\mnist.txt')

