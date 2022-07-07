import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize


class OneHotEncoding:
	def __init__(self):
		self.unique_words = []
		self.one_hot_encoding = []

	def fit(self, data):
		for sentence in data:
			words = word_tokenize(sentence)
			for word in words:
				if word not in self.unique_words:
					self.unique_words.append(word)

		return self.unique_words

	def transform(self, data):
		for sentence in data:
			ohe = []
			words = word_tokenize(sentence)
			for word in self.unique_words:
				if word in words:
					ohe.append(1)
				else:
					ohe.append(0)
			self.one_hot_encoding.append(ohe)

		return self.one_hot_encoding

	def fit_transform(self, data):
		self.fit(data)
		self.transform(data)

		return self.one_hot_encoding



class BagOfWords:
	def __init__(self):
		self.unique_words = []
		self. bag_of_words = []
		self.values = []

	def fit(self, data):
		for sentence in data:
			words = word_tokenize(sentence)
			for word in words:
				if word not in self.unique_words:
					self.unique_words.append(word)
		self.values = [0 for x in range(len(self.unique_words))]

		return self.unique_words

	def transform(self, data):
		for sentence in data:
			bow = dict(zip(self.unique_words, self.values))
			words = word_tokenize(sentence)
			for word in words:
				if word in self.unique_words:
					bow[word] += 1
			self.bag_of_words.append(np.fromiter(bow.values(), dtype=int))
		
		return self.bag_of_words

	def fit_transform(self, data):
		self.fit(data)
		self.transform(data)

		return self.bag_of_words



