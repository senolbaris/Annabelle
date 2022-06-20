import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize


ex = ["She loves pizza, pizza is delicious.",
	"she is a good person.",
	"good people are the best."]

def one_hot_encoding(data):
	unique_words = []
	for sentence in data:
		words = word_tokenize(sentence)
		for word in words:
			if word not in unique_words:
				unique_words.append(word)

	ohe = []
	for word in unique_words:
		onehot = []
		words = word_tokenize(sentence)
		for word in unique_words:
			if word in words:
				onehot.append(1)
			else:
				onehot.append(0)

		ohe.append(onehot)

	return np.array(ohe)

def bag_of_words(data):
	unique_words = []
	for sentence in data:
		words = word_tokenize(sentence)
		for word in words:
			if word not in unique_words:
				unique_words.append(word)
	values = [0 for x in range(len(unique_words))]
	
	bag_of_words = []
	for sentence in data:
		bow = dict(zip(unique_words, values))
		words = word_tokenize(sentence)
		for word in words:
			if word in unique_words:
				bow[word] += 1
		bag_of_words.append(np.fromiter(bow.values(), dtype=int))
		
	return np.array(bag_of_words)






