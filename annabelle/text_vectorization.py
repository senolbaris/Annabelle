import numpy as np
import pandas as pd
import nltk
from ntlk.word_tokenize import word_tokenize

def one_hot_encoding(data):
	unique_words = []
	ohe = []
	for sentence in data:
		words = word_tokenize(sentence)
		for word in words:
			if word not in unique_words:
				unique_words.append(word)

	for word in unique_words:
		onehot = []
		words = word_tokenize(sentence)
		for word in words:
			if word not in unique_words:
				onehot.append(0)
			else:
				onehot.append(1)

		ohe.append(onehot)

	return ohe

