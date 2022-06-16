import numpy as np
import pandas as pd

def sigmoid_function(z):
	sigmoid = 1/(1+np.exp(-z))

	return sigmoid

def loss_function(y, y_pre):
	loss = -[y*np.log(y_pre)+(1-y)np.log(1-y_pre)]

	return loss

def gradient_descent(X, y, y_pre):
	m = X.shape[0]
	dw = (1/m)*np.dot(X.T, (y_pre-y))
	db = (1/m)*np.sum((y_pre-y))

	return dw, db

def train(X, y, epochs, batch_size, learning_rate):
	m, n = X.shape
	
