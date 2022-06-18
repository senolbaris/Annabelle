import numpy as np
import pandas as pd

def sigmoid(z):
	sigmoid = 1/(1+np.exp(-z))

	return sigmoid

def binary_cross_entropy_loss(y, y_pre):
	loss = -np.mean(y*(np.log(y_pre)) - (1-y)*np.log(1-y_pre))

	return loss

def gradient_descent(x, y_pre, y, learning_rate, w, b):
	m = x.shape[0]
	dw = (1/m)*np.dot(x.T, (y_pre-y))
	db = (1/m)*np.sum((y_pre-y))

	w = w - learning_rate*dw
	b = b - learning_rate*db

	return w, b

def train(X, y, epochs, batch_size, learning_rate):
	m, n = X.shape
	w = np.zeros((n,1))
	b = 0
	y = y.reshape(m,1)
	losses = []

	for epoch in range(epochs):
		for i in range((m-1)//batch_size + 1):
			start = i*batch_size
			end = start+batch_size
			xb = X[start:end]
			yb = y[start:end]

			Z = (np.dot(xb, w)+b)
			y_pre = sigmoid(Z)

			w, b = gradient_descent(xb, y_pre, yb, learning_rate, w, b)

		loss = binary_cross_entropy_loss(y, sigmoid(np.dot(X, w)+b))
		losses.append(loss)

	return w, b, losses

def predict(X):
	preds = sigmoid(np.dot(X, w) + b)

	pred_class = []
	pred_class = [1 if i > 0.5 else 0 for i in preds]

	return np.array(pred_class)


def accuracy(y, y_hat):
	accuracy = np.sum(y == y_hat) / len(y)
	return accuracy

