import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:
	def __init__(self, learning_rate=0.02, epochs=100, is_batch=False, batch_size=10):
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.is_batch = is_batch
		self.w = None
		self.b = None
		self.losses = []

	def sigmoid(self, z):
		sigmoid = 1/(1+np.exp(-z))

		return sigmoid

	def binary_cross_entropy_loss(self, y, y_pre):
		loss = -np.mean((y*np.log(y_pre)) + ((1-y)*np.log(1-y_pre)))

		return loss

	def gradient_descent(self, x, y, y_pre):
		m = x.shape[0]
		dw = (1/m)*np.dot(x.T, (y_pre-y))
		db =  (1/m)*np.sum(y_pre-y)
		self.w = self.w - dw*self.learning_rate
		self.b = self.b - db*self.learning_rate		

	def train(self, x, y):
		m, n = x.shape
		self.w = np.zeros((n,1))
		self.b = 0
		y = y.reshape(m,1)
		losses = []
		if self.is_batch is True:
			for epoch in range(self.epochs):
				for i in range((m-1)//self.batch_size + 1):
					start = i*self.batch_size
					end = start+self.batch_size
					xb = x[start:end]
					yb = y[start:end]
					Z = (np.dot(xb, self.w)+self.b)
					y_pre = self.sigmoid(Z)
					self.gradient_descent(xb, yb, y_pre)
				loss = self.binary_cross_entropy_loss(y, self.sigmoid(np.dot(x, self.w)+self.b))
				self.losses.append(loss)
		else:
			for epoch in range(self.epochs):
				Z = (np.dot(x, self.w)+self.b)
				y_pre = self.sigmoid(Z)
				self.gradient_descent(x, y, y_pre)
				loss = self.binary_cross_entropy_loss(y, self.sigmoid(np.dot(x, self.w)+self.b))
				self.losses.append(loss)	


	def predict(self, x):
		preds = self.sigmoid(np.dot(x, self.w) + self.b)
		pred_class = []
		pred_class = [1 if i > 0.5 else 0 for i in preds]

		return np.array(pred_class)

	def accuracy(self, y, y_pre):
		accuracy = np.sum(y == y_pre) / len(y)

		return accuracy

	def plot_loss(self):
		plt.plot(self.losses, color='b', label="Train Loss")
		plt.title("Train Loss")
		plt.xlabel("Number of Epochs")
		plt.ylabel("Loss")
		plt.show()