import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)

@np.vectorize
def sigmoid_derivative(x):
    return x * (1.0 - x)

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_derivative(x):
    return x > 0

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


class NeuralNetwork:

    def __init__(self, hidden_layers_amount, hidden_nodes_amount, learning_rate, activation_function, seed=None):
        # Set seed for random number generator
        self.rng = np.random.default_rng(seed)

        self.input_nodes_amount = 784
        self.output_nodes_amount = 10
        self.hidden_layers_amount = 1
        self.hidden_nodes_amount = hidden_nodes_amount
        self.learning_rate = learning_rate
        self.activation_function = activation_function

        self.weights = []
        self.biases = []

        # Generate weights for each hidden layer + the output layer
        for i in range(self.hidden_layers_amount + 1):
            if i == 0:
                self.generate_weights(self.input_nodes_amount, self.hidden_nodes_amount)
            elif i == self.hidden_layers_amount + 1:
                self.generate_weights(self.hidden_nodes_amount, self.output_nodes_amount)
            else:
                self.generate_weights(self.hidden_nodes_amount, self.hidden_nodes_amount)

    def generate_weights(self, previous_nodes_amount, next_nodes_amount):
        # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        # Xavier weight initialization
        if self.activation_function == 'sigmoid':
            p = previous_nodes_amount
            n = next_nodes_amount
            # calculate the range for the weights
            lower, upper = -(math.sqrt(6.0) / math.sqrt(p + n)), (math.sqrt(6.0) / math.sqrt(p + n))
            # generate random numbers
            numbers = self.rng.integers(1000)
            # scale to the desired range
            return lower + numbers * (upper - lower)
        # He weight initialization
        else:
            n = previous_nodes_amount
            # calculate the range for the weights
            std = math.sqrt(2.0 / n)
            # generate random numbers
            numbers = self.rng.integers(1000)
            # scale to the desired range
            return numbers * std

    def activation_function(self, x):
        activation_functions = ['sigmoid', 'relu']

        # Raise error if given function is not valid
        if self.activation_function not in activation_functions:
            raise ValueError("Invalid activation function type. Expected one of: %s" % activation_functions)

        if self.activation_function == 'sigmoid':
            return sigmoid(x)
        else:
            return ReLU(x)

    def activation_function_derivative(self, x):
        activation_functions = ['sigmoid', 'relu']

        # Raise error if given function is not valid
        if self.activation_function not in activation_functions:
            raise ValueError("Invalid activation function type. Expected one of: %s" % activation_functions)

        if self.activation_function == 'sigmoid':
            return sigmoid_derivative(x)
        else:
            return ReLU_derivative(x)

    def forward_propagation(self, input_vector):
        # Perform calculations for each hidden layer + the output layer
        # Z: Dot product of input and weight + bias, A: Activation of Z
        # Input vector: X
        Z = []
        A = []
        for i in range(self.hidden_layers_amount + 1):
            # Use softmax function on the output layer
            if i == self.hidden_layers_amount:
                Z.append(np.dot(self.weights[i], input_vector) + self.biases[i])
                A.append(softmax(Z[i]))

            else:
                Z.append(np.dot(self.weights[i], input_vector) + self.biases[i])
                A.append(self.activation_function(Z[i]))
        return Z, A

    def backward_propagation(self, Z, A, input_vector, output_vector):
        # Perform calculations for each hidden layer + the output layer
        # Z: Dot product of input and weight + bias, A: Activation of Z
        # Input vector: X, Output vector: Y
        # input_vector.shape[0]: m
        Z_derivatives = []
        weights_derivatives = []
        biases_derivatives = []
        for i in range(self.hidden_layers_amount + 1, 0):
            if i == self.hidden_layers_amount:
                Z_derivatives.append(A[i] - one_hot(output_vector))
                weights_derivatives.append(1 / input_vector.shape[0] * np.dot(Z_derivatives[0], A[i-1].T))
                biases_derivatives.append(1 / input_vector.shape[0] * np.sum(Z_derivatives[0]))
            else:
                Z_derivatives.append(np.dot(self.weights[i+1], Z_derivatives[i+1]) * ReLU_derivative(Z[i]))
                weights_derivatives.append(1 / input_vector.shape[0] * np.dot(Z_derivatives[i], input_vector))
                biases_derivatives.append(1 / input_vector.shape[0] * np.sum(Z_derivatives[i]))

        return weights_derivatives, biases_derivatives

    def update_parameters(self, weights_derivatives, biases_derivatives):
        for i in range(self.hidden_layers_amount + 1):
            self.weights[i] = self.weights[i] - self.learning_rate * weights_derivatives[i]
            self.biases[i] = self.biases[i] - self.learning_rate * biases_derivatives[i]

    # Need to implement SGD
    def gradient_descent(self, input_vector, output_vector, epochs):
        for i in range(epochs):
            for j in range(self.hidden_layers_amount + 1):
                # Z: Dot product of input and weight + bias, A: Activation of Z
                # Input vector: X, Output vector: Y
                Z, A = self.forward_propagation(input_vector)
                weights_derivatives, biases_derivatives = self.backward_propagation(Z, A, input_vector, output_vector)
                self.update_parameters(weights_derivatives, biases_derivatives)
                if i % 10 == 0:
                    print("Iteration: ", i)
                    predictions = self.get_predictions(A[self.hidden_layers_amount])
                    print(self.get_accuracy(predictions, output_vector))

    def get_predictions(self, output_layer):
        return np.argmax(output_layer, 0)

    def get_accuracy(self, predictions, output_vector):
        # print(predictions, output_vector)
        return np.sum(predictions == output_vector) / output_vector.size

    def make_predictions(self, input_vector):
        _, output_layer = self.forward_propagation(input_vector)
        predictions = self.get_predictions(output_layer)
        return predictions
