import math
import numpy as np
from os.path import dirname, abspath, join
import idx2numpy
from matplotlib import pyplot as plt


# The Sigmoid function, which describes an S shaped curve.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The derivative of the Sigmoid function.
# This is the gradient of the Sigmoid curve.
def sigmoid_derivative(x):
    return x * (1.0 - x)


def ReLU(x):
    return np.maximum(x, 0)


def ReLU_derivative(x):
    return x > 0


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def one_hot_encoder(x):
    one_hot = np.zeros((x.size, x.max() + 1))
    one_hot[np.arange(x.size), x] = 1
    one_hot = one_hot.T
    return one_hot


# https://stackoverflow.com/questions/47493559/valueerror-non-broadcastable-output-operand-with-shape-3-1-doesnt-match-the
class NeuralNetwork:

    def __init__(self, hidden_layers_amount, hidden_nodes_amount, learning_rate, activation_function, seed=None):
        # Set seed for random number generator
        self.rng = np.random.default_rng(seed)

        self.input_nodes_amount = 784
        self.output_nodes_amount = 10
        self.hidden_layers_amount = hidden_layers_amount
        self.hidden_nodes_amount = hidden_nodes_amount
        self.learning_rate = learning_rate
        self.a_function = activation_function

        self.weights = []
        self.biases = []

        # Generate weights for each hidden layer + the output layer
        for i in range(self.hidden_layers_amount + 1):
            if i == 0:
                self.weights.append(self.generate_weights(self.input_nodes_amount, self.hidden_nodes_amount))
            elif i == self.hidden_layers_amount:
                self.weights.append(self.generate_weights(self.hidden_nodes_amount, self.output_nodes_amount))
            else:
                self.weights.append(self.generate_weights(self.hidden_nodes_amount, self.hidden_nodes_amount))

        # https://medium.com/@glenmeyerowitz/bias-initialization-in-a-neural-network-2e5d26fed0f0
        # Set bias for each hidden layer + the output layer to 0
        for i in range(self.hidden_layers_amount + 1):
            if i == self.hidden_layers_amount:
                self.biases.append(np.zeros((self.output_nodes_amount, 1)))
            else:
                self.biases.append(np.zeros((self.hidden_nodes_amount, 1)))

    def generate_weights(self, previous_nodes_amount, next_nodes_amount):
        # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        # Xavier weight initialization
        if self.a_function == 'sigmoid':
            # calculate the range for the weights
            lower, upper = -(1.0 / math.sqrt(previous_nodes_amount)), (1.0 / math.sqrt(previous_nodes_amount))
            # generate random numbers
            numbers = self.rng.random((next_nodes_amount, previous_nodes_amount))
            # scale to the desired range
            return lower + numbers * (upper - lower)
        # He weight initialization
        else:
            # calculate the range for the weights
            std = math.sqrt(2.0 / previous_nodes_amount)
            # generate random numbers
            numbers = self.rng.random((next_nodes_amount, previous_nodes_amount))
            # scale to the desired range
            return numbers * std

    def activation_function(self, x):
        activation_functions = ['sigmoid', 'relu']

        # Raise error if given function is not valid
        if self.a_function not in activation_functions:
            raise ValueError("Invalid activation function type. Expected one of: %s" % activation_functions)

        if self.a_function == 'sigmoid':
            return sigmoid(x)
        else:
            return ReLU(x)

    def activation_function_derivative(self, x):
        activation_functions = ['sigmoid', 'relu']

        # Raise error if given function is not valid
        if self.a_function not in activation_functions:
            raise ValueError("Invalid activation function type. Expected one of: %s" % activation_functions)

        if self.a_function == 'sigmoid':
            return sigmoid_derivative(x)
        else:
            return ReLU_derivative(x)

    def forward_propagation(self, input_vector):
        # Perform calculations for each hidden layer + the output layer
        # Z: Dot product of input and weight + bias, A: Activation of Z
        Z = []
        A = []
        for i in range(self.hidden_layers_amount + 1):
            if i == 0:
                Z.append(np.dot(self.weights[i], input_vector) + self.biases[i])
                A.append(self.activation_function(Z[i]))
            # Use softmax function on the output layer
            elif i == self.hidden_layers_amount:
                Z.append(np.dot(self.weights[i], A[i-1]) + self.biases[i])
                A.append(softmax(Z[i]))

            else:
                Z.append(np.dot(self.weights[i], A[i-1]) + self.biases[i])
                A.append(self.activation_function(Z[i]))
        return Z, A

    def backward_propagation(self, Z, A,  input_vector, output_vector):
        # Perform calculations for each hidden layer + the output layer
        # Z: Dot product of input and weight + bias, A: Activation of Z
        # input_vector.shape[1]: m
        m = input_vector.shape[1]
        Z_derivatives = []
        weights_derivatives = []
        biases_derivatives = []
        for i in range(self.hidden_layers_amount, -1, -1):
            if i == 0:
                Z_derivatives.append(np.dot(self.weights[i + 1].T, Z_derivatives[len(Z_derivatives) - 1]) * ReLU_derivative(Z[i]))
                weights_derivatives.append(1 / m * np.dot(Z_derivatives[i], input_vector.T))
                biases_derivatives.append(1 / m * np.sum(Z_derivatives[i]))
            elif i == self.hidden_layers_amount:
                Z_derivatives.append(A[i] - one_hot_encoder(output_vector))
                weights_derivatives.append(1 / m * np.dot(Z_derivatives[0], A[i - 1].T))
                biases_derivatives.append(1 / m * np.sum(Z_derivatives[0]))
            else:
                Z_derivatives.append(np.dot(self.weights[i + 1], Z_derivatives[len(Z_derivatives)-1]) * ReLU_derivative(Z[i]))
                weights_derivatives.append(1 / m * np.dot(Z_derivatives[len(Z_derivatives)-1], A[i - 1].T))
                biases_derivatives.append(1 / m * np.sum(Z_derivatives[len(Z_derivatives)-1]))

        return weights_derivatives, biases_derivatives

    def update_parameters(self, weights_derivatives, biases_derivatives):
        weights_derivatives.reverse()
        biases_derivatives.reverse()
        for i in range(self.hidden_layers_amount + 1):
            self.weights[i] = self.weights[i] - self.learning_rate * weights_derivatives[i]
            self.biases[i] = self.biases[i] - self.learning_rate * biases_derivatives[i]

    # Need to implement SGD
    def gradient_descent(self, input_vector, output_vector, epochs):
        for i in range(epochs):
            Z = []
            A = []
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
        print(predictions, output_vector)
        return np.sum(predictions == output_vector) / output_vector.size

    def make_predictions(self, input_vector):
        _, output_layer = self.forward_propagation(input_vector)
        predictions = self.get_predictions(output_layer)
        return predictions


seed = 1
rng = np.random.default_rng(seed)

data_path = join(dirname(dirname(abspath(__file__))), 'data')

# Load image and label files and convert them to numpy array
image_file = join(data_path, 'train-images-idx3-ubyte')
label_file = join(data_path, 'train-labels-idx1-ubyte')

image_array = idx2numpy.convert_from_file(image_file)
label_array = idx2numpy.convert_from_file(label_file)

# Reshape the data so it's 60000, 784 instead of 60000, 28, 28 and add a label column
data = np.c_[label_array, image_array.reshape(60000, 784)]

row_amount, column_amount = data.shape
rng.shuffle(data)  # shuffle before splitting into dev and training sets

testing_data = data[0:15000].T
testing_data_Y = testing_data[0]
testing_data_X = testing_data[1:column_amount] / 255

training_data = data[45000:row_amount].T
training_data_Y = training_data[0]
training_data_X = training_data[1:column_amount] / 255
_, training_data_column_amount = training_data_X.shape

model = NeuralNetwork(hidden_layers_amount=1, hidden_nodes_amount=10, learning_rate=.1, activation_function="relu", seed=seed)

model.gradient_descent(training_data_X, training_data_Y, epochs=500)

'''
print(train_label_array[70])
plt.imshow(train_image_array[70], cmap=plt.cm.binary)
plt.show()
'''
