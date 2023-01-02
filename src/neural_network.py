import math
from time import process_time
from os.path import dirname, abspath, join

import idx2numpy
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(inputs):
    return 1 / (1 + np.exp(-inputs))

def sigmoid_derivative(inputs):
    return inputs * (1.0 - inputs)

def ReLU(inputs):
    return np.maximum(inputs, 0)

def ReLU_derivative(inputs):
    return inputs > 0

def softmax(inputs):
    return np.exp(inputs) / sum(np.exp(inputs))

# MSE: Mean Squared Error
def calculate_MSE(y_predictions, y_true):
    return np.square(np.subtract(y_true, y_predictions)).mean()

def calculate_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

def one_hot_encoder(x):
    one_hot = np.zeros((x.size, 10))
    one_hot[np.arange(x.size), x] = 1
    one_hot = one_hot.T
    return one_hot


class NeuralNetwork:

    # Initialise the network's parameters, including weights and biases
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

    # Generate the weights depending on the activation function
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

    # Returns the result of the chosen activation function
    def activation_function(self, x):
        activation_functions = ['sigmoid', 'relu']

        # Raise error if given function is not valid
        if self.a_function not in activation_functions:
            raise ValueError("Invalid activation function type. Expected one of: %s" % activation_functions)

        if self.a_function == 'sigmoid':
            return sigmoid(x)
        else:
            return ReLU(x)

    # Returns the result of the derivative of the chosen activation function
    def activation_function_derivative(self, x):
        activation_functions = ['sigmoid', 'relu']

        # Raise error if given function is not valid
        if self.a_function not in activation_functions:
            raise ValueError("Invalid activation function type. Expected one of: %s" % activation_functions)

        if self.a_function == 'sigmoid':
            return sigmoid_derivative(x)
        else:
            return ReLU_derivative(x)

    # Perform calculations for forward propagation for each layer
    def forward_propagation(self, X, dropout_rate):
        # Z: Dot product of input and weight + bias
        # A: Activation of Z
        Z, A = [], []
        for i in range(self.hidden_layers_amount + 1):
            # On the output layer find the dot of weights and the previous
            # layer's A, add the bias and pass the result through the
            # softmax function
            if i == self.hidden_layers_amount:
                Z.append(np.dot(self.weights[i], A[i - 1]) + self.biases[i])
                A.append(softmax(Z[i]))
            else:
                # On the input layer find the dot of weights and X, add the
                # bias and pass the result through the activation function
                if i == 0:
                    Z.append(np.dot(self.weights[i], X) + self.biases[i])
                    A.append(self.activation_function(Z[i]))
                # On any other layer find the dot of weights and the previous
                # layer's A, add the bias and pass the result through the
                # activation function
                else:
                    Z.append(np.dot(self.weights[i], A[i-1]) + self.biases[i])
                    A.append(self.activation_function(Z[i]))

                # Implement dropout by randomly setting some nodes to 0 based on the dropout rate
                if dropout_rate > 0:
                    layer = A[len(A)-1].T
                    input_size, layer_size = layer.shape
                    # Use dropout rate to calculate the number of nodes to drop out, then create a matrix of random
                    # indexes that represent the nodes that will be dropped
                    dropped_nodes_amount = round(layer_size * dropout_rate)
                    dropped_nodes = self.rng.integers(0, layer_size, (input_size, layer_size))
                    dropped_nodes = dropped_nodes.argpartition(dropped_nodes_amount,axis=1)[:,:dropped_nodes_amount]
                    # Set value of the dropped nodes in the layer to 0
                    index = 0
                    for input in dropped_nodes:
                        for node in input:
                            layer[index, node] = 0
                        index += 1

        return Z, A

    # Perform calculations for backward propagation for each layer
    def backward_propagation(self, Z, A, X, y):
        # Z: Dot product of input and weight + bias, A: Activation of Z
        m = X.shape[1]
        Z_derivatives = []
        weights_derivatives = []
        biases_derivatives = []
        # Loop through the layers backwards starting from the output layer and finishing on the first hidden layer
        for i in range(self.hidden_layers_amount, -1, -1):
            # Find the derivatives of the weights and biases for each layer
            if i == 0:
                Z_derivatives.append(np.dot(self.weights[i + 1].T, Z_derivatives[len(Z_derivatives) - 1]) * ReLU_derivative(Z[i]))
                weights_derivatives.append(1 / m * np.dot(Z_derivatives[len(Z_derivatives) - 1], X.T))
                biases_derivatives.append(1 / m * np.sum(Z_derivatives[len(Z_derivatives) - 1]))
            elif i == self.hidden_layers_amount:
                Z_derivatives.append(A[i] - one_hot_encoder(y))
                weights_derivatives.append(1 / m * np.dot(Z_derivatives[0], A[i - 1].T))
                biases_derivatives.append(1 / m * np.sum(Z_derivatives[0]))
            else:
                Z_derivatives.append(np.dot(self.weights[i + 1].T, Z_derivatives[len(Z_derivatives) - 1]) * ReLU_derivative(Z[i]))
                weights_derivatives.append(1 / m * np.dot(Z_derivatives[len(Z_derivatives) - 1], A[i - 1].T))
                biases_derivatives.append(1 / m * np.sum(Z_derivatives[len(Z_derivatives) - 1]))

        return weights_derivatives, biases_derivatives

    # Update weights and biases using their derivatives and the learning rate
    def update_parameters(self, weights_derivatives, biases_derivatives):
        weights_derivatives.reverse()
        biases_derivatives.reverse()
        for i in range(self.hidden_layers_amount + 1):
            self.weights[i] -= self.learning_rate * weights_derivatives[i]
            self.biases[i] -= self.biases[i] - self.learning_rate * biases_derivatives[i]

    # Train the network against the given data
    def fit(self, X, y, epochs, batch_size, stopping_threshold, dropout_rate):
        accuracy_list = []
        loss_list = []
        X = X.T
        for epoch in range(epochs):
            print("Epoch: {0}/{1}".format(epoch + 1, epochs))
            # Start counter to calculate run time of epoch
            epoch_start = process_time()

            # Shuffle the data
            shuffler = self.rng.permutation(len(X))
            X = X[shuffler]
            y = y[shuffler]

            accuracy = 0
            loss = 0
            count = 0

            # Iterate through each batch of the dataset
            for start in range(0, X.shape[0], batch_size):

                # Start counter to calculate run time of iteration
                iteration_start = process_time()

                end = start + batch_size
                X_batch = X[start:end].T
                y_batch = y[start:end]

                # Perform forward propagation, backward propagation and then update the weights and biases
                Z, A = self.forward_propagation(X_batch, dropout_rate)
                weights_derivatives, biases_derivatives = self.backward_propagation(Z, A, X_batch, y_batch)
                self.update_parameters(weights_derivatives, biases_derivatives)

                # End counter to calculate run time of iteration
                iteration_end = process_time()

                count += 1

                # Print some information about the model every 50 iterations
                if count % 100 == 0:
                    # Forward propagation
                    Z, A = self.forward_propagation(X_batch, 0)

                    # Calculate the accuracy and mean squared error and save results to arrays
                    predictions = np.argmax(A[self.hidden_layers_amount], 0)
                    accuracy = calculate_accuracy(predictions, y_batch)
                    loss = calculate_MSE(predictions, y_batch)

                    accuracy_list.append(accuracy)
                    loss_list.append(loss)

                    print("Iteration:", count,
                          "Accuracy: {0:.4f}".format(accuracy),
                          "Loss: {0:.4f}".format(loss),
                          "Computation time: {0:.2f}ms".format((iteration_end - iteration_start) * 1000))

            # Implement early stopping if loss doesn't improve fast enough
            stopping_patience = round(epochs * 0.3)
            # Stop the loop if in the last 30% of epochs the loss has dropped less than the stopping threshold
            if epoch > stopping_patience and loss - loss_list[len(loss_list) - stopping_patience] > -stopping_threshold:
                print("Early stopping | Epoch: {0}/{1}".format(epoch, epochs))
                print("Accuracy: {0:.4f}".format(accuracy), "Loss: {0:.4f}".format(loss))
                break

            # End counter to calculate run time of epoch
            epoch_end = process_time()
            print("Epoch computation time: {0:.2f}s\n".format((epoch_end - epoch_start)))

        return accuracy_list, loss_list

    def predict(self, X):
        _, A = self.forward_propagation(X, 0)
        predictions = np.argmax(A[len(A)-1], 0)
        return predictions

    def test_prediction(self, index, X, y):
        current_image = X[:, index, None]
        prediction = np.argmax(X[:, index, None], 0)
        label = y[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = current_image.reshape((28, 28)) * 255
        plt.imshow(current_image, cmap=plt.cm.binary)
        plt.show()


def load_mnist():
    data_path = join(dirname(dirname(abspath(__file__))), 'data')

    # Load image and label files and convert them to numpy array
    image_file = join(data_path, 'MNIST/train-images-idx3-ubyte')
    label_file = join(data_path, 'MNIST/train-labels-idx1-ubyte')

    image_array = idx2numpy.convert_from_file(image_file)
    label_array = idx2numpy.convert_from_file(label_file)

    # Reshape the data so it's 60000, 784 instead of 60000, 28, 28 and add label column
    data = np.c_[label_array, image_array.reshape(60000, 784)]

    return data
