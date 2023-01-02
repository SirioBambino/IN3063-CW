import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def confusion_matrix(predictions, labels):
    # Get the classes and initialise matrix
    classes = np.unique(labels)
    matrix = np.zeros((len(classes), len(classes)))

    # Iterate through combinations of classes
    for i in range(len(classes)):
        for j in range(len(classes)):
            # Get the sum of each combination and add it to matrix
            matrix[i, j] = np.sum((labels == classes[i]) & (predictions == classes[j]))

    # Convert matrix to int
    matrix = matrix.astype(float).astype(int)

    # Create confusion matrix image
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap='summer')

    # Plot the numbers in the confusion matrix
    for (x, y), value in np.ndenumerate(matrix):
        plt.text(y, x, value, ha='center', va='center', fontsize=10)

    # Display all tick labels
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # Display color bar
    fig.colorbar(cax)
    plt.show()

    return matrix
