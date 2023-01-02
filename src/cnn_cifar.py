from time import process_time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.ToTensor())
testing_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transforms.ToTensor())

training_loader = torch.utils.data.DataLoader(training_data, batch_size=100)
testing_loader = torch.utils.data.DataLoader(testing_data, batch_size=100, shuffle=True)


def calculate_accuracy(predictions, labels):
    return (predictions == labels).sum() / len(labels)


class CNNCIFAR(nn.Module):

    def __init__(self):
        super(CNNCIFAR, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.BatchNorm2d(256),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128)
        )

        self.layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256)
        )

        self.layer_5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=512),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # Pass input through first set of Convolutional -> RelU -> Pooling layers
        output = self.layer_1(x)
        # Pass output from the previous layer through second set of Convolutional -> RelU -> Pooling layers
        output = self.layer_2(output)
        # Pass output from the previous layer through third set of Convolutional -> RelU -> Pooling layers
        output = self.layer_3(output)
        # Pass output from the previous layer through fourth set of Convolutional -> RelU -> Pooling layers
        output = self.layer_4(output)
        # Flatten output from the previous layer and pass it through set of 3 Fully Connected layers with
        # dropout between them
        output = self.layer_5(output)

        return output

    # Train the network against the given data
    def fit(self, data_loader, epochs, error_function, optimiser):
        accuracy_list = []
        loss_list = []
        for epoch in range(epochs):
            print("Epoch: {0}/{1}".format(epoch + 1, epochs))
            # Start counter to calculate run time of epoch
            epoch_start = process_time()
            count = 0
            for images, labels in data_loader:

                # Start counter to calculate run time of iteration
                iteration_start = process_time()

                # Set model to training mode
                self.train()

                # Forward propagation and calculating loss
                outputs = self(images)
                loss = error_function(outputs, labels)
                # Initializing a gradient as 0 so there is no mixing of gradient among the batches
                optimiser.zero_grad()
                # Backward propagation and optimising the parameters
                loss.backward()
                optimiser.step()

                # End counter to calculate run time of iteration
                iteration_end = process_time()

                count += 1

                # Print some information about the model every 50 iterations
                if count % 50 == 0:
                    # Set model to evaluation mode
                    self.eval()

                    # Forward propagation
                    outputs = self(images)

                    # Calculate loss and accuracy  and save results to arrays
                    loss = error_function(outputs, labels)
                    predictions = torch.max(outputs, 1)[1]
                    accuracy = calculate_accuracy(predictions, labels)

                    accuracy_list.append(accuracy)
                    loss_list.append(loss.data)

                    print("Iteration:", count,
                          "Accuracy: {0:.4f}".format(accuracy),
                          "Loss: {0:.4f}".format(loss),
                          "Computation time: {0:.2f}ms".format((iteration_end - iteration_start) * 1000))

            # End counter to calculate run time of epoch
            epoch_end = process_time()
            print("Epoch computation time: {0:.2f}s\n".format((epoch_end - epoch_start)))

        return accuracy_list, loss_list

    def predict(self, data_loader, error_function):
        outputs = torch.Tensor()
        all_labels = torch.Tensor()

        # Set model to evaluation mode
        self.eval()
        for images, labels in data_loader:
            outputs = torch.cat((outputs, self(images)))
            all_labels = torch.cat((all_labels, labels))

        predictions = torch.max(outputs, 1)[1]

        loss = error_function(outputs, all_labels.type(torch.LongTensor))
        accuracy = calculate_accuracy(predictions, all_labels)
        print("Accuracy: {0:.4f}".format(accuracy), "Loss: {0:.4f}".format(loss))
        return predictions, all_labels
