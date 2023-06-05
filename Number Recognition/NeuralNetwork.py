import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes):
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s) / s[1] ** 0.5 for s in weight_shapes]
        self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]

    def predict(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.matmul(w, a) + b)
        return a

    def train(self, training_images, training_labels, epochs, learning_rate):
        for epoch in range(epochs):
            self.print_accuracy(training_images, training_labels)
            for i in range(len(training_images)):
                x = training_images[i]
                y = training_labels[i]

                # Generate random noise and offset for each image
                noise = np.random.normal(0, 0.1, x.shape)
                offset = np.random.uniform(-0.1, 0.1, x.shape)

                # Apply transformations
                augmented_x = x + noise + offset
                augmented_x = np.clip(augmented_x, 0, 1)  # Clip values to [0, 1] range

                # Forward propagation
                activations = [augmented_x]
                for w, b in zip(self.weights, self.biases):
                    augmented_x = self.activation(np.matmul(w, augmented_x) + b)
                    activations.append(augmented_x)

                # Backpropagation
                deltas = [(activations[-1] - y) * self.activation_derivative(activations[-1])]
                for w, a in zip(reversed(self.weights[1:]), reversed(activations[1:-1])):
                    deltas.append(np.matmul(w.transpose(), deltas[-1]) * self.activation_derivative(a))
                deltas.reverse()

                # Update weights and biases
                for j in range(len(self.weights)):
                    self.weights[j] -= learning_rate * np.matmul(deltas[j], activations[j].transpose())
                    self.biases[j] -= learning_rate * deltas[j]

        # Save the trained model
        with open('trained_model.pkl', 'wb') as file:
            pickle.dump(self, file)

    def print_accuracy(self, images, labels):
        predictions = self.predict(images)
        num_correct = sum([np.argmax(a) == np.argmax(b) for a, b in zip(predictions, labels)])
        print('{0}/{1} accuracy: {2}%'.format(num_correct, len(images), (num_correct / len(images)) * 100))

    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def activation_derivative(x):
        return x * (1 - x)
