import NeuralNetwork as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt

with np.load('Number Recognition/mnist.npz') as data:
   training_images = data['training_images']
   training_labels = data['training_labels']

#plt.imshow(training_images[0].reshape(28,28), cmap='gray')
#plt.show()
print(training_labels[0])

layer_sizes = (784, 100, 10)

net = nn.NeuralNetwork(layer_sizes)
net.train(training_images, training_labels, epochs=10, learning_rate=0.1)

# save the trained model
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(net, file)