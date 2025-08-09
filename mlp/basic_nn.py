# build a single layer neural network from scratch using numpy
import numpy as np
import matplotlib.pyplot as plt
# create a class for a neural network

class NeuralNet:
    # contains an init, forward, and backwards pass
    def __init__(self, n_features, n_hidden, n_outputs):
        # contains hyperparameters of network
        # n_features is the number of dimensions of the input
        # initialises weights and biases
        # defines activation function
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        # 1st layer: input -> hidden
        self.w1 = np.random.rand(n_hidden, n_features) # Weights: neurons x features (each neuron touches each feature)
        self.b1 = np.random.rand(n_hidden, 1) # Biases: one per neuron in the hidden layer

        # 2nd layer: hidden -> output
        self.w2 = np.random.rand(n_outputs, n_hidden)
        self.b2 = np.random.rand(n_outputs, 1)

    def forward(self, X):
        # computes input times weight + bias
        # relu activation function
        # hidden layer
        # -> size ((n_hidden, n_features) . (n_features, n_samples))
        # = (n_hidden, n_samples) + (n_hidden, 1)
        self.z1 = np.dot(self.w1, X) + self.b1

        # -> size (n_hidden, n_samples)
        self.a1 = np.maximum(0, self.z1)

        # output layer
        # -> size ((n_outputs, n_hidden) . (n_hidden, n_samples))
        # = (n_outputs, n_samples) + (n_outputs, 1)
        self.z2 = np.dot(self.w2, self.a1) + self.b2
        
        # -> size (n_outputs, n_samples)
        self.y = self.z2

        
        
    def backward(self, X, y_true):
        n_samples = X.shape[1]
        self.loss = np.sum((self.y - y_true)**2) / n_samples # mse loss

        # since we compute loss over each output feature y_i, we can either iterate over each y_i,
        # or vectorize the chain-rule calculations. Hence we use dot product instead of *

        # dL/dy -> size (n_outputss, 1)
        dy = 2 * (self.y - y_true) / n_samples # derivative of loss with respect to y

        # dL/dz2 -> dL/dy * dy/dz2 
        dz2 = 1 * dy # -> size (n_outputs, n_samples)

        # dL/dw2 -> dL/dy * dy/dz2 * dz2/dw2 
        dw2 = np.dot(dz2, self.a1.T) # -> size ((n_outputs, n_samples) . (n_samples, n_hidden)) = (n_outputs, n_hidden)

        # dL/wb2 -> dL/dy * dy/dz2 * dz2/db2
        db2 = np.sum(1 * dz2, axis=1, keepdims=True) # -> size (n_outputs, 1)

        # hidden layer
        # dL/da1 -> dL/dy * dy/dz2 * dz2/da1
        da1 = np.dot(self.w2.T, dz2) # size -> (n_hidden, n_outputs) . (n_outputs, n_samples) = (n_hidden, n_samples)

        # dL/dz1 -> dL/dy * dy/dz2 * dz2/da1 * da1/dz1
        dz1 = 1 * da1 # size -> (n_hidden, n_samples)

        # dL/dw1 -> dL/dy * dy/dz2 * dz2/da1 * da1/dz1 * dz1/dw1
        dw1 = np.dot(dz1, X.T) # size -> (n_hidden, n_samples) . (n_samples, n_features) = (n_hidden, n_features)

        # dL/db1 -> dL/dy * dy/dz2 * dz2/da1 * da1/dz1 * dz1/db1
        db1 = np.sum(1 * dz1, axis=1, keepdims=True) # size -> (n_hidden, 1)
        
        # update weight vectors (learning rate = 0.01)
        self.w2 -= 0.01 * dw2
        self.b2 -= 0.01 * db2
        self.w1 -= 0.01 * dw1
        self.b1 -= 0.01 * db1
    
    def train(self, X, y_true, epochs=1000):
        loss_history = []
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y_true)
            loss_history.append([epoch, self.loss])
            if epoch % 1 == 0:
                print(f'Epoch {epoch}, Loss: {np.mean(self.loss)}')
        return loss_history

if __name__ == "__main__":
    # X is array of 100 points between 0 and 9
    # Y_true is a sin function of X
    n_features = 20
    n_samples = 100
    n_hidden = 50
    n_epochs = 50

    X = np.array(10*np.random.rand(n_features, n_samples))

    # train split
    X_train = X[:, :int(n_samples*0.8)]
    X_test = X[:, int(n_samples*0.8):]

    y_true = np.sin(X_train)
    y_test = np.sin(X_test)
    
    nn = NeuralNet(n_features=n_features, n_hidden=n_hidden, n_outputs=n_features)
    loss_history = nn.train(X_train, y_true, epochs = n_epochs)

    # test split
    nn.forward(X_test)
    y_pred = nn.y

    # flatten to 1D
    x_test_flat = X_test.flatten()
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    plt.figure()
    plt.scatter(x_test_flat, y_test_flat, label='True')
    plt.scatter(x_test_flat, y_pred_flat, label = 'Pred')
    plt.xlabel('X_test value')
    plt.ylabel('Network output')
    plt.legend()
    plt.show()