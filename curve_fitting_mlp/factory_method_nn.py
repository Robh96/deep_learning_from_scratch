import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class LinearLayer(ABC):
    # defines the common behaviours all linear layers must have
    def __init__(self, input_dims, output_dims):
        # initialise weights and biases for single layer
        # Xavier initialisation (accounts for changing distribution throughout layers)
        self.w = np.random.randn(input_dims, output_dims) * np.sqrt(1 / input_dims) # each hidden neuron has n weights (equal to number of hidden neurons in prev layer)
        self.b = np.random.randn(output_dims, 1)

    def forward(self, input):
        pass

    def backward(self, gradient):
        pass

    def update(self):
        pass

class SimpleLinearLayer(LinearLayer):
    def __init__(self, input_dims, output_dims):
        super().__init__(input_dims, output_dims)

    def forward(self, input):
        self.input = input
        self.z = np.dot(self.w.T, self.input) + self.b
        return self.z
    
    def backward(self, gradient):
        self.dw = np.dot(self.input, gradient.T)
        self.db = np.sum(gradient * 1, axis=1, keepdims=True)
        dinput = np.dot(self.w, gradient)
        return dinput

    def update(self, learning_rate):
        self.w = self.w - learning_rate * self.dw
        self.b = self.b - learning_rate * self.db


class LinearLayerWithReLU(LinearLayer):
    def __init__(self, input_dims, output_dims):
        # this inherits the __init__ method from the abstract class, which initialises the weights and biases.
        super().__init__(input_dims, output_dims)
        # reinitialise weights to account for He initialisation
        self.w = np.random.randn(input_dims, output_dims) * np.sqrt(2 / input_dims)

    def forward(self, input):
        # compute forward pass through single layer then add on ReLU to result
        # w.T has dims (output_dims, input_dims), input has dims (input_dims, batch_size)
        self.input = input 
        self.a = np.dot(self.w.T, self.input) + self.b
        z = np.maximum(0, self.a) # ReLU
        return z

    def backward(self, gradient):
        # backprop through the single layer
        # gradient is dL/dz. we need to return dL/dw, dL/db, and dL/dinput
        # first step is getting dL/da, which is equal to dL/dz * da/da
        da = gradient * (self.a > 0)
        # then we need da/dw, da/db, da/dinput
        self.dw = np.dot(self.input, da.T) # dw has dims (input_dims, output_dims), da has dims (output_dims, batch_size), input has dims (input_dims, batch_size)
        self.db = np.sum(da * 1, axis=1, keepdims=True)
        dinput = np.dot(self.w, da)
        return dinput

    def update(self, learning_rate):
        # apply weights update using simple SDG with fixed learning rate
        self.w = self.w - learning_rate * self.dw
        self.b = self.b - learning_rate * self.db
    

class LayerFactory(ABC):
    # responsibility is to declare the factory method

    @abstractmethod
    def create_layer(input_dims, output_dims):
        pass
    
class ReLULayerFactory(LayerFactory):
    def create_layer(self, input_dims, output_dims):
        return LinearLayerWithReLU(input_dims, output_dims)
    
class SimpleLayerFactory(LayerFactory):
    def create_layer(self, input_dims, output_dims):
        return SimpleLinearLayer(input_dims, output_dims)


class SequentialNetwork:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, input_data):
        # for loop through each layer in self.layers
        for layer in self.layers:
            input_data = layer.forward(input_data)
        
        y_predicted = input_data
        return y_predicted
    
    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
            
                
class MSELoss:
    def calculate(self, y_predicted, y_true):
        loss = np.mean((y_predicted - y_true) ** 2)
        return loss

    def derivative(self, y_predicted, y_true):
        gradient = 2 * (y_predicted - y_true) / y_predicted.shape[1]
        return gradient
    
if __name__ == "__main__":
    # Hyperparameters
    n_epochs = 1000
    learning_rate = 1e-3
    batch_size = 200

    # We want the network to learn a sin(X) function
    data = np.random.uniform(low=0, high=2*np.pi, size=10000)

    # Pre-process data
    # data normalisation (subtract mean, divide by standard deviation. i.e. z-score)
    # mean_data = np.mean(data)
    # std_data = np.std(data)
    # data = (data-mean_data) / std_data

    X_train = data[:int(0.8*len(data))] # single input feature
    X_test = data[int(0.8*len(data)):]
    

    y_true = np.sin(X_train)
    y_test = np.sin(X_test)

    # Create a two layer network with ReLU activation, MSE loss, and stochastic gradient descent
    relu_layer = ReLULayerFactory()
    layer1 = relu_layer.create_layer(input_dims=1, output_dims=50)
    layer2 = relu_layer.create_layer(input_dims=50, output_dims=50)
    layer3 = relu_layer.create_layer(input_dims=50, output_dims=50)
    linear_layer = SimpleLayerFactory()
    layer4 = linear_layer.create_layer(input_dims=50, output_dims=1)

    neural_net = SequentialNetwork()
    neural_net.add_layer(layer1)
    neural_net.add_layer(layer2)
    neural_net.add_layer(layer3)
    neural_net.add_layer(layer4)
    loss_fn = MSELoss()

    # pre-allocate loss histories
    train_loss_history = []
    val_loss_history = []
    
    # Training loop
    for epoch in range(n_epochs):
        
        total_train_loss = 0
        n_train_samples = 0
        
        # shuffle X_train and y_true in same way
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_true = y_true[perm]
        
        # Separate data into batches
        for i in range(len(X_train) // batch_size):
            X_batch = X_train[i:i+batch_size].reshape(1, -1)
            y_batch = y_true[i:i+batch_size].reshape(1, -1)

            # forward pass
            y_pred = neural_net.forward(X_batch)

            # calculate loss
            batch_loss = loss_fn.calculate(y_predicted=y_pred, y_true=y_batch)
            gradient = loss_fn.derivative(y_predicted=y_pred, y_true=y_batch)

            # accumulate training loss (corrected for uneven batch sizes)
            total_train_loss += batch_loss * X_batch.shape[1]
            n_train_samples += X_batch.shape[1]

            # backward pass
            neural_net.backward(gradient)

            # update
            neural_net.update(learning_rate=learning_rate)

        mean_train_loss = total_train_loss / n_train_samples


        # validation loss
        total_val_loss = 0
        n_val_samples = 0
        for i in range(len(X_test) // batch_size):
            val_X_batch = X_test[i:i+batch_size].reshape(1, -1)
            val_y_batch = y_test[i:i+batch_size].reshape(1, -1)
            val_y_pred = neural_net.forward(val_X_batch)
            val_batch_loss = loss_fn.calculate(y_predicted=val_y_pred, y_true=val_y_batch)

            total_val_loss += val_batch_loss * val_X_batch.shape[1]
            n_val_samples += val_X_batch.shape[1]

        mean_val_loss = total_val_loss / n_val_samples

        train_loss_history.append([epoch, mean_train_loss])
        val_loss_history.append([epoch, mean_val_loss])
        print(f"Epoch {epoch}: Train Loss = {mean_train_loss:.4f} | Val Loss = {mean_val_loss:.4f}")

        if epoch % 5 == 0:
            # test split
            y_pred = neural_net.forward(X_test.reshape(1, -1))

            # flatten to 1D
            x_test_flat = X_test
            y_test_flat = y_test
            y_pred_flat = y_pred.flatten()

            fig, (ax1, ax2) = plt.subplots(2,1)
            ax1.scatter(x_test_flat, y_test_flat, label = 'True')
            ax1.scatter(x_test_flat, y_pred_flat, label = 'Pred')
            ax1.set_xlim(0, 2*np.pi)
            ax1.set_ylim(-1.1, 1.1)
            ax1.set_xlabel('X_test value')
            ax1.set_ylabel('Network output')
            ax1.legend()
            
            epochs, train_losses = zip(*train_loss_history)
            ax2.plot(epochs, train_losses, '-', color='green', label='train')
            epochs, val_losses = zip(*val_loss_history)
            ax2.plot(epochs, val_losses, '-', color='red', label='validation')
            ax2.set_ylim(1e-5, 1)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MSE Loss')
            ax2.set_yscale("log")
            ax2.legend()
            plt.tight_layout()
            fig.savefig(f"./figs/learning_rate_1e-3/sin_curve_{epoch:0>4}")
            plt.close()