import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        
        # Сохраняем число целевых классов
        self.n_output_classes = n_output_classes
        
        image_width, image_height, n_channels = input_shape
        
        # Инициализируем слои
        self.layers = [ConvolutionalLayer(n_channels, conv1_channels, 3, 1),    # На выходе: (32:32:conv1_channels)
                      ReLULayer(),
                      MaxPoolingLayer(4, 4),                                    # На выходе: (8:8:conv1_channels)
                      ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1), # На выходе: (8:8:conv2_channels)
                      ReLULayer(),
                      MaxPoolingLayer(4, 4),                                    # На выходе: (2:2:conv2_channels)
                      Flattener(),
                      FullyConnectedLayer(2*2*conv2_channels, n_output_classes)]
        
        
        
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # Обнуляем градиенты с предыдущего запуска
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
            
        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        
        # Предсказываем
        prediction = X
        for layer in self.layers:
            prediction = layer.forward(prediction)
            
        loss, grad = softmax_with_cross_entropy(prediction, y)
        
        
        # Накапливаем градиенты
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        
        pred = np.zeros(X.shape[0], np.int)

        out = X.copy()
        
        for layer in self.layers:
            out = layer.forward(out)
                
        return out.argmax(axis=1)

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        
         # Для каждого слоя
        for layer_num in range(len(self.layers)):
            # Для каждого параметра в этом слое
            for p in self.layers[layer_num].params().keys():
                # Добавляем в наш словарь этот параметр
                result[p + str(layer_num)] =  self.layers[layer_num].params()[p]

        return result
