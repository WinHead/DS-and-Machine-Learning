import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        
        # Создаем три слоя, один из которых релу
        self.layers = [FullyConnectedLayer(n_input, hidden_layer_size),
                       ReLULayer(),
                       FullyConnectedLayer(hidden_layer_size, n_output)]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        
        # Осуществляем прямой проход по сети, одновременно Зануляя все параметры
        
        # Устанавливаем начальные иксы
        val = X
        # Для каждого слоя
        for lay_num in range(len(self.layers)):
            
            # Считаем выход на очередном слое
            val = self.layers[lay_num].forward(val)
            
            # Зануляем градиенты каждого из параметров
            for p in self.layers[lay_num].params().keys():
                self.layers[lay_num].params()[p].grad = np.zeros_like(self.layers[lay_num].params()[p].value)
    
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # Находим ошибку и градиент на последнем слое
        loss, grad = softmax_with_cross_entropy(val, y)
        
        # Пробегаем слои в обратном порядке
        for lay_num in reversed(range(len(self.layers))):
            
            # Рассчитываем градиент для очередного слоя, 
            # передавая ему значения градиента с прошлого шага
            grad = self.layers[lay_num].backward(grad)
        
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        
        # Добавляем решуляризаци ко всем параметрам
        for lay_num in range(len(self.layers)):
            for p in self.layers[lay_num].params().keys():
                l2_val, l2_grad = l2_regularization(self.layers[lay_num].params()[p].value, self.reg)
                loss += l2_val
                self.layers[lay_num].params()[p].grad += l2_grad
                

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        output = X.copy()
        
        for layer_num in range(len(self.layers)):
            output = self.layers[layer_num].forward(output)
        
        pred = output.argmax(axis=1)
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        # Для каждого слоя
        for layer_num in range(len(self.layers)):
            
            # Для каждого параметра в этом слое
            for p in self.layers[layer_num].params().keys():
                
                # Добавляем в наш словарь этот параметр
                result[p + str(layer_num)] =  self.layers[layer_num].params()[p]

        return result
