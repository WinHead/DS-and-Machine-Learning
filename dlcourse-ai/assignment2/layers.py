import numpy as np

def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    preds = (predictions.T - np.max(predictions, axis=1)).T
    exps = np.exp(preds)
    
    return (exps.T / exps.sum(axis=1)).T


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    return - np.log(probs[np.arange(len(probs)), target_index]).mean()

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.power(W, 2).sum()
    grad = reg_strength * 2 * W
    
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    # считаем Вероятности
    probs = softmax(preds)
    # Считаем loss
    loss = cross_entropy_loss(probs, target_index).mean()
    
    dprediction = np.zeros_like(preds)
    dprediction[np.arange(preds.shape[0]), target_index] = 1
    dprediction = - (dprediction - softmax(preds)) / dprediction.shape[0]
    
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        # Реализуем Relu - возвращаем 0, если значение меньше нуля.
        output = np.maximum(0, X)
        # Запоминаем наше преобразование
        self.is_positive = output.astype(bool)
        
        return output

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        
        # Обратное распространение градиента ошибки - раскручиваем как производную сложной функции
        # Берем передаваемое значение и умножаем на локальную производную (градиент)
        # Если X был больше нуля, производная = 1; домнажаем вход функции на 1
        # Если нет, то производная = 0; домножаем вход функции на 0
        d_result = self.is_positive.astype(int) * d_out
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        
        # Сохраним значение X
        self.X = X
        
        # Вернем результат матричного умножения иксов на веса + смещение
        return np.dot(X, self.W.value) + self.B.value
        

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        # Рассчитываем градиент для X
        d_input = np.dot(d_out, self.W.value.T)
        # Накапливаем градиент для W
        self.W.grad += np.dot(self.X.T, d_out)
        # Накапливаем градиент для B
        self.B.grad += 2 * np.mean(d_out, axis=0)
        

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
