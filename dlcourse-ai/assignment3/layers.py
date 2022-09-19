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
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.power(W, 2).sum()
    grad = reg_strength * 2 * W
    
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    '''
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
    '''
    # TODO copy from the previous assignment
    # считаем Вероятности
    probs = softmax(preds)
    # Считаем loss
    loss = cross_entropy_loss(probs, target_index).mean()
    
    dprediction = np.zeros_like(preds)
    dprediction[np.arange(preds.shape[0]), target_index] = 1
    dprediction = - (dprediction - softmax(preds)) / dprediction.shape[0]
    
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        # Реализуем Relu - возвращаем 0, если значение меньше нуля.
        output = np.maximum(0, X)
        # Запоминаем наше преобразование
        self.is_positive = output.astype(bool)
        
        return output

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        # Обратное распространение градиента ошибки - раскручиваем как производную сложной функции
        # Берем передаваемое значение и умножаем на локальную производную (градиент)
        # Если X был больше нуля, производная = 1; домнажаем вход функции на 1
        # Если нет, то производная = 0; домножаем вход функции на 0
        d_result = self.is_positive.astype(int) * d_out
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        
        # Сохраним значение X
        self.X = X
        
        # Вернем результат матричного умножения иксов на веса + смещение
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        # Рассчитываем градиент для X
        d_input = np.dot(d_out, self.W.value.T)
        # Накапливаем градиент для W
        self.W.grad += np.dot(self.X.T, d_out)
        # Накапливаем градиент для B
        self.B.grad += 2 * np.mean(d_out, axis=0)
        

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        
        # Формируем X, вокруг которого будут пиксели со значением 0.
        X_padding = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, self.in_channels))
        # Заполняем реальным занчением пикселей
        X_padding[:, self.padding:self.padding + height, self.padding:self.padding + width, :] = X
        
        # Запоминаем X_padding и X
        self.X_padding = X_padding
        self.X = X
        
        # Добавляем еще одно измерение
        X_padding = X_padding[:, :, :, :, np.newaxis]

        W = self.W.value[np.newaxis, :, :, :, :]
        
        # Указываем длину и ширину выхоа
        out_height = height - self.filter_size + 2*self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1
        
        # Создаем нулевую матрицу формы выхода
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        # В цикле применяем наш "фильтр" к каждому фрагменту в картинке
        for y in range(out_height):
            for x in range(out_width):
                # Получаем знаечние одного фрагмента
                X_slice = X_padding[:, y:y + self.filter_size, x:x + self.filter_size, :, :]
                # формируем соответствующий выход
                out[:, y, x, :] = np.sum(X_slice * self.W.value, axis=(1, 2, 3)) + self.B.value
                
        return out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Формируем матрицу для градиентов
        X_grad = np.zeros_like(self.X_padding)
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                
                # Воссоздаем X
                X_slice = self.X_padding[:, y:y + self.filter_size, x:x + self.filter_size, :, np.newaxis]
                
                grad = d_out[:, y, x, np.newaxis, np.newaxis, np.newaxis, :]
                
                self.W.grad += np.sum(grad * X_slice, axis=0)

                X_grad[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.sum(self.W.value * grad, axis=-1)

        self.B.grad += np.sum(d_out, axis=(0, 1, 2))

        return X_grad[:, self.padding:self.padding + height, self.padding:self.padding + width, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        
        # Запоминаем X
        self.X = X
        
        # Считаем, сколько раз наше окошко будет двигаться вдоль и поперек картинки
        self.out_height = int( (height - self.pool_size) / self.stride) + 1
        self.out_width = int( (width - self.pool_size) / self.stride) + 1
        
        # Инициализируем out нужной размерности
        out = np.zeros((batch_size, self.out_height, self.out_width, channels))
        
        # Теперь в цикле производим рассчет max pool для каждой ячейки
        for y in range(self.out_height):
            for x in range(self.out_width):
                # Ищем максимум для фрагмента X по длине и ширине
                out[:,y,x,:] = np.amax( X[:, y: y+self.pool_size, x: x+self.pool_size,:], axis=(1,2) )
        
        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        
        # Инициализация out соотвествующего размера
        out = np.zeros_like(self.X)

        # Перебираем тот же цикл и выполняем накопление градиента только максимальному элементу
        for y in range(self.out_height):
            for x in range(self.out_width):
                
                # Из X получаем значение фрагмента
                X_slice = self.X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                
                # Расширяем градиент до нужной нам размерности
                grad = d_out[:, y, x, :][:, np.newaxis, np.newaxis, :]
                
                # Создаем максу для фрагмента чтобы знать, какому именно элементу передать градиент
                # np.amax( X_slice, (1, 2) даст матрицу меньшей размерности, поэтому увеличиваем её
                mask = ( X_slice == np.amax( X_slice, (1, 2) )[:, np.newaxis, np.newaxis, :] )
                
                # Накапливаем градиент соответствующих элементов
                out[:, y: y+self.pool_size, x: x+self.pool_size,:] += grad * mask
        
        return out

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        
        # Запомним размер X
        self.X_shape = X.shape
        
        return X.reshape(batch_size, height*width*channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
