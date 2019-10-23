import keras
import numpy as np
from sklearn.model_selection import train_test_split


class Datasets:
    def mnist():
        """Returns np.ndarray of MNIST dataset.
        Returns:
            (x_train, y_train), (x_test, y_test)
                train: (60000, 784) (60000, 10)
                test:  (10000, 784) (10000, 10)
        """
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        # train: (60000, 28, 28) (60000,)
        # test:  (10000, 28, 28) (10000,)
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        # train: (60000, 784) (60000,)
        # test:  (10000, 784) (10000,)
        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        # train: (60000, 784) (60000, 10)
        # test:  (10000, 784) (10000, 10)
        return (x_train, y_train), (x_test, y_test)
