from datasets import Datasets
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
import keras
import time


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class MLP:
    def __init__(self, layers, data, optimizer, hid_act='tanh',
                 out_act='softmax', dropout=None, wdecay=None,
                 loss='categorical_crossentropy', metrics=['accuracy']):
        """Builds a feedforward neural network. Uses stochastic gradient
        descent for optimization.

        Parameters:
            layers      list, number of hidden units in each layer
            data        data tuple: (x_train, y_train), (x_test, y_test)
        Optional:
            hid_act     hidden activation function
            out_act     output activation function
            dropout     dropout value
            loss        what loss function to use
            metrics     what metrics to use for evaluation
        """

        # check input
        assert len(layers) > 0

        # preprocess optimizer
        if isinstance(optimizer, str):
            if optimizer == 'sgd':
                optimizer = SGD(lr=0.01, decay=1e-6,
                                momentum=0.9, nesterov=True)
            elif optimizer == 'rms':
                optimizer = RMSprop()
            else:
                raise ValueError('Invalid optimizer')

        # preprocess data
        if isinstance(data, str):
            if data == 'mnist':
                data = Datasets.mnist()

        # save parameters
        self.hid_act = hid_act
        self.out_act = out_act
        self.loss = loss
        self._layers = layers.copy()

        # load the data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        input_dim = self.x_train.shape[1]
        output_dim = self.y_train.shape[1]

        # preprocess the layers
        if dropout:
            layers = [int(e / dropout) for e in layers]

        # create the model
        self.model = Sequential()

        # add first hidden layer
        layer = Dense(layers[0], activation=hid_act, input_dim=input_dim)
        if wdecay:
            layer.kernel_regularizer = keras.regularizers.l2(wdecay)
        self.model.add(layer)
        if dropout:
            dropout_layer = Dropout(dropout)
            self.model.add(dropout_layer)

        # add subsequent layers
        for i in range(1, len(layers)):
            layer = Dense(layers[i], activation=hid_act)
            self.model.add(layer)
            if dropout:
                dropout_layer = Dropout(dropout)
                self.model.add(dropout_layer)

        # add output layer
        output_layer = Dense(output_dim, activation=out_act)
        self.model.add(output_layer)

        # setup optimizer
        self.optimizer = optimizer

        # compile model
        self.model.compile(loss=loss,
                           optimizer=self.optimizer,
                           metrics=metrics)

    def train(self, epochs, batch_size, patience=0):
        """Train the MLP.

        Parameters:
            epochs      how many epochs to train
            batch_size  size of batches

        Returns:
            hist        history object for training losses
        """

        # time callback
        callbacks = [
            TimeHistory()
        ]

        if patience > 0:
            epochs = 1000
            callbacks.append(
                keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=patience)
            )

        # train the model
        hist = self.model.fit(self.x_train,
                              self.y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(self.x_test, self.y_test),
                              callbacks=callbacks)

        # evaluate model
        score = self.model.evaluate(self.x_test,
                                    self.y_test,
                                    batch_size=batch_size)

        # log test scores
        print('Test loss: {}\nTest acc:  {}'.format(
            score[0], score[1]))

        # extract times
        times = callbacks[0].times

        # extract history
        hist = hist.history

        # save all stats in one dictionary
        stats = {
            'epoch': [], 'time': [], 'val_loss': [],
            'val_accuracy': [], 'train_loss': [], 'train_accuracy': [],
            'comments': []
        }
        for i in range(len(times)):
            comments = '[{}]'.format(', '.join(map(str, self._layers)))
            comments += ', {}, {}, {}, {}, patience {}'.format(
                self.hid_act, self.out_act, self.loss, str(self.optimizer),
                patience
            )
            stats['epoch'] += [i + 1]
            stats['time'] += ['{:.3f}'.format(times[i])]
            stats['val_loss'] += ['{:.4f}'.format(hist['val_loss'][i])]
            stats['val_accuracy'] += ['{:.4f}'.format(hist['val_accuracy'][i])]
            stats['train_loss'] += ['{:.4f}'.format(hist['loss'][i])]
            stats['train_accuracy'] += ['{:.4f}'.format(hist['accuracy'][i])]
            stats['comments'] += [comments]

        # return statistics dictionary
        return stats
