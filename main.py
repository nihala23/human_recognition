import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split

% matplotlib
inline
# Setting the style and mounting the drive.
sns.set(style="whitegrid", palette="muted", font_scale=1.5)
RANDOM_SEED = 42

from google.colab import drive

drive.mount('/content/drive')
from google.colab import files

uploaded = files.upload()
from google.colab import files

uploaded = files.upload()


def create_LSTM_model(inputs):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_features, N_hidden_units])),
        'output': tf.Variable(tf.random_normal([N_hidden_units, N_classes]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_hidden_units], mean=0.1)),
        'output': tf.Variable(tf.random_normal([N_classes]))
    }
    X = tf.transpose(inputs, [1, 0, 2])
    X = tf.reshape(X, [-1, N_features])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden = tf.split(hidden, N_time_steps, 0)

    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(
        N_hidden_units, forget_bias=1.0) for _ in range(2)]
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers,
                                           hidden, dtype=tf.float32)

    lstm_last_output = outputs[-1]
    return tf.matmul(lstm_last_output, W['output']) + biases['output'

    from __future__ import print_function
    from matplotlib import pyplot as plt
    % matplotlib
    inline
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats
    from IPython.display import display, HTML

    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn import preprocessing

    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Reshape
    from keras.layers import Conv2D, MaxPooling2D
    from keras.utils import np_utils

    Setting
    few
    parameters
    which
    are
    standard
    upfront
    pd.options.display.float_format = '{:.1f}'.format
    sns.set()  # Look and feel of the default seaborn
    plt.style.use('ggplot')
    print('keras version ', keras.__version__)
    # Similar labels will be used again throughout the whole program
    LABELS = ['Downstairs',
              'Jogging',
              'Sitting',
              'Standing',
              'Upstairs',
              'Walking']
    # The number of mathods within a single time segment
    TIME_PERIODS = 80
    # Setting few parameters which are standard upfront
    pd.options.display.float_format = '{:.1f}'.format
    sns.set()  # Look and feel of the default seaborn
    plt.style.use('ggplot')
    print('keras version ', keras.__version__)
    # Similar labels will be used again throughout the whole program
    LABELS = ['Downstairs',
              'Jogging',
              'Sitting',
              'Standing',
              'Upstairs',
              'Walking']
    # The number of mathods within a single time segment
    TIME_PERIODS = 80

    Reading
    the
    dataset
    "Testing_set.csv".
    df = pd.read_csv('Testing_set.csv')

    # Reading the dataset "Training_set.csv".
    df = pd.read_csv('Training_set.csv')

    df.shape

    df.tail()
    # Importing of the packages and summary of the models and addition of the models.
    from keras.callbacks import EarlyStopping
    from tensorflow import keras

    early_stop = EarlyStopping(monitor='loss', patience=2)
    model = keras.Sequential()

    model.add(keras.layers.Dense(128, activation='relu', input_shape=(129,)))

    model.add(keras.layers.Dense(256, activation='relu'))

    model.add(keras.layers.Dense(128, activation='relu'))

    model.add(keras.layers.Dense(256, activation='relu'))

    model.add(keras.layers.Dense(128, activation='relu'))

    model.add(keras.layers.Dense(64, activation='relu'))

    model.add(keras.layers.Dense(128, activation='relu'))

    model.add(keras.layers.Dense(256, activation='relu'))

    model.add(keras.layers.Dense(128, activation='relu'))

    model.add(keras.layers.Dense(64, activation='relu'))

    model.add(keras.layers.Dense(13, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()


def main():
    # observations / data
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    # estimating coefficients
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plot_regression_line(x, y, b)


if __name__ == "__main__":
    main()
mport
numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

# Get X and y for training data
X_train = training_data.drop(columns=['Activity', 'subject'])
y_train = training_data["Activity"]

# Get X and y for testing data
y_test = testing_data['Activity']
X_test = testing_data.drop(columns=['Activity', 'subject'])

Acc = 0
Gyro = 0
other = 0

for value in X_train.columns:
    if "Acc" in str(value):
        Acc += 1
    elif "Gyro" in str(value):
        Gyro += 1
    else:
        other += 1

plt.figure(figsize=(12, 8))
plt.bar(['Accelerometer', 'Gyroscope', 'Others'], [Acc, Gyro, other], color=('r', 'g', 'b'))


