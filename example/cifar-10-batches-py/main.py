import sys, os
sys.path.insert(1, os.path.split(os.path.split(sys.path[0])[0])[0])
import cPickle as pkl
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from classes.neural_net import NeuralNetwork

def unpickle(file):
    fo = open(file, 'rb')
    dict = pkl.load(fo)
    fo.close()
    return dict

le = preprocessing.LabelEncoder()
le.classes_ = unpickle(sys.path[0] + '/batches.meta')['label_names']

train_images = None
train_labels = []

test_images = None
test_labels = []
for i in xrange(1, 6):
    data = unpickle(sys.path[0] + '/data_batch_'+str(i))
    if train_images is None:
        train_images = data['data']
    else:
        train_images = np.vstack((train_images, data['data']))

    train_labels += data['labels']

# train_images = train_images[:100]
# train_labels = train_labels[:100]

train_images = train_images.reshape(-1, 3, 32, 32)
train_images = train_images.astype(np.float128)
train_images /= 255.0

r_mean = np.average(train_images[:, 0])
g_mean = np.average(train_images[:, 1])
b_mean = np.average(train_images[:, 2])

train_images[:, 0] -= r_mean
train_images[:, 1] -= g_mean
train_images[:, 2] -= b_mean

train_images = train_images.astype(np.float32)

data = unpickle(sys.path[0] + '/test_batch')
test_images = data['data'].reshape(-1, 3, 32, 32)
test_images = test_images.astype(np.float128)
test_images /= 255.0

test_images[:, 0] -= r_mean
test_images[:, 1] -= g_mean
test_images[:, 2] -= b_mean

test_images.astype(np.float32)

test_labels = data['labels']

lr = 1e-4
dropout_percent = 0.5
l2_reg = 4e-7
learning_rate_decay = np.float32(99e-2)
batch_size = 10

cnn = NeuralNetwork(train_images.shape[1:],
                    [
                        {'type': 'conv', 'k': 16, 'u_type': 'adam', 'f': 5, 's': 1, 'p': 2},
                        {'type': 'pool'},
                        {'type': 'conv', 'k': 20, 'u_type': 'adam', 'f': 5, 's': 1, 'p': 2},
                        {'type': 'pool'},
                        {'type': 'conv', 'k': 20, 'u_type': 'adam', 'f': 5, 's': 1, 'p': 2},
                        {'type': 'pool'},
                        {'type': 'output', 'k': len(le.classes_), 'u_type': 'adam'}
                    ]
                    , lr, l2_reg=l2_reg, dropout_p=dropout_percent)

cnn.epoch_count = 0

for i in xrange(60000000):
    start = i * batch_size % len(train_images)
    end = start + batch_size

    if start == 0 and i != 0:
        cnn.epoch_count += 1
        train_images, train_labels = shuffle(train_images, train_labels)
        print '{} epoch finish. learning rate is {}'.format(str(cnn.epoch_count), str(cnn.lr))
        cnn.lr *= learning_rate_decay

        loss, acc = cnn.predict(train_images[:4000], train_labels[:4000])
        print 'training acc:{}'.format(acc)
        print 'training loss:{}'.format(loss)

        test_loss, test_acc = cnn.predict(test_images, test_labels)
        print 'test acc:{}'.format(test_acc)
        print 'test loss:{}'.format(test_loss)

    cnn.t += 1
    loss, acc = cnn.epoch(train_images[start:end], train_labels[start:end])
    # print loss
    # print acc
