import numpy as np
import scipy.io
from nn import *

import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

# ex1 = train_x[5000]
# ex1=ex1.reshape(32,32)
# plt.imshow(ex1)
# plt.show()


max_iters = 50
# pick a batch size, learning rate
batch_size = 50
learning_rate = 1e-2
in_size = train_x.shape[1]
hidden_size = 64
out_size = train_y.shape[1]
# Shuffle
batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)
# initialize layers here
params = {}
initialize_weights(in_size,hidden_size,params,'layer1')
initialize_weights(hidden_size,out_size,params,'output')


# with default settings, you should get loss < 150 and accuracy > 80%
train_loss_log, train_acc_log = [],[]
valid_loss_log, valid_acc_log = [],[]
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        pass
        # forward
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        # loss
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc
        # backward
        delta1 = probs-yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)
        # apply gradient
        params['Wlayer1'] -= params['grad_Wlayer1']*learning_rate#/xb.shape[0]
        params['blayer1'] -= params['grad_blayer1']*learning_rate#/xb.shape[0]
        params['Woutput'] -= params['grad_Woutput']*learning_rate#/xb.shape[0]
        params['boutput'] -= params['grad_boutput']*learning_rate#/xb.shape[0]

    total_acc /= len(batches)
    # Logging - training
    train_loss_log.append(total_loss)
    train_acc_log.append(total_acc)
    # Logging - validation
    h1 = forward(valid_x,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
    valid_loss_log.append(valid_loss)
    valid_acc_log.append(valid_acc)
    # Display status
    if itr % 2 == 0:
        print("itr (test): {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
        print("itr (vald): {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,valid_loss,valid_acc))

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####
##########################

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()