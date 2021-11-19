import numpy as np
import scipy.io
from nn import *
from collections import Counter
from util import *

import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
in_size = 1024
hidden_size = 32
out_size = 1024
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(in_size,hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'layer2')
initialize_weights(hidden_size,hidden_size,params,'layer3')
initialize_weights(hidden_size,out_size,params,'output')

# should look like your previous training loops
train_loss_log, train_acc_log = [],[]
valid_loss_log, valid_acc_log = [],[]
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        pass
        # forward
        h1 = forward(xb,params,'layer1',relu)
        h2 = forward(h1,params,'layer2',relu)
        h3 = forward(h2,params,'layer3',relu)
        probs = forward(h3,params,'output',sigmoid)
        # loss
        loss, acc = compute_loss_and_acc(xb, probs)
        total_loss += loss
        # backward
        delta1 = -2*(xb-probs)
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2,params,'layer3',relu_deriv)
        delta4 = backwards(delta3,params,'layer2',relu_deriv)
        backwards(delta4,params,'layer1',relu_deriv)
        # apply gradient
        params['Wlayer1'] -= params['grad_Wlayer1']*learning_rate
        params['blayer1'] -= params['grad_blayer1']*learning_rate
        params['Wlayer2'] -= params['grad_Wlayer2']*learning_rate
        params['blayer2'] -= params['grad_blayer2']*learning_rate
        params['Wlayer3'] -= params['grad_Wlayer3']*learning_rate
        params['blayer3'] -= params['grad_blayer3']*learning_rate
        params['Woutput'] -= params['grad_Woutput']*learning_rate
        params['boutput'] -= params['grad_boutput']*learning_rate

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################
