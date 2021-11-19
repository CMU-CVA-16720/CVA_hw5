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
# max_iters = 1
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
# Initialize momentums
params['m_Wlayer1'] = np.zeros(params['Wlayer1'].shape)
params['m_blayer1'] = np.zeros(params['blayer1'].shape)
params['m_Wlayer2'] = np.zeros(params['Wlayer2'].shape)
params['m_blayer2'] = np.zeros(params['blayer2'].shape)
params['m_Wlayer3'] = np.zeros(params['Wlayer3'].shape)
params['m_blayer3'] = np.zeros(params['blayer3'].shape)
params['m_Woutput'] = np.zeros(params['Woutput'].shape)
params['m_boutput'] = np.zeros(params['boutput'].shape)


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
        # update momentum
        params['m_Wlayer1'] = 0.9 * params['m_Wlayer1'] - params['grad_Wlayer1']*learning_rate
        params['m_blayer1'] = 0.9 * params['m_blayer1'] - params['grad_blayer1']*learning_rate
        params['m_Wlayer2'] = 0.9 * params['m_Wlayer2'] - params['grad_Wlayer2']*learning_rate
        params['m_blayer2'] = 0.9 * params['m_blayer2'] - params['grad_blayer2']*learning_rate
        params['m_Wlayer3'] = 0.9 * params['m_Wlayer3'] - params['grad_Wlayer3']*learning_rate
        params['m_blayer3'] = 0.9 * params['m_blayer3'] - params['grad_blayer3']*learning_rate
        params['m_Woutput'] = 0.9 * params['m_Woutput'] - params['grad_Woutput']*learning_rate
        params['m_boutput'] = 0.9 * params['m_boutput'] - params['grad_boutput']*learning_rate
        # update parameters
        params['Wlayer1'] += params['m_Wlayer1']
        params['blayer1'] += params['m_blayer1']
        params['Wlayer2'] += params['m_Wlayer2']
        params['blayer2'] += params['m_blayer2']
        params['Wlayer3'] += params['m_Wlayer3']
        params['blayer3'] += params['m_blayer3']
        params['Woutput'] += params['m_Woutput']
        params['boutput'] += params['m_boutput']
    # logging
    train_loss_log.append(total_loss)
    # Printing
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
# Visualize results
if False:
    from mpl_toolkits.axes_grid1 import ImageGrid
    for i in range(0,xb.shape[0]):
        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(1, 2),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )
        for indx,ax in enumerate(grid):
            # Iterating over the grid returns the Axes.
            if(indx == 0):
                img = np.reshape(xb[i,:],(32,32))
            else:
                img = np.reshape(probs[i,:],(32,32))
            ax.imshow(img)
        plt.show()
# Graphs
if False:
    ax = plt.axes()
    ax.plot(np.arange(0,max_iters), train_loss_log, color='red') # training loss
    plt.xlim(0, max_iters)
    plt.title("Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

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
