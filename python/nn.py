import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None
    # uniform distribution [a,b]: var = (b-a)^2/12
    # want var = 2/sum(N), so (b-a)^2 = 24/sum(N)
    # if a = -b, so range is [-b,b], get b = sqrt(6/sum(N))
    b = (6/(in_size+out_size))**(1/2)
    W = np.random.uniform(-b,b,(in_size, out_size))
    b = np.zeros((out_size,))

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None
    res = 1/(1+np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    pre_act = X@W+b
    post_act = activation(pre_act)


    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None
    x_shift = (x-np.expand_dims(np.max(x,axis=1),axis=1))
    x_exp = np.exp(x_shift)
    res = x_exp/np.expand_dims(np.sum(x_exp,axis=1),axis=1)

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    loss = -np.trace(y.T@np.log(probs))
    prediction = np.argmax(probs,axis=1)
    answer = np.argmax(y,axis=1)
    acc = np.count_nonzero(answer==prediction)/len(answer)


    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    delta2 = delta*activation_deriv(post_act)
    grad_W = np.transpose(X)@delta2
    grad_b = np.sum(delta2,axis=0)
    grad_X = delta2@np.transpose(W)

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    # Create random seq of numbers
    rand_indx = np.arange(x.shape[0])
    np.random.shuffle(rand_indx)
    # Produce most batches
    for i in range(0, x.shape[0]//batch_size):
        cur_indx = rand_indx[i*batch_size:(i+1)*batch_size]
        batch1_x = x[cur_indx]
        batch1_y = y[cur_indx]
        batches.append((batch1_x, batch1_y))
    # Last batch
    if((i+1)*batch_size < x.shape[0]):
        cur_indx = rand_indx[(i+1)*batch_size:]
        batch1_x = x[cur_indx]
        batch1_y = y[cur_indx]
        batches.append((batch1_x, batch1_y))
    return batches
