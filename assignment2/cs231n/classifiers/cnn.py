import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from cs231n.classifiers.fc_net import*


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # pass
    self.params['W1'] =  np.random.normal(0,weight_scale , (num_filters, input_dim[0], filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)

    relu_width = input_dim[1]/2
    relu_height = input_dim[2]/2
    flatteded_size = relu_width*relu_height*num_filters



    self.params['W2'] = np.random.normal(0,weight_scale ,( flatteded_size, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = np.random.normal(0,weight_scale ,(hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # pass
    conv_relu_pool_fwd, cache_conv_relu_pool_fwd =  conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    affine_relu_fwd, cache_affine_relu_fwd = affine_relu_forward(conv_relu_pool_fwd, W2, b2)
    affine_fwd, cache_affine_fwd = affine_forward(affine_relu_fwd, W3, b3)

    scores = affine_fwd
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # pass
    loss, dout = softmax_loss(scores, y) 
    loss += 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3) ) 

    dhigher1, grads['W3'], grads['b3'] = affine_backward(dout, cache_affine_fwd)
    grads['W3'] = grads['W3'] + self.reg * W3

    grad_higher2, grads['W2'] , grads['b2'] = affine_relu_backward(dhigher1, cache_affine_relu_fwd)
    grads['W2'] = grads['W2']+ self.reg * W2
    

    dx, grads['W1'] , grads['b1'] = conv_relu_pool_backward(grad_higher2, cache_conv_relu_pool_fwd)
    grads['W1']= grads['W1']+ self.reg * W1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass


def conv_batchnorm_relu_pool_forward(x, w, b, conv_param, gamma, pool_param, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """

  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  bn, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(bn)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, bn_cache, relu_cache, pool_cache)


  return out, cache


def conv_batchnorm_relu_pool_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dxbn, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
  dx, dw, db = conv_backward_fast(dxbn, conv_cache)
  return dx, dw, db , dgamma, dbeta


class MultiLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32,64], filter_size=[3,3],
               dropout=0, use_batchnorm=True,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, seed = None):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_layers = 1 + len(num_filters) + 1
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # pass
    # layer 1 conv params
    self.params['W1'] =  np.random.normal(0,weight_scale , (num_filters[0], input_dim[0], filter_size[0], filter_size[0]))
    self.params['b1'] = np.zeros(num_filters[0])
    # after this size will be (num_filters[0], H, W)
    # layer 1 batch normalization
    self.params['gamma' + str(1)] = np.random.randn(num_filters[0])
    self.params['beta' + str(1)] = np.random.randn(num_filters[0])
    # After this we will apply relu - no parameters are required for this
    # Next we will use pool - will use 2*2 pool  (num_filters[0], H/2, W/2)

    # Repeat conv-spatialbatchnorm-relu-pool one more time
    relu_width_layer1 = input_dim[1]/2
    relu_height_layer1 = input_dim[2]/2

    self.params['W2'] =  np.random.normal(0,weight_scale , (num_filters[1], num_filters[0], filter_size[1], filter_size[1]))
    self.params['b2'] = np.zeros(num_filters[1])
    # layer 2 batch normalization
    self.params['gamma' + str(2)] = np.random.randn(num_filters[1])
    self.params['beta' + str(2)] = np.random.randn(num_filters[1])

    relu_width_layer2 = relu_width_layer1/2
    relu_height_layer2 = relu_height_layer1/2

    # now fully connected layers

    flatteded_size = relu_width_layer2*relu_height_layer2*num_filters[1]


    self.params['W3'] = np.random.normal(0,weight_scale ,( flatteded_size, hidden_dim))
    self.params['b3'] = np.zeros(hidden_dim)
    self.params['gamma' + str(3)] = np.random.randn(hidden_dim)
    self.params['beta' + str(3)] = np.random.randn(hidden_dim)



    # softmax layer
    self.params['W4'] = np.random.normal(0,weight_scale ,(hidden_dim, num_classes))
    self.params['b4'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
        
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    mode = 'test' if y is None else 'train'

    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode  

    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode
   # print self.dropout_param

    W1, b1, gamma1, beta1 = self.params['W1'], self.params['b1'],self.params['gamma' + str(1)], self.params['beta' + str(1)]
    W2, b2, gamma2, beta2 = self.params['W2'], self.params['b2'],self.params['gamma' + str(2)], self.params['beta' + str(2)]
    W3, b3, gamma3, beta3 = self.params['W3'], self.params['b3'],self.params['gamma' + str(3)], self.params['beta' + str(3)]
    W4, b4 = self.params['W4'], self.params['b4']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size1 = W1.shape[2]
    conv_param1 = {'stride': 1, 'pad': (filter_size1 - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param1 = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


    # pass conv_param to the forward pass for the convolutional layer
    filter_size2 = W2.shape[2]
    conv_param2 = {'stride': 1, 'pad': (filter_size2 - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param2 = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # pass
    conv_batchnorm_relu_pool_forward_1, cache_conv_batchnorm_relu_pool_forward_1 =  conv_batchnorm_relu_pool_forward(X, 
        W1, b1, conv_param1, gamma1, pool_param1, beta1, self.bn_params[0])
    conv_batchnorm_relu_pool_forward_2, cache_conv_batchnorm_relu_pool_forward_2 =  conv_batchnorm_relu_pool_forward(conv_batchnorm_relu_pool_forward_1, 
        W2, b2, conv_param2, gamma2, pool_param2, beta2, self.bn_params[1])

    hidden_1, hidden_1_cache = affine_batchnorm_relu_dropout_forward(conv_batchnorm_relu_pool_forward_2, 
                                W3, b3,gamma3, beta3,self.bn_params[2], self.dropout_param)
    affine_fwd, cache_affine_fwd = affine_forward(hidden_1, W4, b4)

    scores = affine_fwd
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # pass
    loss, dout = softmax_loss(scores, y) 
    loss += 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3) + np.sum(W4*W4)) 

    dhigher1, grads['W4'], grads['b4'] = affine_backward(dout, cache_affine_fwd)
    grads['W4'] = grads['W4'] + self.reg * W4

    grad_higher2, grads['W3'] , grads['b3'], grads['gamma3'], grads['beta3'] = affine_batchnorm_relu_dropout_backward(dhigher1, hidden_1_cache)
    grads['W3'] = grads['W3']+ self.reg * W3

    grad_higher3, grads['W2'] , grads['b2'], grads['gamma2'], grads['beta2'] = conv_batchnorm_relu_pool_backward(grad_higher2, cache_conv_batchnorm_relu_pool_forward_2)
    grads['W2']= grads['W2']+ self.reg * W2

    dX, grads['W1'] , grads['b1'], grads['gamma1'], grads['beta1'] = conv_batchnorm_relu_pool_backward(grad_higher3, cache_conv_batchnorm_relu_pool_forward_1)
    grads['W1']= grads['W1']+ self.reg * W1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass




