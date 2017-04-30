import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  #pass
  N = x.shape[0]
  x_reshape = np.reshape(x,(N,-1))
  out = np.dot(x_reshape, w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  #pass
  dx = np.dot(dout, w.T)
  dx = np.reshape(dx, x.shape)

  N = x.shape[0]
  x_reshaped = np.reshape(x, (N,-1))
  dw = np.dot(x_reshaped.T, dout) # shape is (D,M)

  db = np.sum(dout, axis = 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db

def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = x*(x>0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################

  dx = dout
  dx[x<0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
 
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    # pass


    # Step 1 compute mean
    mu = np.mean(x, axis = 0) #(D,)

    # Step 2 : Compute xu = x - mu
    xmu = x - mu # (N,D)

    # Step 3: Compute xmusquare
    xmusquare = xmu**2 #(N,D)

    # Step 4 : Compute variance
    xvar = np.mean(xmusquare, axis = 0) # (D,)

    # Step 5: Compute inverse of standard deviation
    xvarinv = 1/(np.sqrt(xvar + eps)) #(D,)

    # Step 6: Compute xhat
    xhat = xvarinv*xmu #(N,D)

    # Step 7 : Compute xhat*Gamma
    xhatgamma = xhat*gamma #(N,D)

    # Step 8 : Compute xhatgamma + beta = out
    xhatgammabeta = xhatgamma + beta

    out = xhatgammabeta

    sample_mean = mu #np.mean(x ,axis = 0)
    sample_var = xvar #np.var(x, axis = 0)
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    cache = (x, mu, xmu, xmusquare, xvar, xvarinv, xhat, xhatgamma, xhatgammabeta, gamma, beta, eps)

   










    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    # pass
    x_hat = (x - running_mean)/np.sqrt(running_var + eps)
    out = gamma*x_hat + beta
    cache = None


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var
 

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
 
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
 
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
 
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  # pass  x, mu, xmu, xmusquare, xvar, xvarinv, xhat, xhatgamma, xhatgammabeta, gamma, beta
  x, mu, xmu, xmusquare, xvar, xvarinv, xhat, xhatgamma, xhatgammabeta, gamma, beta, eps = cache
  #############################################################################
  #                             END OF YOUR CODE                              #

  N, D = x.shape

  # Step 8
  dbeta = np.sum(dout, axis =0) # (D,)

  # Step 7
  dxhatgamma = dout #(N,D)

  # Step6
  dgamma = np.sum(dxhatgamma*xhat, axis = 0) #(D,)

  # Step 5
  dxhat = dxhatgamma*gamma #(N,D)

  # Step 4
  dxmu1 = dxhat*xvarinv #(N,D)

  # Step 3
  dxvarinv = np.sum(dxhat*xmu, axis = 0) #(D,)

  # Step 2
  dxvar = -.5*dxvarinv/((xvar + eps)**1.5) # (D,)

  # Step 2
  dxmusquare = (1./N)*np.ones((N,D))*dxvar #(N,D)

  # Step 1
  dxmu2 = dxmusquare*2*xmu #(N,D)

  # add the gradients for xmu
  dxmu = dxmu1 + dxmu2 #(N,D)

  # Step 0
  dx1 = dxmu # (N,D)

  # Step -1
  dmu = -np.sum(dxmu, axis = 0) #(D,)

  # Step -2
  dx2 = (1./N)*np.ones((N,D))*dmu #(N,D)

  # Step -3
  dx = dx1 + dx2 #(N,D)



  #print  dmu, dx2

  #############################################################################

  return dx, dgamma, dbeta

def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    #pass
    #mask = (np.random.rand((x.shape[0],x.shape[1]))<p)/p
    #print type(x.shape)
    
    mask = (np.random.random_sample(x.shape)<p)/p
    out = x*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    #pass
    mask = np.ones((x.shape))
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    #pass
    p = dropout_param['p']
    dx = dout*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  # pass
  stride, pad = conv_param['stride'], conv_param['pad']
  npad = ((0,0) ,(0,0), (pad, pad), (pad, pad))
  padded_image = np.pad(x, pad_width = npad, mode = 'constant', constant_values = 0)
  # We will use "valid"  convolution ??

  # w: Filter weights of shape (F, C, HH, WW)
  # output of convolution will have F activation layers
  # Compute each activation layer one by one - this is a naive implementation

  number_images = x.shape[0]
  number_activation_layers = w.shape[0]
  H_dash = 1 + (x.shape[2] + 2*pad - w.shape[2])/stride
  W_dash = 1 + (x.shape[3] + 2*pad - w.shape[3])/stride
  filter_width = w.shape[3]
  filter_height = w.shape[2]
  
  #print filter_height,filter_width 
  #print "="*50
  #print number_images, number_activation_layers, H_dash, W_dash
  convolved_layer = np.zeros((number_images, number_activation_layers, H_dash, W_dash)) # initialise 
  # output layer to appropriate size
  #print convolved_layer.shape
  # for k in range(number_images):
  #   this_example = padded_image[k,:,:,:]
  #   for j in range(number_activation_layers): # iterate over each activation layer from convolution
  #     filter_for_this_layer_activation = w[j, :, :, :]
  #     output_layer_height_index = 0
  #     output_layer_width_index = 0
  #     #print this_example.shape
  #     for y in range(0, this_example.shape[1]-stride-1, stride):
  #       output_layer_width_index = 0
  #       for x in range(0, this_example.shape[2]-stride-1, stride):
  #         #image_height_subsection = y:filter_height
  #         #image_width_subsection = x:filter_width
  #          #print (y,x)
  #         image_subsection = this_example[:,y:(y+filter_height), x:(x+filter_width)]
  #         #print image_subsection.shape, filter_for_this_layer_activation.shape
  #         #print np.sum(image_subsection*filter_for_this_layer_activation) + b[j]
  #         convolved_layer[k, j,output_layer_height_index,output_layer_width_index  ] = \
  #               np.sum(image_subsection*filter_for_this_layer_activation) + b[j]
  #         output_layer_width_index+=1
  #         #print output_layer_width_index
  #       output_layer_height_index+=1
  #       #print "output_layer_height_index", output_layer_height_index

  # out = convolved_layer
  #print out
  # Alternate implementaion using the method mentioned online
  number_filters = w.shape[0]
  filter_depth = w.shape[1]
  image_channels = filter_depth
  X_Col_number_of_rows = filter_width*filter_height*image_channels
  X_Col_number_of_columns_1 = ((padded_image.shape[2] - filter_width)/stride) + 1
  X_Col_number_of_columns_2 = ((padded_image.shape[3] - filter_height)/stride) + 1
  X_forward = np.zeros((number_images, X_Col_number_of_rows, X_Col_number_of_columns_1*X_Col_number_of_columns_2))
#  print X_forward.shape
  #print filter_height,filter_width

  for k in range(number_images):
    this_example = padded_image[k,:,:,:]
    # Apply im2col as menioned http://cs231n.github.io/convolutional-networks/
    t = 0
    for y in range(0, X_Col_number_of_columns_1):
      output_layer_width_index = 0
      for xx in range(0, X_Col_number_of_columns_2):
        image_subsection = this_example[:,y*stride: y*stride + filter_height, xx*stride: xx*stride + filter_width]
       # print "#"*50
      #  print this_example
        #print image_subsection
        #print "$"*50
     #   print np.sum(image_subsection*w[0,:,:,:]) + b[0]
     #   print image_subsection.shape, t
        X_forward[k,:,t] =  np.reshape(image_subsection, (-1,))
        t = t+1
     
  #print X_forward[0]
 # print padded_image[0]
 # print c+1
#  print "="*22
  W_row = np.reshape(w, (number_filters, -1))
 # print W_row.shape, X_forward.shape
    
  for j in range(number_images):
    dot_product = np.dot(W_row, X_forward[j, :,:]) + b.reshape(-1,1)
#    print W_row
 #   print w[j]
 #   print  X_forward[j, :,:]
 #   print dot_product.shape
   # print c+2
    convolved_layer[j,:,:, :] = np.reshape(dot_product, (number_filters, X_Col_number_of_columns_1, X_Col_number_of_columns_2 ))
  out = convolved_layer
 
#  print convolved_layer



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
 # print x.shape
  return out, cache



def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  # pass
  x, w, b, conv_param = cache

  stride, pad = conv_param['stride'], conv_param['pad']
  npad = ((0,0) ,(0,0), (pad, pad), (pad, pad))
  padded_image = np.pad(x, pad_width = npad, mode = 'constant', constant_values = 0)
  # We will use "valid"  convolution 

  # w: Filter weights of shape (F, C, HH, WW)
  # output of convolution will have F activation layers
  # Compute each activation layer one by one - this is a naive implementation

  number_images = x.shape[0]
  number_activation_layers = w.shape[0]
  H_dash = 1 + (x.shape[2] + 2*pad - w.shape[2])/stride
  W_dash = 1 + (x.shape[3] + 2*pad - w.shape[3])/stride
  filter_width = w.shape[3]
  filter_height = w.shape[2]

  convolved_layer = np.zeros((number_images, number_activation_layers, H_dash, W_dash)) # initialise 
  number_filters = w.shape[0]
  filter_depth = w.shape[1]
  image_channels = filter_depth
  X_Col_number_of_rows = filter_width*filter_height*image_channels
  X_Col_number_of_columns_1 = ((padded_image.shape[2] - filter_width)/stride) + 1
  X_Col_number_of_columns_2 = ((padded_image.shape[3] - filter_height)/stride) + 1
  X_forward = np.zeros((number_images, X_Col_number_of_rows, X_Col_number_of_columns_1*X_Col_number_of_columns_2))

  # applying padding to x
  for k in range(number_images):
    this_example = padded_image[k,:,:,:]
    # Apply im2col as menioned http://cs231n.github.io/convolutional-networks/
    t = 0
    for yy in range(0, X_Col_number_of_columns_1):
      output_layer_width_index = 0
      for xx in range(0, X_Col_number_of_columns_2):
        image_subsection = this_example[:,yy*stride:(yy*stride+filter_height), xx*stride:(xx*stride+filter_width)]
        X_forward[k,:,t] =  np.reshape(image_subsection, (-1,))
        t = t+1

  W_row = np.reshape(w, (number_filters, -1))
  X_col = X_forward
 

  ############ 
  dW_row = np.zeros_like(W_row)
  dX_col = np.zeros_like(X_col)
  db = np.zeros_like(b)
  number_of_samples = dout.shape[0]
  for ex in range(0, number_of_samples):
    this_sample_dout = dout[ex, :, :, :]  # (2,5,5) for the test case
    this_dout_reshaped = np.reshape(this_sample_dout, (this_sample_dout.shape[0], -1)) # (2,25)
    
    dW_row += np.dot(this_dout_reshaped , X_col[ex,:,:].T)
    dX_col[ex,:,:] += np.dot(W_row.T, this_dout_reshaped)
    db += np.sum(this_dout_reshaped, axis = 1)

  dw = np.reshape(dW_row, (w.shape))

  number_of_examples = x.shape[0]
  dx = np.zeros_like(x)
  dx_padded = np.zeros_like(padded_image)
  for ex in range(number_of_examples): 
    # consider only one example
    this_example_dout  = dout[ex] # 
    for output_layers in range(dout.shape[1]):
      # consider only one output layer
      this_filter = w[output_layers]
      dout_for_this_filter = this_example_dout[output_layers]
      for input_layer in range(padded_image.shape[1]):
        # consider only one input layer
        this_layer_image = padded_image[ex, input_layer,:,:] # simple 2d matrix
        this_layer_filter = this_filter[input_layer]
        # consider dout element by element
        for dout_column in range(0,dout_for_this_filter.shape[0]):
          for dout_row in range(0, dout_for_this_filter.shape[1]):
            this_dout = dout_for_this_filter[dout_column, dout_row]
            this_padded_segment_for_convolution =  this_layer_image[(dout_column*stride):(dout_column*stride + filter_height), (dout_row*stride):(dout_row*stride + filter_width)]
            # consider each element in segment one by one
            for segment_column in range(0,this_padded_segment_for_convolution.shape[0]):
              for segment_row in range(0, this_padded_segment_for_convolution.shape[1]):
                #print segment_row
                #print this_layer_filter[segment_column, segment_row]*this_dout
                dx_padded[ex, input_layer, segment_column+dout_column*stride, segment_row + dout_row*stride] += this_layer_filter[segment_column, segment_row]*this_dout










  # # flatten dout similar to x
  # padded_dout = np.pad(dout, pad_width = npad, mode = 'constant', constant_values = 0)

  # # convert dout to appropriate shape for back propogation
  # number_of_examples = dout.shape[0]
  # number_of_output_layers = dout.shape[1]

  # # We will pad dout for each example the the shape will be  
  # #filter_width*filter_height*number_of_output_layers, H_dash*W_dash
  # dout_reshaped = np.zeros((number_of_examples, filter_width*filter_height*number_of_output_layers, H_dash*W_dash))

  # dout_Col_number_of_columns_1 = ((padded_dout.shape[2] - filter_width)/stride) + 1
  # dout_Col_number_of_columns_2 = ((padded_dout.shape[3] - filter_height)/stride) + 1

  # for j in range(number_of_examples):
  #   this_dout = padded_dout[j,:,:,:]
  #   for k in range(number_of_output_layers):
  #     this_layer_dout = this_dout[k]
  #     dout_reshaped_width_index = 0
  #     for yy in range(0, dout_Col_number_of_columns_1):
  #       for xx in range(0, dout_Col_number_of_columns_2):
  #         padded_dout_subsection = this_layer_dout[ yy*stride:(yy*stride+filter_height), xx*stride:(xx*stride+filter_width)]
  #         row_range = range((k*filter_width*filter_height),(k+1)*filter_width*filter_height)
  #         dout_reshaped[j,row_range,dout_reshaped_width_index] = np.reshape(padded_dout_subsection, (-1,))
  #         dout_reshaped_width_index+=1

  # number_of_filter_layers = w.shape[1]
  # reshaped_filter = np.zeros((number_of_filter_layers, w.shape[0]*w.shape[2]*w.shape[3]))
  
  # # We will have to reshape the filter into (#layers in x, w.shape[0]*w.shape[2]*w.shape[3]) and then reverse it.
  # # This is an for example for the case when we have only one filter
  # #             [w133, w132, w131, w123, w122, w121, w113, w112, w111]
  # #             [w233, w232, w231, w223, w222, w221, w213, w212, w211]
  # #             [w333, w332, w331, w323, w322, w321, w313, w312, w311]
  # # If we have multiple filters arrange it [F1, F2, F3] # columns will be appended
  # # number of rpws will stay the same

  # for j in range(number_of_filter_layers):
  #   filter_for_this_layer=[]
  #   for k in range(number_filters):
  #     jth_layer_kth_filter = w[k,j,:,:]
  #     jth_layer_kth_filter_reshaped = np.reshape(jth_layer_kth_filter,(1,-1))
  #     jth_layer_kth_filter_reshaped =  np.fliplr(jth_layer_kth_filter_reshaped)
  #     jth_layer_kth_filter_reshaped = np.reshape(jth_layer_kth_filter_reshaped,(-1,))
  #     filter_for_this_layer = np.concatenate((filter_for_this_layer,jth_layer_kth_filter_reshaped) )
  #   reshaped_filter[j,:] = filter_for_this_layer

  # dx = np.zeros_like(cache[0]) # dx will have same shape as x

  # for j in range(number_of_examples):
  #   dx_for_this_ex = np.dot(reshaped_filter, dout_reshaped[j])# dx for jth example
  #   dx_for_this_ex = np.reshape(dx_for_this_ex, (dx.shape[1], dx.shape[2], dx.shape[3]))
  #   dx[j] = dx_for_this_ex

 
    

  dx = dx_padded[:,:,pad:(pad+x.shape[2]),pad:(pad+x.shape[3])]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db 




def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  # pass
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  number_of_examples = x.shape[0]
  number_of_output_layers = x.shape[1]
  H_dash = 1 + (x.shape[2] - pool_width)/stride
  W_dash = 1 + (x.shape[3] - pool_height)/stride
  depth = x.shape[1]
  out = np.zeros((number_of_examples, depth, H_dash, W_dash))


  for j in range(number_of_examples):
    this_example = x[j,:,:,:]
    for k in range(number_of_output_layers):
      this_layer = this_example[k]
      index_for_out_in_this_layer_height= 0

      for yy in range(0, H_dash):
        index_for_out_in_this_layer_width = 0
        for xx in range(0, W_dash):
          this_layer_subsection = this_layer[ yy*stride:(yy*stride + pool_height), xx*stride:(xx*stride + pool_width)]
          pool_max = np.max(this_layer_subsection)
         # print pool_max
          out[j,k,index_for_out_in_this_layer_height,index_for_out_in_this_layer_width] = pool_max
          index_for_out_in_this_layer_width+=1
          
        index_for_out_in_this_layer_height+=1

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.
  - x: Input data, of shape (N, C, H, W)

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  # pass
  x, pool_param = cache
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  number_of_examples = x.shape[0]
  number_of_layers_in_x = x.shape[1]
  input_height = x.shape[2]
  input_width = x.shape[3]
  dx = np.zeros_like(x)
  output_height = dout.shape[2]
  output_width = dout.shape[3]


  for j in range(number_of_examples):
    this_example = x[j,:,:,:]
    for k in range(number_of_layers_in_x):
      this_layer = this_example[k]
      index_for_out_in_this_layer_height= 0

      for yy in range(0, output_height):
        index_for_out_in_this_layer_width = 0
        for xx in range(0, output_width):
          this_layer_subsection = this_layer[ yy*stride:(yy*stride + pool_height), xx*stride:(xx*stride + pool_width)]
          col_row_max_index = np.where(this_layer_subsection == this_layer_subsection.max())
          pool_max_index_column = np.ndarray.tolist(col_row_max_index[0])
          pool_max_index_row  = np.ndarray.tolist(col_row_max_index[1])
          dx[j, k, pool_max_index_column[0] + yy*stride, pool_max_index_row[0] + xx*stride] = dout[j, k, yy, xx]
          index_for_out_in_this_layer_width+=1
          
        index_for_out_in_this_layer_height+=1



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  #pass
  N,C,H,W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
  x_swapped_axis = np.transpose(x,  (0, 2, 3, 1)) # N,H,W,C
  x_reshaped = np.reshape(x_swapped_axis, (-1, C)) # (N*H*W, C)
  out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
  out_temp = np.reshape(out_reshaped, (N,H,W,C))
  out = np.transpose(out_temp, (0, 3, 1, 2))
  



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  # pass
  N,C,H,W = dout.shape[0], dout.shape[1], dout.shape[2], dout.shape[3]
  dout_swapped_axis = np.transpose(dout,  (0, 2, 3, 1)) # N,H,W,C
  dout_reshaped = np.reshape(dout_swapped_axis, (-1, C)) # (N*H*W, C)
  dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
  dx_temp =  np.reshape(dx_reshaped, (N,H,W,C))
  dx = np.transpose(dx_temp, (0, 3, 1, 2))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
