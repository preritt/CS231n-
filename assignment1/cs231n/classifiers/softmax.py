import numpy as np
from random import shuffle

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """

    ### YOUR CODE HERE
#     raise NotImplementedError
    ### YOUR CODE HERE
    # Case1 When dimension is greater than 1
    if (x.ndim>1):
        x=x-np.max(x,axis=1,keepdims=True)
#         print x
        # now take exp
        x=np.exp(x)
#         print x
        
        x=x/x.sum(axis=1,keepdims=True)
    else:
        x=x-np.max(x, axis=0, keepdims=True)
        x=np.exp(x)
        x=x/x.sum(axis=0, keepdims=True)


    ### END YOUR CODE
#     max_arg=np.max_arg
#     x=
    return x

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    #print(scores.shape, "scores")
    scores_softmaxed = softmax(scores)
    #print(scores_softmaxed, "scores_softmaxed")
    correct_class_softmax_score = scores_softmaxed[y[i]]
    loss += -np.log(correct_class_softmax_score)
    #print(loss,"loss")
    this_eg = np.reshape(X[i], (-1,1))
    scores_softmaxed = np.reshape(scores_softmaxed, (1,-1))
    target = np.zeros_like(scores_softmaxed)
    #print(scores_softmaxed.shape, "scores_softmaxed")
    target[0,y[i]] =1 
    
    delta = scores_softmaxed - target
    this_gradient = np.dot(this_eg,delta )
    #print(this_gradient.shape,"this_gradient")
    dW += this_gradient
    
  dW /= num_train
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros_like(W)
  scores = X.dot(W)
  scores_softmax = softmax(scores)
  targets = np.zeros_like(scores_softmax)
  targets[range(X.shape[0]),y] = 1
  delta = scores_softmax - targets
  #print(delta.shape, "delta")
  dW = np.dot(X.T, delta)
  softmax_corresponding_to_correct_labels = scores_softmax[range(X.shape[0]), y]
  loss = -np.sum(np.log(softmax_corresponding_to_correct_labels))

  dW /= num_train
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

