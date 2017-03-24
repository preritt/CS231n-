import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    no_classes_that_dont_meet_margin = 0;
    
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        no_classes_that_dont_meet_margin +=1
        dW[:, j] += X[i]
    #print(dW.shape) 
    #print((X[i]).shape)
    dW[:, y[i]] += -X[i]*no_classes_that_dont_meet_margin
    #print(dW.shape)
     
    #input_eg = np.reshape(X[i], (-1, 1))
    #dw += np.dot(input_eg , margin)
  dW /= num_train
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  #pass
  #scores = np.dot(X, W) # (N,C)
  #scores_for_correct_labels = scores[range(X.shape[0]),y]
  #scores_for_correct_labels = np.reshape(scores_for_correct_labels, (-1,1))
  #print(scores_for_correct_labels.shape)
  #print(scores.shape)
  #scores =  scores - scores_for_correct_labels + 1
  #scores[range(X.shape[0]),y] = 0
  #scores[scores<0] = 0
  #count_clasees_that_didnt_meet_margin = scores
  #count_clasees_that_didnt_meet_margin[count_clasees_that_didnt_meet_margin>0] = 1 #(N,C)
  #count_clasees_that_didnt_meet_margin_row_sum = np.sum(count_clasees_that_didnt_meet_margin, axis = 1)
  #dW[:,y] += np.dot(X.T, count_clasees_that_didnt_meet_margin_row_sum)
  #print(dW)
  #dW +=  np.dot(X.T,count_clasees_that_didnt_meet_margin)
  
  #print(dW.shape)
  
  
  #dW[]

  scores = np.dot(X, W) # (N,C)
  scores_for_correct_labels = scores[range(X.shape[0]),y]
  scores_for_correct_labels = np.reshape(scores_for_correct_labels, (-1 ,1))
  scores =  scores - scores_for_correct_labels + 1
  scores[range(X.shape[0]),y] = 0
  scores[scores<0] = 0
  indicator_scores = scores>0
  #print(indicator_scores.shape)
  indicator_scores = indicator_scores.astype(int)  
  scale_factor_for_correct_class = -np.sum(indicator_scores, axis = 1) # number of incorrect classes based on 
# hinge loss which will be used to find gradient wrt the correct class for each example
  
  #incorrect_classes_in_each_row_scaled = np.reshape(scale_factor_for_correct_class, (-1 ,1))
  #print("incorrect_classes_in_each_row_scaled", incorrect_classes_in_each_row_scaled.shape)
  indicator_scores[range(X.shape[0]), y] = scale_factor_for_correct_class # replacing the index corresponding to the correct class
# with a factor that will consider that example a number of times dependent on the number of incorrect classes for that example
  # gradient update due to all but the correct class
  #dW = (np.dot(indicator_scores.T , X)).T
  dW = np.dot(X.T,indicator_scores)
  
  #print (dW.shape)
  
  loss = np.sum(scores) 
  loss  = loss/X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)
  #print(loss)
  dW /= num_train
  dW += reg*W
 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
