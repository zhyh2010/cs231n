import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    f = np.dot(X[i], W)
    expf = np.exp(f)
    sum_expf = np.sum(expf)
    loss += - np.log(expf[y[i]] / sum_expf)
    for j in range(W.shape[1]):
        dW[:, j] += expf[j] / sum_expf * X[i]
    dW[:, y[i]] -= X[i]
    
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)
  dW /= X.shape[0]
  dW += reg * W * 2
      
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
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = np.dot(X, W)
  expf = np.exp(f)
  expf_sum = np.sum(expf, axis = 1)
  expf_norm = np.divide(expf, np.matrix(expf_sum).T * np.ones((1, W.shape[1])))
  expf_each = np.divide(expf[range(f.shape[0]), y], expf_sum)
  loss = - np.mean(np.log(expf_each)) + np.sum(W * W) * reg
  expf_norm[range(expf_norm.shape[0]), y] -= 1
  dW = np.dot(X.T, expf_norm) / X.shape[0] + reg * W * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

