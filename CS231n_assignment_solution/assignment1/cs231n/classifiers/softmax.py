import numpy as np
from random import shuffle
from past.builtins import xrange
import math

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  temp_mat = np.zeros([num_train,num_classes])
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    sum_total = 0.0
    for j in xrange(num_classes):
        sum_total += math.exp(scores[j])
    for j in xrange(num_classes):
        if j==y[i]:
            temp_mat[i,j] = (math.exp(scores[j]) - sum_total) / sum_total
        else:
            temp_mat[i,j] = math.exp(scores[j]) / sum_total
    temp = math.exp(correct_class_score)
    loss += -math.log(temp / sum_total)
  loss /= num_train
  loss += 0.5*reg * np.sum(W * W)
  dW = np.dot(X.T,temp_mat)
  dW = dW / num_train
  dW = dW + reg*W  
  pass
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  temp_mat = np.zeros([num_train,num_classes])
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores_mat = X.dot(W)
  scores_mat = np.exp(scores_mat)
  sum_mat = np.sum(scores_mat,axis=1)[np.newaxis]
  prob_mat = scores_mat / sum_mat.T
  loss = -np.sum(np.log(prob_mat[list(range(num_train)),y]))
  loss /= num_train
  loss += 0.5*reg * np.sum(W * W)
  prob_mat[list(range(num_train)),y] -= 1
  dW = np.dot(X.T,prob_mat)
  dW = dW / num_train
  dW = dW + reg*W 
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

