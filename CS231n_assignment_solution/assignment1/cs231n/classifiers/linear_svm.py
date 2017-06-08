import numpy as np
from random import shuffle
from past.builtins import xrange

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
  temp_mat = np.zeros([num_train,num_classes])
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        #temp_mat[i,j] = -(num_classes-1)
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        temp_mat[i,j] = 1
        count = count + 1 
      else:
        temp_mat[i,j] = 0;
    temp_mat[i,y[i]] = -count

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  dW = np.dot(X.T,temp_mat)
  dW = dW / num_train
  dW = dW + 2*reg*W
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  temp_mat = np.zeros([X.shape[0],W.shape[1]])
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  temp_score = np.dot(X,W)
  temp_correct_class_score = temp_score[list(range(temp_score.shape[0])),y]
  temp_correct_class_score = np.array(temp_correct_class_score)[np.newaxis]
  temp_score = temp_score - temp_correct_class_score.T + 1
  temp_score[list(range(temp_score.shape[0])),y] = 0
  temp_mat = np.maximum(temp_score,0)
  temp_loss = np.sum(np.sum(temp_mat,axis=0))
  loss = temp_loss / X.shape[0]
  loss += reg * np.sum(W * W)
  pass
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
  pos = temp_mat > 0
  temp_mat[pos] = 1
  sum_vec = np.sum(temp_mat,axis=1)
  sum_vec = np.array(sum_vec)[np.newaxis]
  temp_mat[list(range(temp_mat.shape[0])),y] = - sum_vec
  dW = np.dot(X.T,temp_mat)
  dW = dW / X.shape[0]
  dW = dW + 2*reg*W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
