function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X = [ones(m, 1) X];
tempy = zeros(m,num_labels);
for i = 1:m,
  tempy(i,y(i,1)) = 1;
end;

temp1 = X*Theta1';
temp1 = sigmoid(temp1);

m1 = size(temp1, 1);
temp1 = [ones(m1, 1) temp1];

temp2 = temp1*Theta2';
temp2 = sigmoid(temp2);

temp3 = log(temp2);
temp4 = log(1-temp2);

sumtemp = 0;

for i = 1:m,
  for j = 1:num_labels,
    sumtemp = sumtemp + (tempy(i,j)*temp3(i,j) + (1-tempy(i,j))*temp4(i,j));
  end;
end;

J = 1/m * sumtemp * (-1);

% regularized version

[nrows1 ncolms1] = size(Theta1);
[nrows2 ncolms2] = size(Theta2);

sumtheta1 = 0;
for i = 1:nrows1,
  for j = 2:ncolms1,
    sumtheta1 = sumtheta1 + (Theta1(i,j))^2;
  end;
end;

sumtheta2 = 0;
for i = 1:nrows2,
  for j = 2:ncolms2,
    sumtheta2 = sumtheta2 + (Theta2(i,j))^2;
  end;
end; 

sumall = sumtheta1 + sumtheta2;
sumall = lambda/(2*m) * sumall;

J = J + sumall;

bigdelta3 = zeros(m,num_labels);
bigdelta2 = zeros(m,hidden_layer_size);
bigdelta1 = zeros(m,input_layer_size);


for i = 1:m,
  a1 = X(i,:);
  temp1 = a1*Theta1';
  a2 = sigmoid(temp1);
  a2 = [ones(1,1) a2];
  temp2 = a2*Theta2';
  a3 = sigmoid(temp2);
  delta3 = a3 - tempy(i,:);
  temp1 = [ones(1,1) temp1];
  delta2 = (Theta2'*delta3')'.*sigmoidGradient(temp1);
  delta2 = delta2(2:end);
  bigdelta2(i,:) = bigdelta2(i,:) + 
  
  
  













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
