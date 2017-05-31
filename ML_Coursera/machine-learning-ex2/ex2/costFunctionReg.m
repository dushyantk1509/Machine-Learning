function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

[J1, grad1] = costFunction(theta, X, y);

a = size(theta);
sum = 0;

for i = 2:a,
  sum = sum + (theta(i,1))^2;
end;

J = J1 + (lambda/(2*m) * sum); 

grad(1,1) = grad1(1,1);

for i=2:a,
  grad(i,1) = grad1(i,1) + (lambda/m)*theta(i,1);
end;




% =============================================================

end
