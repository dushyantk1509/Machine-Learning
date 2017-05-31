function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%J = (1/(2*m) * (X*theta - y)'*(X*theta - y)) + (lambda/(2*m) * (theta'*theta) - theta(1,1)*theta(1,1));

temp1 = X*theta;
temp2 = temp1 - y;
temp3 = temp2'*temp2;
temp4 = temp3/(2*m);
temp5 = theta'*theta;
temp5 = temp5 - theta(1,1)*theta(1,1);
temp5 = temp5*lambda;
temp5 = temp5/(2*m);
J = temp4 + temp5;

temp6 = X'*temp2;
temp6 = 1/m * temp6;
temp7 = lambda/m * theta;
temp7(1,1) = temp7(1,1) - (lambda/m * theta(1,1));
grad = temp6 + temp7;









% =========================================================================

grad = grad(:);

end
