function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
%temp_size = length(sigma_vec)*length(C_vec);
prediction_error_mat = zeros(length(sigma_vec), length(C_vec));


for i = 1:length(sigma_vec),
  temp_sigma = sigma_vec(i);
  for j = 1:length(C_vec),
    temp_C = C_vec(j);
    model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
    predictions = svmPredict(model, Xval);
    temp_error = mean(double(predictions ~= yval));
    prediction_error_mat(i,j) = temp_error;
  end;
end;


temp_index_i = 1;
temp_index_j = 1;

for i = 1:length(sigma_vec),
  for j = 1:length(C_vec),
    if(prediction_error_mat(i,j) < prediction_error_mat(temp_index_i,temp_index_j)),
      temp_index_i = i;
      temp_index_j = j;
    end;
  end;
end;

sigma = sigma_vec(temp_index_i);
C = C_vec(temp_index_j);
      




% =========================================================================

end
