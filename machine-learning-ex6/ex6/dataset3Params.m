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
choice_val=[0.01,0.03,0.1,0.3,1,3,10,30]';
pred_mat=zeros(200:1);
err_mat=zeros(size(choice_val)(1)^2:1);
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
%0.01,0.03,0.1,0.3,1,3,10,30
k=1;
for i=1:numel(choice_val)
  for j=1:numel(choice_val)
    model=svmTrain(X, y, choice_val(i), @(x1, x2) gaussianKernel(x1, x2, choice_val(j)));
    pred_mat=svmPredict(model,Xval);
    err_mat(k)=mean(double(pred_mat~=yval));
    k=k+1;
  end
end


[minimum idx]=min(err_mat);

sigma_index=mod(idx,numel(choice_val));

if (sigma_index==0)
  sigma_index=numel(choice_val);
endif

c_index=floor(idx/numel(choice_val));

C=choice_val(c_index);
sigma=choice_val(sigma_index);





% =========================================================================

end
