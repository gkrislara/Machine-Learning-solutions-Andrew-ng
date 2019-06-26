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
err=zeros(size(X*theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

err=X*theta-y;%X*theta= arr of h(x)
s=sumsq(err);
r=sumsq(theta(2:length(theta)));
r=lambda*r*0.5/m; %regularisation
J=(0.5*s/m)+r; %cost function

for i=1:size(theta)(1)
  if(i!=1)
  grad(i)= (sum(err.*X(:,i))/m)+(lambda*theta(i)/m);  
  else
  grad(i)= (sum(err.*X(:,i))/m);
  endif
end 







% =========================================================================

grad = grad(:);

end
