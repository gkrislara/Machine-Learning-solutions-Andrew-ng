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
k=0;
l=0;
r=0;
h=zeros(size(theta));
t=zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

  for i=1:m
     l=sigmoid(X(i,:)*theta);
     h=h+((l-y(i))*X(i,:)');
     k= y(i)*log(l)+(1-y(i))*log(1-l);
     J=J-(k/m);       
   end
   for j=2:size(theta)(1)
    r=r+(theta(j)^2);
   end
    J=J+((lambda*r)/(2*m));
     t=theta;
     t(1)=0;
     grad=h/m;
     grad=grad+(lambda*t/m);
     




% =============================================================

end
