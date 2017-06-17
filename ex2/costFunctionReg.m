function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
h = sigmoid(X * theta);

theta(1) = 0;
% You need to return the following variables correctly 
J = (1/m)*sum(-y.*log(h) .-(1.-y).*log(1-h)) + lambda/(2*m)*sum(theta.^2);
grad = (1/m) * (X'*(h .- y)) + lambda/m * theta;

end
