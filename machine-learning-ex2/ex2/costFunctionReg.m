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

z = X * theta;
h_theta = sigmoid(z);
yM = [-y;-(1-y)];
hM = [log(h_theta); log(1-h_theta)];
theta_reg = [0;theta(2:end, :);];
reg = (lambda / (2 * m)) * (theta_reg'*theta_reg);

J = (1/m) * (yM' * hM) + reg;

grad = ((1/m)* sum((h_theta - y) .* X))' + (lambda/m) .* theta_reg;
output_precision(3);

% =============================================================

end
