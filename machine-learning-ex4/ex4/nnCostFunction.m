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

Y_temp = zeros(size(Theta2,1),size(y,1));

for i = 1:size(y,1),
	Y_temp(y(i),i) = 1;
end


col_x1 = ones(m,1);
a1 = [col_x1, X];

z2 = Theta1 * a1';
a2 = sigmoid(z2);
col_x2 = ones(size(a2,2),1);
a2 = [col_x2'; a2];

z3 = Theta2 * a2;
h = sigmoid(z3);

tl = log(h);
tr = log(1 - h);

Theta1_reg = Theta1(1:size(Theta1,1),2:size(Theta1,2));
Theta2_reg = Theta2(1:size(Theta2,1),2:size(Theta2,2));


J = -(1/m) * (sum(sum(tl .* Y_temp + tr .* (1-Y_temp)))) + (lambda / (2*m)) * ...
                             (sum(sum((Theta1_reg).^2)) + sum(sum((Theta2_reg).^2)));


delta_3 = h .- Y_temp; 
% dim(delta3) = 10 x 5000

z2_d = [col_x2';z2];
delta_2_temp = Theta2' * delta_3 .* sigmoidGradient(z2_d); 
delta_2 = delta_2_temp(2:size(delta_2_temp,1),1:size(delta_2_temp,2));
% dim(Theta2) = 10 x 26; dim(delta_2_temp) = 26 x 5000; 
% remove delta2_0; dim(delta_2) = 25 x 5000;

Delta2 = 0;
Delta1 = 0;

zero_t1 = zeros(size(Theta1,1),1);
Theta1_dt = [zero_t1,Theta1(:,2:end)];
zero_t2 = zeros(size(Theta2,1),1);
Theta2_dt = [zero_t2,Theta2(:,2:end)];

Delta2 = Delta2 + delta_3 * a2' + lambda * Theta2_dt;
Delta1 = Delta1 + delta_2 * a1 + lambda * Theta1_dt;

Theta1_grad = (1/m) * Delta1;  % 25 * 401
Theta2_grad = (1/m) * Delta2;  % 10 * 26

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
