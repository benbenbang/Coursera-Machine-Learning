% Clean up
clear; close all; clc;

% Data 1 
load('ex8_movies.mat');
load ('ex8_movieParams.mat');

%  Reduce the data set size so that this runs faster
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);
params = [X(:) ; Theta(:)];
[J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0);


n_u = 3; n_m = 4; n = 5;
X2 = reshape(sin(1:n_m*n), n_m, n);
Theta2 = reshape(cos(1:n_u*n), n_u, n);
Y2 = reshape(sin(1:2:2*n_m*n_u), n_m, n_u);
R2 = Y2 > 0.5;
pval2 = [abs(Y2(:)) ; 0.001; 1];
yval2 = [R2(:) ; 1; 0];
params2 = [X2(:) ; Theta2(:)];

[J2, grad2] = cofiCostFunc(params2, Y2, R2, n_u, n_m, n, 0);