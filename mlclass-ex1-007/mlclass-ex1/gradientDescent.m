function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp_sum = zeros(2,1);
alpham = alpha / m;


%fprintf('initial theta: %f, %f \n', theta(1), theta(2));
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    for i = 1:m
      temp_sum += alpham * ((theta' * X(i,:)') - y(i)) .* X(i,:)';
    end

    theta -= temp_sum;

    cost = computeCost(X, y, theta);
    fprintf('iter: %d \t\t cost : %f \t\t theta: [%f, %f]\n', iter, cost, theta(1), theta(2));
    temp_sum = zeros(2,1);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = cost;

    if (iter > 1)
      if ((cost - J_history(iter -1)) >= 0)
        break;
      end
    end
end

end
