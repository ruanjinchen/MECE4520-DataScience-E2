clc; clear; close all;

n_samples = 100; % Number of samples
x1 = -1 + 2 * rand(n_samples, 1); % Random x1 in [-1, 1]
x2 = -1 + 2 * rand(n_samples, 1); % Random x2 in [-1, 1]

% Define the true coefficients for the system
a_true = [1.0, 0.5, -0.8, 0.3, 0.2, -0.4, 0.1, -0.2, 0.05, 0.01]; % For x1_dot equation
b_true = [0.8, -0.6, 0.4, -0.3, 0.2, -0.1, 0.05, -0.05, 0.02, -0.01]; % For x2_dot equation

% Generate x1_dot and x2_dot based on true coefficients
x1_dot_generated = a_true(1) + a_true(2)*x1 + a_true(3)*x2 + ...
                    a_true(4)*x1.^2 + a_true(5)*x1.*x2 + a_true(6)*x2.^2 + ...
                    a_true(7)*x1.^3 + a_true(8)*x1.^2.*x2 + a_true(9)*x1.*x2.^2 + a_true(10)*x2.^3;

x2_dot_generated = b_true(1) + b_true(2)*x1 + b_true(3)*x2 + ...
                    b_true(4)*x1.^2 + b_true(5)*x1.*x2 + b_true(6)*x2.^2 + ...
                    b_true(7)*x1.^3 + b_true(8)*x1.^2.*x2 + b_true(9)*x1.*x2.^2 + b_true(10)*x2.^3;

% Prepare delayed inputs for NARX model
delay = 2; % Number of past timesteps to use
X = zeros(n_samples - delay, 4); % Features: x1(t-1), x1(t-2), x2(t-1), x2(t-2)
Y1 = x1_dot_generated(delay+1:end); 
Y2 = x2_dot_generated(delay+1:end); 

for t = delay+1:n_samples
    X(t-delay, :) = [x1(t-1), x1(t-2), x2(t-1), x2(t-2)];
end

X_poly_x1 = [X(:,1).^2, X(:,2).^2, X(:,3).^2, X(:,4).^2, ...
             X(:,1).*X(:,3), X(:,2).*X(:,4), X]; 
X_poly_x2 = X_poly_x1; 

% Solve for coefficients using Least Squares Estimation
a_estimated = (X_poly_x1' * X_poly_x1) \ (X_poly_x1' * Y1); % LSE solution for x1_dot
b_estimated = (X_poly_x2' * X_poly_x2) \ (X_poly_x2' * Y2); % LSE solution for x2_dot

% Predictions using the NARX model
Y1_pred = X_poly_x1 * a_estimated; % Predicted x1_dot
Y2_pred = X_poly_x2 * b_estimated; % Predicted x2_dot

% Compare coefficients
disp('True coefficients for x1_dot:');
disp(a_true');
disp('Estimated coefficients for x1_dot:');
disp(a_estimated);
disp('True coefficients for x2_dot:');
disp(b_true');
disp('Estimated coefficients for x2_dot:');
disp(b_estimated);

% Plot for x1_dot
figure;
plot(delay+1:n_samples, Y1, 'b', 'LineWidth', 1.5); % Original x1_dot
hold on;
plot(delay+1:n_samples, Y1_pred, 'r--', 'LineWidth', 1.5); % Predicted x1_dot
title('Original vs Predicted x1\_dot (NARX)');
xlabel('Sample Index'); ylabel('x1\_dot'); legend('Original x1\_dot', 'Predicted x1\_dot'); grid on;

% Plot for x2_dot
figure;
plot(delay+1:n_samples, Y2, 'b', 'LineWidth', 1.5); % Original x2_dot
hold on;
plot(delay+1:n_samples, Y2_pred, 'r--', 'LineWidth', 1.5); % Predicted x2_dot
title('Original vs Predicted x2\_dot (NARX)');
xlabel('Sample Index'); ylabel('x2\_dot'); legend('Original x2\_dot', 'Predicted x2\_dot'); grid on;