clear; close all; clc;

a = [0.8, -1.5, 0.3]; % coefficients for dx1: [x1, x2, x1*x2]
b = [-0.6, 1.2, -0.4]; % coefficients for dx2: [x1, x2, x1*x2]

% Define the system of ODEs
f_ode = @(t, X) [a(1)*X(1) + a(2)*X(2) + a(3)*X(1)*X(2); ...
                 b(1)*X(1) + b(2)*X(2) + b(3)*X(1)*X(2)];

tspan = linspace(0, 5, 200); % Time range
X0 = [1; -1]; % Initial conditions
[time, X] = ode45(f_ode, tspan, X0);

x1 = X(:, 1); % Extract x1
x2 = X(:, 2); % Extract x2
dt = time(2) - time(1); % Time step

% Compute derivatives using central differences
x1dot = [0; (x1(3:end) - x1(1:end-2)) / (2*dt); 0]; % Derivative of x1
x2dot = [0; (x2(3:end) - x2(1:end-2)) / (2*dt); 0]; % Derivative of x2

Theta = [x1, x2, x1.*x2]; % Design matrix with terms [x1, x2, x1*x2]

alpha = 1e-4; % Regularization parameter
a_est = (Theta' * Theta + alpha * eye(size(Theta, 2))) \ (Theta' * x1dot);
b_est = (Theta' * Theta + alpha * eye(size(Theta, 2))) \ (Theta' * x2dot);

disp('True Coefficients for x1dot:');
disp(a');
disp('Estimated Coefficients for x1dot:');
disp(a_est);
disp('True Coefficients for x2dot:');
disp(b');
disp('Estimated Coefficients for x2dot:');
disp(b_est);

x1dot_est = Theta * a_est; % Estimated derv.x1
x2dot_est = Theta * b_est; % Estimated derv.x2

figure;
subplot(2, 1, 1);
plot(time, x1dot, 'k', 'LineWidth', 2); hold on;
plot(time, x1dot_est, 'r--', 'LineWidth', 2);
title('dx1: Original vs. Estimated');
xlabel('Time'); ylabel('x1dot');
legend('Original', 'Estimated');
grid on;
subplot(2, 1, 2);
plot(time, x2dot, 'k', 'LineWidth', 2); hold on;
plot(time, x2dot_est, 'r--', 'LineWidth', 2);
title('dx2: Original vs. Estimated');
xlabel('Time'); ylabel('x2dot');
legend('Original', 'Estimated');
grid on;