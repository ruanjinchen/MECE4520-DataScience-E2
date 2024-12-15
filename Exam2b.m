clc; clear; close all;

xi = linspace(-10, 10, 400); % Spatial domain
t = linspace(0, 15, 200); % Temporal domain
dt = t(2) - t(1); % Time step
[xgrid, T] = meshgrid(xi, t);

f1 = xgrid .* exp(8i * T);         % x * [cos(4T) + i*sin(4T)]
f2 = (xgrid.^2) .* exp(6i * T);    % x^2 * [cos(2T) + i*sin(2T)]

f = f1 + f2; % Sum of two waves
X = f';

% Plot f1, f2, and f
figure(1);
subplot(221); 
surf(real(f1)); title('f1: Wave 1 (Real Part)'); xlabel('x'); ylabel('t'); grid;
shading interp; colormap(gray);

subplot(222); 
surf(real(f2)); title('f2: Wave 2 (Real Part)'); xlabel('x'); ylabel('t'); grid;
shading interp; colormap(gray);

subplot(223);
surf(real(f)); xlabel('space (x)'); ylabel('time (t)');
shading interp; colormap(gray); title('f (=f1+f2)');

% Define X1 & its time-shifted version
X1 = X(:, 1:end-1);
X2 = X(:, 2:end);

% SVD of X1
r = 2; % Only 2 significant singular values
[U, S, V] = svd(X1, 'econ');
Ur = U(:, 1:r);
Sr = S(1:r, 1:r);
Vr = V(:, 1:r);

% Plot singular values
figure(2); semilogy(diag(S), 'k*'); title('Singular Values'); grid;

% Eigendecomposition of reduced A 
Atilde = Ur' * X2 * Vr / Sr;
[W, L] = eig(Atilde); % Eigendecomposition of reduced A
Phi = X2 * (Vr / Sr) * W; % Spatial modes

% Eigenvalues (frequencies of temporal part)
lamda = diag(L);
om = log(lamda) / dt;

% Initial Conditions
x1 = X1(:, 1); % Initial condition
b = Phi \ x1; % Project IC onto eigenvector space

% Reconstruct
y = zeros(r, length(t));
for i = 1:length(t)
    y(:, i) = b .* exp(om * t(i)); % Reconstructed temporal part
end
xdmd = Phi * y; % Phi: spatial part, y: temporal part

% DMD reconstruction plot
figure(1); subplot(224); 
surf(real(xdmd'));
shading interp; colormap(gray);
xlabel('x'); ylabel('t'); title('DMD Reconstructed f');

% Plot Modes
figure(3);
subplot(221);
plot(t, real(y(1, :))); grid;
title('Temporal Mode 1'); % Close to temporal part of f1

subplot(222);
plot(t, real(y(2, :))); grid;
title('Temporal Mode 2'); % Close to temporal part of f2

subplot(223);
plot(xi, real(Phi(:, 1))); grid;
title('Spatial Mode 1'); % Close to spatial part of f1

subplot(224);
plot(xi, real(Phi(:, 2))); grid;
title('Spatial Mode 2'); % Close to spatial part of f2
