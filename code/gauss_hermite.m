function [x, w] = gauss_hermite(n)
% GAUSS_HERMITE  Gauss-Hermite quadrature nodes and weights.
%
%   [x, w] = gauss_hermite(n)
%
%   Returns n nodes x and weights w for the physicist Hermite convention:
%
%     integral_{-inf}^{inf}  exp(-x^2) f(x) dx  ~=  sum_{i=1}^n  w(i) * f(x(i))
%
%   To compute expectations of a N(0,1) random variable epsilon:
%
%     E[g(epsilon)]  ~=  (1/sqrt(pi)) * sum  w(i) * g( sqrt(2) * x(i) )
%
%   Uses the Golub-Welsch algorithm (eigenvalue decomposition of the
%   symmetric tridiagonal Jacobi matrix for the Hermite polynomials).

    if n == 1
        x = 0;
        w = sqrt(pi);
        return;
    end

    % Sub-diagonal entries of the Jacobi matrix
    beta_j = sqrt( (1:(n-1)) / 2 );

    % Symmetric tridiagonal companion matrix
    J = diag(beta_j, 1) + diag(beta_j, -1);

    % Eigenvalue decomposition
    [V, D] = eig(J);

    x = diag(D);                       % nodes = eigenvalues
    w = sqrt(pi) * V(1, :)'.^2;        % weights from first row of eigenvectors

    % Sort nodes in ascending order
    [x, idx] = sort(x);
    w = w(idx);
end
