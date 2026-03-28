function sol = solve_period1(y1, b1, par, csv, x0)
% SOLVE_PERIOD1  Solve the Period 1 equilibrium and recover all variables.
%
%   sol = solve_period1(y1, b1, par, csv)
%   sol = solve_period1(y1, b1, par, csv, x0)
%
%   Inputs:
%     y1   - Period 1 endowment
%     b1   - Outstanding sovereign debt (issued in Period 0)
%     par  - Parameter struct (must contain nbar, gamma, mu, Rstar, alpha, eta, sigma, sigma_u)
%     csv  - CSV function handles (from csv_functions)
%     x0   - [optional] Initial guess [D; omegabar; Mf]
%
%   Output:
%     sol  - Struct containing all Period 1 equilibrium variables:
%            D, omegabar, Mf, N, ell, K, pm, Z, Z_Rstar,
%            mf, md, C1, Pi, lambda1, dpm_dN, exitflag

    % Store state variables in par for the residual function
    par.y1 = y1;
    par.b1 = b1;

    % Default initial guess
    if nargin < 5 || isempty(x0)
        D0   = b1 * 0.3;
        N0   = par.nbar + par.gamma * (b1 - D0);
        Mf0  = N0 / 0.5;            % target leverage ell ~ 0.5
        wb0  = 0.5;                  % omegabar initial guess
        x0   = [D0; wb0; Mf0];
    end

    % fsolve options
    opts = optimoptions('fsolve', ...
        'Display',          'off', ...
        'TolFun',           1e-12, ...
        'TolX',             1e-12, ...
        'MaxFunEvals',      5000, ...
        'MaxIter',          1000);

    % Solve the 3-equation nonlinear system
    [xsol, ~, exitflag] = fsolve( ...
        @(x) period1_residuals(x, par, csv), x0, opts);

    if exitflag <= 0
        warning('solve_period1:noConverge', ...
            'fsolve did not converge (y1=%.4f, b1=%.4f, flag=%d).', ...
            y1, b1, exitflag);
    end

    % ---- Extract solution ----
    sol.D        = xsol(1);
    sol.omegabar = xsol(2);
    sol.Mf       = xsol(3);

    % ---- Derived quantities ----
    sol.N   = par.nbar + par.gamma * (b1 - sol.D);
    sol.ell = 1 - sol.N / sol.Mf;
    sol.K   = sol.Mf - sol.N;                            % working capital

    % CSV evaluations at solution
    Gam  = csv.Gamma(sol.omegabar);
    Gp   = csv.GammaPrime(sol.omegabar);
    Gpp  = csv.GammaDoublePrime(sol.omegabar);
    G_v  = csv.G(sol.omegabar);
    Psi  = csv.Psi(sol.omegabar);
    Psip = csv.PsiPrime(sol.omegabar);
    muG  = par.mu * G_v;

    % Prices
    sol.pm      = par.Rstar * sol.ell / Psi;             % import price
    sol.Z       = par.Rstar * sol.omegabar / Psi;        % lending rate
    sol.Z_Rstar = sol.omegabar / Psi;                    % external finance premium

    % Allocations
    sol.mf  = sol.Mf * (1 - muG);                       % effective imports to household
    sol.md  = sol.mf * (par.alpha * sol.pm / (1 - par.alpha))^par.sigma;  % domestic inputs
    sol.C1  = ( par.alpha * sol.md^par.eta ...
              + (1 - par.alpha) * sol.mf^par.eta )^(1/par.eta);          % consumption
    sol.Pi  = (1 - Gam) * sol.pm * sol.Mf;               % aggregate dividends

    % Shadow value of income:  lambda = u'(C) * alpha * (C / md)^(1-eta)
    sol.lambda1 = sol.C1^(-par.sigma_u) * par.alpha ...
                  * (sol.C1 / sol.md)^(1 - par.eta);

    % ---- Derivative dpm/dN (needed for Euler equation) ----
    %
    % d omegabar / d N  [eq:domega_dn]:
    denom_inner = Gp * (sol.Mf - sol.N) ...
                + sol.N * (Gpp * sol.omegabar + Gp);
    domega_dN   = -Gp * sol.omegabar * sol.Mf ...
                / ( (sol.Mf - sol.N) * denom_inner );

    % dpm / dN  [eq:dpm_dn]:
    %   = R* * [ -Psi/Mf  -  ell * Psi' * (domega/dN) ] / Psi^2
    sol.dpm_dN = par.Rstar ...
                * ( -Psi / sol.Mf  -  sol.ell * Psip * domega_dN ) ...
                / Psi^2;

    sol.exitflag = exitflag;
end
