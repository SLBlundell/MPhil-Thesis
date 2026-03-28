%% SOLVE_GE_MODEL
%  Two-Period General Equilibrium Model with Sovereign Default
%  and Costly State Verification (CSV) Financial Frictions
%
%  This script solves the full two-period competitive equilibrium:
%
%    Period 1:  Given (y1, b1), solve a 3-equation nonlinear system
%              for (D, omegabar, Mf) using fsolve.
%
%    Period 0:  Given Period 1 policy functions, solve the Euler
%              equation for optimal bond issuance b1 using fsolve.
%
%  The model features:
%    - CES consumption aggregator with complementary inputs (eta < 0)
%    - Endogenous sovereign default with convex penalty
%    - CSV financial friction (Bernanke-Gertler-Gilchrist 1999)
%    - Home bias in sovereign debt holdings (gamma)
%
%  See main.tex, equations [sys_leverage], [sys_clearing], [sys_default].

clear; clc;
fprintf('============================================================\n');
fprintf('  Two-Period GE Model with Sovereign Default & CSV Friction\n');
fprintf('============================================================\n\n');

%% ========================= PARAMETERS ================================

% Preferences
par.beta    = 0.96;         % discount factor
par.sigma_u = 2;            % CRRA risk aversion

% CES aggregator
par.alpha   = 0.68;         % Armington weight on domestic inputs
par.eta     = -1;           % CES parameter (eta < 0  =>  complements)
par.sigma   = 1/(1-par.eta);% elasticity of substitution = 1/(1-eta)

% Financial sector
par.nbar    = 0.03;         % long-run intermediary net worth
par.gamma   = 0.60;         % home bias share of sovereign debt
par.sigma_w = 0.35;         % idiosyncratic shock volatility (sigma_omega)
par.mu      = 0.25;         % CSV monitoring cost

% International
par.Rstar   = 1.04;         % world gross interest rate

% Endowment process:  log(g1) = (1-rho) log(mu_g) + rho log(g0) + eps
par.y0      = 1.0;          % Period 0 endowment (normalised)
par.b0      = 0.30;         % pre-existing debt
par.rho     = 0.90;         % AR(1) persistence of growth
par.mu_g    = 1.00;         % unconditional mean growth rate
par.sigma_g = 0.034;        % growth shock std. dev.
par.g0      = 1.00;         % initial growth rate

fprintf('Parameters:\n');
fprintf('  beta=%.2f  sigma_u=%.1f  alpha=%.2f  eta=%.1f  sigma=%.2f\n', ...
    par.beta, par.sigma_u, par.alpha, par.eta, par.sigma);
fprintf('  nbar=%.2f  gamma=%.2f  sigma_w=%.2f  mu=%.2f  R*=%.2f\n', ...
    par.nbar, par.gamma, par.sigma_w, par.mu, par.Rstar);
fprintf('  y0=%.2f  b0=%.2f  rho=%.2f  mu_g=%.2f  sigma_g=%.3f\n\n', ...
    par.y0, par.b0, par.rho, par.mu_g, par.sigma_g);

%% ================== CSV FUNCTION HANDLES ==============================

csv = csv_functions(par.sigma_w, par.mu);

%% ================== GAUSS-HERMITE QUADRATURE ==========================

nq = 7;                     % number of quadrature nodes
[xi, wi] = gauss_hermite(nq);

% Transform nodes to y1 realisations
%   log(g1) ~ N( mu_log, sigma_g^2 )
mu_log_g1 = (1 - par.rho) * log(par.mu_g) + par.rho * log(par.g0);
y1_nodes  = par.y0 * exp( mu_log_g1 + sqrt(2) * par.sigma_g * xi );
weights   = wi / sqrt(pi);             % normalised probability weights

fprintf('Gauss-Hermite quadrature (%d nodes):\n', nq);
fprintf('  %6s  %10s  %10s\n', 'j', 'y1', 'weight');
for j = 1:nq
    fprintf('  %6d  %10.6f  %10.6f\n', j, y1_nodes(j), weights(j));
end
fprintf('  Sum of weights = %.10f\n\n', sum(weights));

%% ======================== SOLVE PERIOD 0 ==============================
%  Find optimal bond issuance b1 from the Euler equation.

fprintf('============================================================\n');
fprintf('  Solving for optimal b1 (Period 0 Euler equation)\n');
fprintf('============================================================\n\n');

b1_init = 0.35;             % initial guess

opts0 = optimoptions('fsolve', ...
    'Display',      'iter', ...
    'TolFun',       1e-10, ...
    'TolX',         1e-10);

[b1_star, fval0, exitflag0] = fsolve( ...
    @(b1) period0_euler(b1, par, csv, y1_nodes, weights), ...
    b1_init, opts0);

fprintf('\n  b1*           = %.6f\n', b1_star);
fprintf('  Euler residual = %.2e\n', fval0);
fprintf('  Exit flag      = %d\n\n', exitflag0);

%% =================== RECOVER FULL EQUILIBRIUM =========================

fprintf('============================================================\n');
fprintf('  Full Equilibrium\n');
fprintf('============================================================\n\n');

% --- Solve Period 1 for each state ---
D_vals  = zeros(nq, 1);
sol1    = cell(nq, 1);
x0      = [];
for j = 1:nq
    sol1{j}  = solve_period1(y1_nodes(j), b1_star, par, csv, x0);
    D_vals(j) = sol1{j}.D;
    x0 = [sol1{j}.D; sol1{j}.omegabar; sol1{j}.Mf];
end

% --- Bond price ---
q0_star = (1 / par.Rstar) * sum( weights .* (1 - D_vals / b1_star) );

% --- Period 0 allocations (frictionless intermediation) ---
md0 = par.y0 - par.b0 + q0_star * b1_star;
mf0 = md0 * ( (1 - par.alpha) / (par.alpha * par.Rstar) )^par.sigma;
C0  = ( par.alpha * md0^par.eta ...
      + (1 - par.alpha) * mf0^par.eta )^(1/par.eta);

% Sovereign spread:  yield = R*/q0,  spread = yield - R*
yield0  = par.Rstar / q0_star;
spread_bp = (yield0 - par.Rstar) * 10000;

% --- Display Period 0 ---
fprintf('--- Period 0 ---\n');
fprintf('  b1          = %10.6f\n', b1_star);
fprintf('  q0          = %10.6f\n', q0_star);
fprintf('  md0         = %10.6f\n', md0);
fprintf('  mf0         = %10.6f\n', mf0);
fprintf('  C0          = %10.6f\n', C0);
fprintf('  Yield (R/q) = %10.6f\n', yield0);
fprintf('  Spread (bp) = %10.1f\n\n', spread_bp);

% --- Display Period 1 by state ---
fprintf('--- Period 1 (by state) ---\n');
fprintf('  %8s %8s %8s %8s %8s %8s %8s %8s %8s\n', ...
    'y1', 'D', 'D/b1', 'wbar', 'Mf', 'pm', 'mf', 'C1', 'Z/R*');
fprintf('  %s\n', repmat('-', 1, 80));
for j = 1:nq
    s = sol1{j};
    fprintf('  %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n', ...
        y1_nodes(j), s.D, s.D/b1_star, s.omegabar, s.Mf, s.pm, s.mf, s.C1, s.Z_Rstar);
end

% --- Summary statistics ---
E_D      = sum(weights .* D_vals);
E_haircut = E_D / b1_star;
C1_vals   = cellfun(@(s) s.C1, sol1);

fprintf('\n--- Summary ---\n');
fprintf('  E[D]             = %10.6f\n', E_D);
fprintf('  E[D/b1] (haircut)= %10.6f\n', E_haircut);
fprintf('  E[1-D/b1] (recov)= %10.6f\n', 1 - E_haircut);
fprintf('  E[C1]            = %10.6f\n', sum(weights .* C1_vals));

fprintf('\n============================================================\n');
fprintf('  Solver complete.\n');
fprintf('============================================================\n');
