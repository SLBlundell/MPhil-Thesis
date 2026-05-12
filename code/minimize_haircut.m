%% MINIMIZE_HAIRCUT
%  Grid search over (gamma, sigma_w, mu) to find the parameter
%  combination that minimises the expected haircut E[D/b1].
%
%  Re-uses the V0-maximising competitive-equilibrium solver pattern from
%  solve_ge_model_value_max.m / comparative_static_surfaces.m:
%
%    For each (gamma, sigma_w, mu):
%      1. Build CSV function handles
%      2. Multi-start fminbnd on V0(b1) -> b1*
%      3. Recover Period-1 default policy at b1* across y1 nodes
%      4. Record E[D/b1*]
%
%  At the end, the (gamma, sigma_w, mu) triple with the smallest
%  E[D/b1] is reported, alongside the top-10 leaderboard and a
%  diagnostic re-solve of the best point.
%
%  Dependencies: csv_functions.m, gauss_hermite.m, solve_period1.m
%
%  Output: minimize_haircut_results.mat in the current directory.

clear; clc;
fprintf('============================================================\n');
fprintf('  Haircut-minimising parameter search over (gamma, sigma_w, mu)\n');
fprintf('============================================================\n\n');

%% ====================== GRID SETTINGS ===============================
ngamma   = 8;     gamma_lo   = 0.15;   gamma_hi   = 0.50;
nsigma   = 8;     sigma_w_lo = 0.20;   sigma_w_hi = 0.40;
nmu      = 8;     mu_lo      = 0.20;   mu_hi      = 0.50;

b1_max   = 2.0;
n_coarse = 15;    % multi-start b1 grid

gamma_grid   = linspace(gamma_lo,   gamma_hi,   ngamma);
sigma_w_grid = linspace(sigma_w_lo, sigma_w_hi, nsigma);
mu_grid      = linspace(mu_lo,      mu_hi,      nmu);

[GA, SW, MU] = ndgrid(gamma_grid, sigma_w_grid, mu_grid);
N            = numel(GA);

fprintf('Grid: %d x %d x %d = %d points\n', ngamma, nsigma, nmu, N);
fprintf('  gamma   in [%.2f, %.2f]\n', gamma_lo,   gamma_hi);
fprintf('  sigma_w in [%.2f, %.2f]\n', sigma_w_lo, sigma_w_hi);
fprintf('  mu      in [%.2f, %.2f]\n\n', mu_lo,    mu_hi);

%% ================== BASELINE PARAMETERS =============================
%  All non-swept parameters match solve_ge_model_value_max.m.
par_base.beta    = 0.815;
par_base.sigma_u = 2;
par_base.alpha   = 0.7;
par_base.eta     = -1;
par_base.sigma   = 1/(1 - par_base.eta);
par_base.nbar_base = 0.065;
par_base.Rstar   = 1.104;
par_base.y0      = 1.00;
par_base.b0      = 0.18;
par_base.rho     = 0.60;
par_base.mu_g    = 1.20;
par_base.sigma_g = 0.15;
par_base.g0      = 0.95;

%% =============== Y1 NODES (Gauss-Hermite) ===========================
nq          = 7;
[xi, wi]    = gauss_hermite(nq);
mu_log_g1   = (1 - par_base.rho) * log(par_base.mu_g) + par_base.rho * log(par_base.g0);
y1_nodes    = par_base.y0 * exp(mu_log_g1 + sqrt(2) * par_base.sigma_g * xi);
weights     = wi / sqrt(pi);

%% ============= ALLOCATE OUTPUT ARRAYS ===============================
EH   = nan(N, 1);
B1   = nan(N, 1);
SPRD = nan(N, 1);
OK   = false(N, 1);

%% =============== PARALLEL POOL ======================================
use_par = false;
try
    p = gcp('nocreate');
    if isempty(p)
        fprintf('Starting parallel pool...\n');
        parpool('local');
    end
    use_par = true;
catch
    fprintf('Parallel Computing Toolbox unavailable -- running serial.\n');
end

%% ======================== MAIN LOOP =================================
fprintf('Solving %d grid points...\n', N);
t0 = tic;

if use_par
    parfor k = 1:N
        [EH(k), B1(k), SPRD(k), OK(k)] = solve_one_point( ...
            GA(k), SW(k), MU(k), par_base, y1_nodes, weights, b1_max, n_coarse);
    end
else
    for k = 1:N
        [EH(k), B1(k), SPRD(k), OK(k)] = solve_one_point( ...
            GA(k), SW(k), MU(k), par_base, y1_nodes, weights, b1_max, n_coarse);
        if mod(k, max(1, floor(N/20))) == 0
            fprintf('  %3d%% done\n', round(100*k/N));
        end
    end
end

fprintf('Completed in %.1f s (%d/%d successful).\n\n', toc(t0), sum(OK), N);

%% ====================== FIND MINIMUM ================================
EH_search = EH;
EH_search(~OK) = Inf;
[E_haircut_min, k_min] = min(EH_search);

if ~isfinite(E_haircut_min)
    error('No grid point solved successfully.');
end

gamma_star   = GA(k_min);
sigma_w_star = SW(k_min);
mu_star      = MU(k_min);
b1_at_min    = B1(k_min);
sprd_at_min  = SPRD(k_min);

fprintf('============================================================\n');
fprintf('  Haircut-minimising parameter combination\n');
fprintf('============================================================\n');
fprintf('  gamma*    = %.4f\n', gamma_star);
fprintf('  sigma_w*  = %.4f\n', sigma_w_star);
fprintf('  mu*       = %.4f\n', mu_star);
fprintf('  --------------------------------\n');
fprintf('  E[D/b1]   = %.6f   (minimum on grid)\n', E_haircut_min);
fprintf('  b1*       = %.6f\n', b1_at_min);
fprintf('  Spread    = %.1f bp\n\n', sprd_at_min);

%% ====================== LEADERBOARD =================================
[EH_sorted, idx_sorted] = sort(EH_search, 'ascend');
nshow = min(10, sum(OK));
fprintf('Top-%d lowest E[D/b1] on grid:\n', nshow);
fprintf('  %4s  %8s  %8s  %8s  %12s  %10s  %10s\n', ...
    'rank', 'gamma', 'sigma_w', 'mu', 'E[D/b1]', 'b1*', 'Spread bp');
fprintf('  %s\n', repmat('-', 1, 72));
for r = 1:nshow
    kk = idx_sorted(r);
    fprintf('  %4d  %8.4f  %8.4f  %8.4f  %12.6f  %10.4f  %10.1f\n', ...
        r, GA(kk), SW(kk), MU(kk), EH_sorted(r), B1(kk), SPRD(kk));
end
fprintf('\n');

%% =============== EDGE-OF-GRID WARNING ===============================
on_edge = (gamma_star   == gamma_lo   || gamma_star   == gamma_hi)   || ...
          (sigma_w_star == sigma_w_lo || sigma_w_star == sigma_w_hi) || ...
          (mu_star      == mu_lo      || mu_star      == mu_hi);
if on_edge
    fprintf('NOTE: Minimum is on the grid boundary -- consider widening the search range.\n\n');
end

%% =============== DIAGNOSTIC RE-SOLVE AT MINIMUM =====================
fprintf('============================================================\n');
fprintf('  Diagnostic re-solve at the minimum\n');
fprintf('============================================================\n');

par_star         = par_base;
par_star.gamma   = gamma_star;
par_star.sigma_w = sigma_w_star;
par_star.mu      = mu_star;
par_star.A_ref   = par_star.nbar_base + par_star.gamma * par_star.b0;
par_star.nbar    = par_star.A_ref - par_star.gamma * par_star.b0;
csv_star         = csv_functions(par_star.sigma_w, par_star.mu);

ce_star = solve_b1_CE(par_star, csv_star, y1_nodes, weights, b1_max, n_coarse);

fprintf('  b1*           = %.6f\n', ce_star.b1);
fprintf('  q0            = %.6f\n', ce_star.q0);
fprintf('  E[D/b1]       = %.6f\n', ce_star.E_haircut);
fprintf('  Spread        = %.1f bp\n\n', ce_star.spread_bp);

fprintf('Per-state default policy at the minimum:\n');
fprintf('  %8s %10s %10s\n', 'y1', 'D*', 'D*/b1');
fprintf('  %s\n', repmat('-', 1, 32));
for j = 1:nq
    sj = solve_period1(y1_nodes(j), ce_star.b1, par_star, csv_star, []);
    fprintf('  %8.4f %10.4f %10.4f\n', y1_nodes(j), sj.D, sj.D / ce_star.b1);
end
fprintf('\n');

%% ====================== SAVE RESULTS ================================
EH3   = reshape(EH,   size(GA));
B13   = reshape(B1,   size(GA));
SPRD3 = reshape(SPRD, size(GA));
OK3   = reshape(OK,   size(GA));

save('minimize_haircut_results.mat', ...
    'gamma_grid','sigma_w_grid','mu_grid','GA','SW','MU', ...
    'EH3','B13','SPRD3','OK3','par_base', ...
    'gamma_star','sigma_w_star','mu_star','E_haircut_min', ...
    'b1_at_min','sprd_at_min','ce_star');

fprintf('Results saved to minimize_haircut_results.mat\n');
fprintf('============================================================\n');


%% ====================================================================
%  ============================ LOCAL FUNCTIONS ========================
%% ====================================================================

function [eh, b1, sp, ok] = solve_one_point(ga, sw, mu, par_base, ...
                                            y1n, w, b1max, n_coarse)
    par         = par_base;
    par.gamma   = ga;
    par.sigma_w = sw;
    par.mu      = mu;
    par.A_ref   = par.nbar_base + par.gamma * par.b0;
    par.nbar    = par.A_ref - par.gamma * par.b0;

    eh = NaN; b1 = NaN; sp = NaN; ok = false;
    try
        csv_k = csv_functions(par.sigma_w, par.mu);
        ce    = solve_b1_CE(par, csv_k, y1n, w, b1max, n_coarse);
        eh    = ce.E_haircut;
        b1    = ce.b1;
        sp    = ce.spread_bp;
        ok    = isfinite(eh);
    catch
        % keep NaNs
    end
end

function ce = solve_b1_CE(par, csv, y1n, w, b1max, n_coarse)
% Multi-start V0 maximisation:
%   1. Coarse grid evaluation of V0 across [eps, b1max]
%   2. fminbnd refinement in the neighbourhood of the coarse arg-max

    b1_coarse = linspace(1e-3, b1max, n_coarse);
    V_coarse  = -inf(size(b1_coarse));
    for i = 1:n_coarse
        try %#ok<TRYNC>
            V_coarse(i) = V0(b1_coarse(i), par, csv, y1n, w);
        end
    end

    [~, idx] = max(V_coarse);
    lo = b1_coarse(max(idx-1, 1));
    hi = b1_coarse(min(idx+1, n_coarse));

    opts          = optimset('Display','off','TolX',1e-7);
    [b1_star, ~]  = fminbnd(@(b) -V0(b, par, csv, y1n, w), lo, hi, opts);

    ce = compute_moments(b1_star, par, csv, y1n, w);
end

function v = V0(b1, par, csv, y1n, w)
    nq = numel(y1n);
    Dv = zeros(nq,1); C1v = zeros(nq,1);
    for j = 1:nq
        s      = solve_period1(y1n(j), b1, par, csv, []);
        Dv(j)  = s.D;
        C1v(j) = s.C1;
    end
    q0  = (1/par.Rstar) * sum(w .* (1 - Dv/b1));
    md0 = par.y0 - par.b0 + q0*b1;
    if md0 <= 0
        v = -1e10; return
    end
    mf0 = md0 * ((1-par.alpha)/(par.alpha*par.Rstar))^par.sigma;
    C0  = (par.alpha*md0^par.eta + (1-par.alpha)*mf0^par.eta)^(1/par.eta);
    u0  = C0^(1-par.sigma_u) / (1-par.sigma_u);
    Eu1 = sum(w .* (C1v.^(1-par.sigma_u)/(1-par.sigma_u)));
    v   = u0 + par.beta * Eu1;
end

function out = compute_moments(b1, par, csv, y1n, w)
    nq = numel(y1n);
    Dv = zeros(nq,1);
    Hv = zeros(nq,1);
    for j = 1:nq
        s     = solve_period1(y1n(j), b1, par, csv, []);
        Dv(j) = s.D;
        Hv(j) = s.D / b1;
    end
    q0           = (1/par.Rstar) * sum(w .* (1 - Dv/b1));
    out.b1        = b1;
    out.q0        = q0;
    out.E_haircut = sum(w .* Hv);
    out.spread_bp = (par.Rstar/q0 - par.Rstar) * 10000;
end
