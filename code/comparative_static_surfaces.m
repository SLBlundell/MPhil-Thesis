%% COMPARATIVE_STATIC_SURFACES (CE-only, V0-max with multi-start)
%  Generates 3-D comparative-static surfaces over (sigma_omega, gamma)
%  for the four CE objects in Section 8.3:
%
%    (i)   Expected haircut          E[D/b1]
%    (ii)  Sovereign spread          R*/q0 - R*           (basis points, capped)
%    (iv)  External finance premium  E[Z1/R*]
%    (vi)  Import compression        E[mf | D=0] - E[mf]   (>= 0)
%
%  ----------------------------------------------------------------------
%  CE solver: multi-start grid search + fminbnd refinement on V0.
%  Mirrors solve_ge_model_value_max.m but adds a coarse global scan to
%  defend against local maxima of V0(b1).
%
%  Conceptual note: V0 max with q0(b1) computed inside is technically
%  the planner's b1 -- the strict CE differs by the q0'(b1)*b1 term in
%  the Period 0 FOC.  Matches the choice in the main solver script.
%  ----------------------------------------------------------------------
%
%  Workflow:
%    1. Diagnostic solve at baseline (sigma_w=0.25, gamma=0.32).
%    2. Parallel grid solve over restricted (sigma_omega, gamma).
%    3. Save .mat and produce 2x2 combined PDF + individual surface PDFs
%       + diagnostic b1 and corner-count surfaces.
%
%  Dependencies (in your code/ directory or on path):
%      csv_functions.m, solve_period1.m, gauss_hermite.m

clear; clc; close all;

%% ====================== USER SETTINGS ===============================
nsigma     = 12;
ngamma     = 12;
sigma_w_lo = 0.20;     sigma_w_hi = 0.40;
gamma_lo   = 0.15;     gamma_hi   = 0.50;
b1_max     = 2.0;
spread_cap = 20000;                      % bumped from 5000

n_coarse   = 15;                         % b1 grid for multi-start

out_dir    = 'cs_output';
save_individual_panels = true;

%% ================== BASELINE PARAMETERS =============================
par_base.beta    = 0.815;
par_base.sigma_u = 2;
par_base.alpha   = 0.7;
par_base.eta     = -1;
par_base.sigma   = 1/(1 - par_base.eta);
par_base.nbar    = 0.065;
par_base.mu      = 0.400;
par_base.Rstar   = 1.104;
par_base.y0      = 1.00;
par_base.b0      = 0.18;
par_base.rho     = 0.60;
par_base.mu_g    = 1.20;
par_base.sigma_g = 0.20;
par_base.g0      = 0.95;

sigma_w_base = 0.25;
gamma_base   = 0.32;

%% ====================== GRID =========================================
sigma_w_grid = linspace(sigma_w_lo, sigma_w_hi, nsigma);
gamma_grid   = linspace(gamma_lo,   gamma_hi,   ngamma);
[SW, GA]     = meshgrid(sigma_w_grid, gamma_grid);
N            = numel(SW);

%% =============== Y1 NODES ===========================================
nq          = 7;
[xi, wi]    = gauss_hermite(nq);
mu_log_g1   = (1-par_base.rho)*log(par_base.mu_g) + par_base.rho*log(par_base.g0);
y1_nodes    = par_base.y0 * exp(mu_log_g1 + sqrt(2)*par_base.sigma_g*xi);
weights     = wi / sqrt(pi);

%% ============= DIAGNOSTIC BLOCK =====================================
fprintf('\n============================================================\n');
fprintf('  DIAGNOSTIC: Interior-default check at baseline\n');
fprintf('  (sigma_omega = %.2f, gamma = %.2f)\n', sigma_w_base, gamma_base);
fprintf('============================================================\n');

par_diag         = par_base;
par_diag.sigma_w = sigma_w_base;
par_diag.gamma   = gamma_base;
csv_diag         = csv_functions(par_diag.sigma_w, par_diag.mu);

ce_diag = solve_b1_CE(par_diag, csv_diag, y1_nodes, weights, b1_max, n_coarse);

fprintf('\n  CE equilibrium:\n');
fprintf('    b1*           = %.6f\n', ce_diag.b1);
fprintf('    q0            = %.6f\n', ce_diag.q0);
fprintf('    E[D/b1]       = %.6f\n', ce_diag.E_haircut);
fprintf('    Spread        = %.1f bp\n', ce_diag.spread_bp);
fprintf('    E[Z/R*]       = %.6f\n', ce_diag.E_ZR);
fprintf('    E[mf]         = %.6f\n', ce_diag.E_mf);
fprintf('    E[mf | D=0]   = %.6f\n', ce_diag.E_mf_nd);
fprintf('    Imp. comp.    = %.6f\n\n', ce_diag.import_comp);

fprintf('  Per-state default policy:\n');
fprintf('  %6s %8s %8s %14s %12s\n', 'y1', 'D*', 'D*/b1', 'Status', '|dC1/dD|');
fprintf('  %s\n', repmat('-', 1, 60));

n_interior = 0; n_corner = 0;
for j = 1:nq
    y1   = y1_nodes(j);
    sj   = solve_period1(y1, ce_diag.b1, par_diag, csv_diag, []);
    Dj   = sj.D;
    haircut_j = Dj / ce_diag.b1;

    at_lower = (Dj < 1e-4);
    at_upper = (abs(Dj - ce_diag.b1) < 1e-4 * ce_diag.b1);

    if at_lower
        status = 'lower-corner';   dC1dD = NaN;
        n_corner = n_corner + 1;
    elseif at_upper
        status = 'upper-corner';   dC1dD = NaN;
        n_corner = n_corner + 1;
    else
        eps_D = 1e-4 * ce_diag.b1;
        Cp = compute_C1_at_D(Dj + eps_D, y1, ce_diag.b1, par_diag, csv_diag);
        Cm = compute_C1_at_D(Dj - eps_D, y1, ce_diag.b1, par_diag, csv_diag);
        if isnan(Cp) || isnan(Cm)
            status = 'interior?';   dC1dD = NaN;
        else
            dC1dD = (Cp - Cm) / (2*eps_D);
            if abs(dC1dD) < 1e-3
                status = 'interior(OK)';
            else
                status = 'interior(?)';
            end
        end
        n_interior = n_interior + 1;
    end

    fprintf('  %6.4f %8.4f %8.4f %14s %12.2e\n', ...
        y1, Dj, haircut_j, status, dC1dD);
end

fprintf('\n  Summary: %d/%d interior, %d/%d corner.\n', ...
    n_interior, nq, n_corner, nq);
fprintf('\n');

%% ============= ALLOCATE OUTPUT ARRAYS ===============================
EH       = nan(N,1);
SPRD     = nan(N,1);
EFP      = nan(N,1);
IMPCOMP  = nan(N,1);
B1       = nan(N,1);
NCORNER  = nan(N,1);
OK       = false(N,1);

%% =============== PARALLEL POOL =====================================
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

%% ====================== MAIN LOOP ===================================
fprintf('Solving %d (sigma_omega, gamma) grid points...\n', N);
t0 = tic;

if use_par
    parfor k = 1:N
        [EH(k), SPRD(k), EFP(k), IMPCOMP(k), B1(k), NCORNER(k), OK(k)] = ...
            solve_one_point(SW(k), GA(k), par_base, ...
                            y1_nodes, weights, b1_max, n_coarse);
    end
else
    for k = 1:N
        [EH(k), SPRD(k), EFP(k), IMPCOMP(k), B1(k), NCORNER(k), OK(k)] = ...
            solve_one_point(SW(k), GA(k), par_base, ...
                            y1_nodes, weights, b1_max, n_coarse);
    end
end

fprintf('Completed in %.1f s (%d/%d successful).\n', toc(t0), sum(OK), N);
fprintf('Mean corner states per grid point: %.2f / %d\n', ...
    mean(NCORNER(OK)), nq);

%% ============= RESHAPE & APPLY SPREAD CAP ===========================
EH       = reshape(EH,      size(SW));
SPRD     = reshape(SPRD,    size(SW));
EFP      = reshape(EFP,     size(SW));
IMPCOMP  = reshape(IMPCOMP, size(SW));
B1       = reshape(B1,      size(SW));
NCORNER  = reshape(NCORNER, size(SW));

SPRD_capped = min(SPRD, spread_cap);

%% ====================== SAVE & PLOT =================================
if ~exist(out_dir,'dir'); mkdir(out_dir); end

save(fullfile(out_dir,'cs_results.mat'), ...
    'sigma_w_grid','gamma_grid','SW','GA', ...
    'EH','SPRD','SPRD_capped','EFP','IMPCOMP','B1','NCORNER','OK', ...
    'par_base','spread_cap');

plot_surfaces(SW, GA, EH, SPRD_capped, EFP, IMPCOMP, B1, NCORNER, ...
              spread_cap, nq, out_dir, save_individual_panels);

fprintf('All output written to %s/\n', out_dir);

%% ====================================================================
%  ============================ LOCAL FUNCTIONS ========================
%% ====================================================================

function [eh, sp, efp, imp, b1, nc, ok] = solve_one_point(sw, ga, par_base, ...
                                            y1n, w, b1max, n_coarse)
    par         = par_base;
    par.sigma_w = sw;
    par.gamma   = ga;
    csv_k       = csv_functions(par.sigma_w, par.mu);
    eh = NaN; sp = NaN; efp = NaN; imp = NaN;
    b1 = NaN; nc = NaN; ok = false;
    try
        ce  = solve_b1_CE(par, csv_k, y1n, w, b1max, n_coarse);
        eh  = ce.E_haircut;
        sp  = ce.spread_bp;
        efp = ce.E_ZR;
        imp = ce.import_comp;
        b1  = ce.b1;
        nc  = ce.n_corner;
        ok  = true;
    catch
        % keep NaNs
    end
end

function ce = solve_b1_CE(par, csv, y1n, w, b1max, n_coarse)
% Multi-start V0 max:
%   1. Coarse grid evaluation of V0 across [eps, b1max]
%   2. fminbnd refinement in neighbourhood of global max
%
% Defends against local maxima of V0(b1), which arise when the Period 1
% policy function induces a non-concave value function (typical in
% partial-default models).

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
    nq        = numel(y1n);
    Dv        = zeros(nq,1);
    Zv        = zeros(nq,1);
    Hv        = zeros(nq,1);
    mf_eq     = zeros(nq,1);
    mf_nd     = zeros(nq,1);
    is_corner = false(nq,1);

    for j = 1:nq
        % Equilibrium
        s         = solve_period1(y1n(j), b1, par, csv, []);
        Dv(j)     = s.D;
        Zv(j)     = s.Z_Rstar;
        Hv(j)     = s.D / b1;
        mf_eq(j)  = s.mf;
        is_corner(j) = (s.D < 1e-4) || (abs(s.D - b1) < 1e-4*b1);

        % Counterfactual: D=0 forced at same b1
        mf_nd(j) = compute_mf_at_D(0, y1n(j), b1, par, csv);
    end

    q0       = (1/par.Rstar) * sum(w .* (1 - Dv/b1));

    % Robust expectation for counterfactual
    valid_nd = ~isnan(mf_nd);
    if any(valid_nd)
        E_mf_nd = sum(w(valid_nd) .* mf_nd(valid_nd)) / sum(w(valid_nd));
    else
        E_mf_nd = NaN;
    end
    E_mf = sum(w .* mf_eq);

    out.b1          = b1;
    out.q0          = q0;
    out.E_haircut   = sum(w .* Hv);
    out.spread_bp   = (par.Rstar/q0 - par.Rstar) * 10000;
    out.E_ZR        = sum(w .* Zv);
    out.E_mf        = E_mf;
    out.E_mf_nd     = E_mf_nd;
    out.import_comp = E_mf_nd - E_mf;
    out.n_corner    = sum(is_corner);
end

%% ----------------- INNER-SOLVE HELPERS ------------------------------

function mf = compute_mf_at_D(D, y1, b1, par, csv)
% Imports mf at fixed D, holding b1 and y1 fixed.  Used for the
% no-default counterfactual.

    par.y1 = y1; par.b1 = b1;
    N           = par.nbar + par.gamma * (b1 - D);
    income_base = y1 - (b1 - D);
    if N <= 0 || income_base <= 0
        mf = NaN; return
    end

    opts = optimoptions('fsolve','Display','off','TolFun',1e-8);
    [log_wb_star, ~, flag] = fsolve(@(x) mc_resid(x, N, income_base, par, csv), ...
                                    log(0.45), opts);
    if flag <= 0
        mf = NaN; return
    end

    omegabar = exp(log_wb_star);
    Gam      = csv.Gamma(omegabar);
    Gp       = csv.GammaPrime(omegabar);
    muG      = par.mu * csv.G(omegabar);
    Mf       = N * (1 + Gp*omegabar/(1-Gam));
    mf       = Mf * (1 - muG);
end

function C1 = compute_C1_at_D(D, y1, b1, par, csv)
% C1 at fixed D.  Used for diagnostic FOC check.

    par.y1 = y1; par.b1 = b1;
    N           = par.nbar + par.gamma * (b1 - D);
    income_base = y1 - (b1 - D);
    if N <= 0
        C1 = NaN; return
    end

    opts = optimoptions('fsolve','Display','off','TolFun',1e-8);
    [log_wb_star, ~, flag] = fsolve(@(x) mc_resid(x, N, income_base, par, csv), ...
                                    log(0.45), opts);
    if flag <= 0
        C1 = NaN; return
    end

    omegabar = exp(log_wb_star);
    Gam      = csv.Gamma(omegabar);
    Gp       = csv.GammaPrime(omegabar);
    Psi      = csv.Psi(omegabar);
    muG      = par.mu * csv.G(omegabar);
    Mf       = N * (1 + Gp*omegabar/(1-Gam));
    ell      = 1 - N/Mf;
    pm       = par.Rstar * ell / Psi;
    mf       = Mf * (1 - muG);
    md       = mf * (par.alpha*pm/(1-par.alpha))^par.sigma;
    C1       = (par.alpha*md^par.eta + (1-par.alpha)*mf^par.eta)^(1/par.eta);
end

function R = mc_resid(log_wb, N, income_base, par, csv)
    omegabar = exp(log_wb);
    Gam = csv.Gamma(omegabar);
    if Gam >= 1, R = 1e6; return; end
    Gp  = csv.GammaPrime(omegabar);
    Psi = csv.Psi(omegabar);
    muG = par.mu * csv.G(omegabar);
    Mf  = N * (1 + Gp*omegabar/(1-Gam));
    ell = 1 - N/Mf;
    pm  = par.Rstar * ell / Psi;
    mf  = Mf * (1 - muG);
    LHS = mf * ((par.alpha*pm/(1-par.alpha))^par.sigma + pm);
    RHS = income_base + (1 - Gam) * pm * Mf;
    R   = LHS - RHS;
end

%% ----------------- PLOTTING -----------------------------------------

function plot_surfaces(SW, GA, EH, SPRD, EFP, IMPCOMP, B1, NCORNER, ...
                       spread_cap, nq, outdir, save_each)

    panels = {
        EH,       'Expected Haircut $E[D/b_1]$',                       'haircut';
        SPRD,     sprintf('Sovereign Spread (bp, capped at %d)', spread_cap), 'spread';
        EFP,      'External Finance Premium $E[Z_1/R^*]$',             'efp';
        IMPCOMP,  'Import Compression $E[m^f | D{=}0] - E[m^f]$',      'import_comp';
    };

    %% Combined 2x2 figure ---------------------------------------------
    fig = figure('Color','w','Units','normalized','Position',[.05 .05 .85 .85]);
    for i = 1:size(panels,1)
        subplot(2,2,i);
        surf(SW, GA, panels{i,1}, 'EdgeAlpha', 0.25);
        shading interp; colormap(viridis_cmap());
        xlabel('$\sigma_\omega$', 'Interpreter','latex','FontSize',13);
        ylabel('$\gamma$',        'Interpreter','latex','FontSize',13);
        title(panels{i,2},        'Interpreter','latex','FontSize',14);
        view(135,25); grid on; box on; axis tight;
        set(gca,'TickLabelInterpreter','latex','FontSize',11);
    end
    sgt = sgtitle('Comparative Statics over $(\sigma_\omega, \gamma)$');
    set(sgt,'Interpreter','latex','FontSize',16);

    exportgraphics(fig, fullfile(outdir,'cs_surfaces.pdf'), 'ContentType','vector');
    exportgraphics(fig, fullfile(outdir,'cs_surfaces.png'), 'Resolution',300);

    %% Individual panels ------------------------------------------------
    if save_each
        for i = 1:size(panels,1)
            f = figure('Color','w','Units','normalized','Position',[.2 .2 .42 .55]);
            surf(SW, GA, panels{i,1}, 'EdgeAlpha', 0.25);
            shading interp; colormap(viridis_cmap());
            xlabel('$\sigma_\omega$','Interpreter','latex','FontSize',14);
            ylabel('$\gamma$',       'Interpreter','latex','FontSize',14);
            title(panels{i,2},       'Interpreter','latex','FontSize',15);
            view(135,25); grid on; box on; axis tight;
            set(gca,'TickLabelInterpreter','latex','FontSize',12);
            exportgraphics(f, fullfile(outdir, ['cs_' panels{i,3} '.pdf']), ...
                           'ContentType','vector');
            close(f);
        end
    end

    %% Diagnostic surface: b1 ------------------------------------------
    f2 = figure('Color','w','Units','normalized','Position',[.2 .2 .42 .55]);
    surf(SW, GA, B1, 'EdgeAlpha', 0.25);
    shading interp; colormap(viridis_cmap());
    xlabel('$\sigma_\omega$','Interpreter','latex','FontSize',14);
    ylabel('$\gamma$',       'Interpreter','latex','FontSize',14);
    title('Equilibrium Debt Issuance $b_1^*$','Interpreter','latex','FontSize',15);
    view(135,25); grid on; box on; axis tight;
    set(gca,'TickLabelInterpreter','latex','FontSize',12);
    exportgraphics(f2, fullfile(outdir,'cs_b1.pdf'), 'ContentType','vector');
    close(f2);

    %% Diagnostic surface: corner-state count -------------------------
    f3 = figure('Color','w','Units','normalized','Position',[.2 .2 .42 .55]);
    surf(SW, GA, NCORNER, 'EdgeAlpha', 0.25);
    shading interp; colormap(viridis_cmap());
    xlabel('$\sigma_\omega$','Interpreter','latex','FontSize',14);
    ylabel('$\gamma$',       'Interpreter','latex','FontSize',14);
    title(sprintf('Corner States per Grid Point (out of %d)', nq), ...
          'Interpreter','latex','FontSize',15);
    view(135,25); grid on; box on; axis tight;
    set(gca,'TickLabelInterpreter','latex','FontSize',12);
    caxis([0, nq]);
    exportgraphics(f3, fullfile(outdir,'cs_n_corner.pdf'), 'ContentType','vector');
    close(f3);
end

function cmap = viridis_cmap()
    base = [
        0.2670 0.0049 0.3294
        0.2810 0.1659 0.4729
        0.2530 0.2650 0.5290
        0.2070 0.3717 0.5530
        0.1640 0.4710 0.5580
        0.1280 0.5680 0.5510
        0.1340 0.6580 0.5170
        0.2660 0.7480 0.4400
        0.4770 0.8210 0.3180
        0.7410 0.8730 0.1500
        0.9930 0.9060 0.1440
    ];
    cmap = interp1(linspace(0,1,size(base,1)), base, linspace(0,1,256));
end