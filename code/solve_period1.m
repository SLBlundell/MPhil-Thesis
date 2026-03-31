function sol = solve_period1(y1, b1, par, csv, ~)
    par.y1 = y1;
    par.b1 = b1;
    
    % --- Outer Optimisation: Maximise Utility over D ---
    % fminbnd handles the bounds [0, b1]
    % testing both corner solutions and interior points.
    opts_fmin = optimset('Display', 'off', 'TolX', 1e-8);
    [D_star, ~] = fminbnd(@(D) neg_utility(D, par, csv), 0, b1, opts_fmin);
    
    % --- Final solve at optimum to recover variables ---
    sol = solve_inner(D_star, par, csv);
    sol.D = D_star;
    
    % --- Compute dpm/dN via finite differences ---
    % This avoids the analytical "denominator trap"
    N1 = par.nbar + par.gamma * (b1 - D_star);
    eps_N = 1e-5 * max(abs(N1), 1e-4);
    
    sp = solve_inner_at_N(N1 + eps_N, par.y1 - (par.b1 - D_star), par, csv);
    sm = solve_inner_at_N(N1 - eps_N, par.y1 - (par.b1 - D_star), par, csv);
    
    if ~isempty(sp) && ~isempty(sm)
        sol.dpm_dN = (sp.pm - sm.pm) / (2 * eps_N);
    else
        sol.dpm_dN = 0; % Fallback
    end
    sol.exitflag = 1;
end

function val = neg_utility(D, par, csv)
    s = solve_inner(D, par, csv);
    if isempty(s) || s.C1 <= 0
        val = 1e10; % Penalty for infeasible region
    else
        val = -s.C1^(1 - par.sigma_u) / (1 - par.sigma_u);
    end
end

function sol = solve_inner(D, par, csv)
    % N is predetermined given D
    N = par.nbar + par.gamma * (par.b1 - D);
    disposable_income = par.y1 - (par.b1 - D);
    sol = solve_inner_at_N(N, disposable_income, par, csv);
end

function sol = solve_inner_at_N(N, disposable_income, par, csv)
    if N <= 0
        sol = [];
        return
    end
    
    % --- 1D Root Finding over log(omegabar) ---
    opts = optimoptions('fsolve', 'Display', 'off', 'TolFun', 1e-10);
    [log_wb_star, ~, flag] = fsolve( ...
        @(log_wb) market_clearing_resid(log_wb, N, disposable_income, par, csv), ...
        log(0.45), opts);
        
    if flag <= 0
        sol = [];
        return
    end
    
    % --- Recover Full Allocations ---
    omegabar = exp(log_wb_star);
    Gam = csv.Gamma(omegabar);
    Gp  = csv.GammaPrime(omegabar);
    
    % Mf is pinned down by the leverage equation
    Mf = N * (1 + Gp * omegabar / (1 - Gam));
    ell = 1 - N / Mf;
    
    if Mf <= N || ell <= 0 || ell >= 1
        sol = [];
        return
    end
    
    Psi = csv.Psi(omegabar);
    muG = par.mu * csv.G(omegabar);
    
    sol.omegabar = omegabar;
    sol.Mf       = Mf;
    sol.N        = N;
    sol.ell      = ell;
    sol.pm       = par.Rstar * ell / Psi;
    sol.mf       = Mf * (1 - muG);
    sol.md       = sol.mf * (par.alpha * sol.pm / (1 - par.alpha))^par.sigma;
    sol.C1       = (par.alpha * sol.md^par.eta + (1 - par.alpha) * sol.mf^par.eta)^(1/par.eta);
    sol.Pi       = (1 - Gam) * sol.pm * Mf;
    sol.lambda1  = sol.C1^(-par.sigma_u) * par.alpha * (sol.C1 / sol.md)^(1 - par.eta);
    sol.Z_Rstar  = omegabar / Psi;
end

function R = market_clearing_resid(log_wb, N, income_base, par, csv)
    omegabar = exp(log_wb); % Ensures omegabar is always strictly positive
    
    Gam = csv.Gamma(omegabar);
    if Gam >= 1
        R = 1e6; return; % Prevent division by zero or negative Mf
    end
    
    Gp = csv.GammaPrime(omegabar);
    Psi = csv.Psi(omegabar);
    muG = par.mu * csv.G(omegabar);
    
    % Leverage condition (Equation 1) substituted here
    Mf = N * (1 + Gp * omegabar / (1 - Gam));
    ell = 1 - N / Mf;
    
    pm = par.Rstar * ell / Psi;
    mf = Mf * (1 - muG);
    
    % Market Clearing residual (Equation 2)
    LHS = mf * ((par.alpha * pm / (1 - par.alpha))^par.sigma + pm);
    RHS = income_base + (1 - Gam) * pm * Mf;
    
    R = LHS - RHS;
end
