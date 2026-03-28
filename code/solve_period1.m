function sol = solve_period1(y1, b1, par, csv, x0)
    par.y1 = y1;
    par.b1 = b1;

    % Default initial guess
    if nargin < 5 || isempty(x0)
        D0   = b1 * 0.5;
        N0   = par.nbar + par.gamma * (b1 - D0);
        wb0  = 0.45;
        Gam0 = csv.Gamma(wb0);
        Gp0  = csv.GammaPrime(wb0);
        Mf0  = N0 * (1 + Gp0 * wb0 / (1 - Gam0));
        x0   = [D0; log(wb0); Mf0];
    end

    opts = optimoptions('fsolve', ...
        'Display',          'off', ...
        'Algorithm',        'trust-region-dogleg', ...
        'TolFun',           1e-10, ...
        'TolX',             1e-10, ...
        'MaxFunEvals',      5000, ...
        'MaxIter',          1000);

    % --- Step 1: Try interior default solution ---
    [xsol, ~, exitflag] = fsolve( ...
        @(x) period1_residuals(x, par, csv), x0, opts);

    R_final = period1_residuals(xsol, par, csv);
    interior_ok = (exitflag > 0) && (max(abs(R_final)) < 1e-6) && (xsol(1) > 1e-8);

    % --- Step 2: If interior failed or D<=0, use corner D=0 ---
    if ~interior_ok
        D_fixed = 0;
        N_c = par.nbar + par.gamma * (b1 - D_fixed);
        wb0_c = 0.45;
        Gam0 = csv.Gamma(wb0_c);
        Gp0  = csv.GammaPrime(wb0_c);
        Mf0_c = N_c * (1 + Gp0 * wb0_c / (1 - Gam0));

        [xsol_c, ~, exitflag] = fsolve( ...
            @(z) corner_residuals(z, D_fixed, par, csv), ...
            [log(wb0_c); Mf0_c], opts);

        xsol = [D_fixed; xsol_c(1); xsol_c(2)];
    end

    % --- Extract and recover ---
    sol.D        = xsol(1);
    sol.omegabar = exp(xsol(2));
    sol.Mf       = xsol(3);
    sol.exitflag = exitflag;

    sol.N   = par.nbar + par.gamma * (b1 - sol.D);
    sol.ell = 1 - sol.N / sol.Mf;
    sol.K   = sol.Mf - sol.N;

    Gam  = csv.Gamma(sol.omegabar);
    Gp   = csv.GammaPrime(sol.omegabar);
    Gpp  = csv.GammaDoublePrime(sol.omegabar);
    G_v  = csv.G(sol.omegabar);
    Psi  = csv.Psi(sol.omegabar);
    Psip = csv.PsiPrime(sol.omegabar);
    muG  = par.mu * G_v;

    sol.pm      = par.Rstar * sol.ell / Psi;
    sol.Z       = par.Rstar * sol.omegabar / Psi;
    sol.Z_Rstar = sol.omegabar / Psi;

    sol.mf  = sol.Mf * (1 - muG);
    sol.md  = sol.mf * (par.alpha * sol.pm / (1 - par.alpha))^par.sigma;
    sol.C1  = ( par.alpha * sol.md^par.eta ...
              + (1 - par.alpha) * sol.mf^par.eta )^(1/par.eta);
    sol.Pi  = (1 - Gam) * sol.pm * sol.Mf;

    sol.lambda1 = sol.C1^(-par.sigma_u) * par.alpha ...
                  * (sol.C1 / sol.md)^(1 - par.eta);

    denom_inner = Gp * (sol.Mf - sol.N) ...
                + sol.N * (Gpp * sol.omegabar + Gp);
    domega_dN   = -Gp * sol.omegabar * sol.Mf ...
                / ( (sol.Mf - sol.N) * denom_inner );

    sol.dpm_dN = par.Rstar ...
                * ( -Psi / sol.Mf  -  sol.ell * Psip * domega_dN ) ...
                / Psi^2;
end

% --- Inline corner residuals: just R1 and R2 with D fixed ---
function R = corner_residuals(z, D_fixed, par, csv)
    omegabar = exp(z(1));
    Mf       = z(2);

    N   = par.nbar + par.gamma * (par.b1 - D_fixed);
    ell = 1 - N / Mf;

    if Mf <= N || N <= 0 || Mf <= 0
        R = [1e6; 1e6];
        return
    end

    Gam = csv.Gamma(omegabar);
    Gp  = csv.GammaPrime(omegabar);
    Psi = csv.Psi(omegabar);
    muG = par.mu * csv.G(omegabar);
    pm  = par.Rstar * ell / Psi;
    mf  = Mf * (1 - muG);

    R1 = (1 - Gam) - Gp * omegabar * N / (Mf - N);

    LHS2 = mf * ((par.alpha * pm / (1 - par.alpha))^par.sigma + pm);
    RHS2 = par.y1 - (par.b1 - D_fixed) + (1 - Gam) * pm * Mf;
    R2   = LHS2 - RHS2;

    R = [R1; R2];
end