function R = period1_residuals(x, par, csv)
% PERIOD1_RESIDUALS  Residual system for the Period 1 equilibrium.
%
%   R = period1_residuals(x, par, csv)
%
%   Unknowns:  x = [D; omegabar; Mf]
%     D        - sovereign default (haircut)
%     omegabar - intermediary default threshold
%     Mf       - aggregate raw foreign inputs intermediated
%
%   Returns R = [R1; R2; R3], a 3x1 vector that equals zero in equilibrium.
%
%   The three equations are:
%     R1: Leverage-threshold condition       [eq:sys_leverage]
%     R2: Foreign input market clearing      [eq:sys_clearing]
%     R3: Interior default optimality        [eq:sys_default]
%
%   par must contain fields: y1, b1, nbar, gamma, mu, Rstar, alpha, eta, sigma.
%   csv must be the struct returned by csv_functions().

    D        = x(1);
    omegabar = x(2);
    Mf       = x(3);

    % --- Derived quantities ---
    N   = par.nbar + par.gamma * (par.b1 - D);          % intermediary net worth
    ell = 1 - N / Mf;                                   % leverage ratio

    % --- CSV evaluations ---
    Gam  = csv.Gamma(omegabar);
    Gp   = csv.GammaPrime(omegabar);
    Gpp  = csv.GammaDoublePrime(omegabar);
    G_v  = csv.G(omegabar);
    Psi  = csv.Psi(omegabar);
    Psip = csv.PsiPrime(omegabar);
    muG  = par.mu * G_v;

    % Import price:  pm = R* (1 - N/Mf) / Psi(omegabar)
    pm = par.Rstar * ell / Psi;

    % Household import demand (= intermediary supply net of auditing losses)
    mf = Mf * (1 - muG);

    % =====================================================================
    %  R1: Leverage-threshold condition  [eq:sys_leverage]
    %
    %    1 - Gamma(omegabar) = Gamma'(omegabar) * omegabar * N / (Mf - N)
    % =====================================================================
    R1 = (1 - Gam) - Gp * omegabar * N / (Mf - N);

    % =====================================================================
    %  R2: Foreign input market clearing  [eq:sys_clearing]
    %
    %    mf * [ (alpha pm / (1-alpha))^sigma  +  pm ]
    %        = y1 - (b1 - D)  +  (1 - Gamma) * pm * Mf
    %
    %  LHS: household expenditure on domestic + foreign inputs
    %  RHS: disposable income  =  endowment - repayment + dividends
    % =====================================================================
    LHS2 = mf * ( (par.alpha * pm / (1 - par.alpha))^par.sigma + pm );
    RHS2 = par.y1 - (par.b1 - D) + (1 - Gam) * pm * Mf;
    R2   = LHS2 - RHS2;

    % =====================================================================
    %  R3: Interior default condition  [eq:sys_default]
    %
    %    Psi^2  =  gamma * R* * Mf * (1 - mu G) *
    %              [ Psi / Mf  +  ell * Psi' * |d omegabar / d N| ]
    %
    %  where |d omegabar / d N| is from [eq:domega_dn]:
    %
    %    d omegabar / d N  =  - Gamma' * omegabar * Mf
    %                         / { (Mf - N) * [ Gamma'(Mf-N) + N(Gamma'' omegabar + Gamma') ] }
    % =====================================================================
    denom_inner   = Gp * (Mf - N) + N * (Gpp * omegabar + Gp);
    abs_domega_dN = Gp * omegabar * Mf / ( (Mf - N) * denom_inner );

    R3 = Psi^2 - par.gamma * par.Rstar * Mf * (1 - muG) * ...
         ( Psi / Mf  +  ell * Psip * abs_domega_dN );

    R = [R1; R2; R3];
end
