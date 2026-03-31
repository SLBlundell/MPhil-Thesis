function R = period0_euler(b1, par, csv, y1_nodes, weights)
% PERIOD0_EULER  Euler equation residual for the Period 0 equilibrium.
%
%   R = period0_euler(b1, par, csv, y1_nodes, weights)
%
%   Single unknown: b1 (bond issuance).
%   Returns a scalar residual from the Euler equation [eq:euler]:
%
%     lambda_0 * q_0  =  beta * E[ lambda_1 * (1 + gamma * mf_1 * dpm/dN) ]
%
%   For each quadrature node y1_j, the Period 1 equilibrium is solved
%   to obtain D_j, lambda1_j, mf1_j, and dpm_dN_j.
%
%   Bond price determined by [eq:bondprice]:
%     q0 = (1/R*) * E[1 - D/b1]
%
%   Period 0 allocations use the frictionless intermediation assumption:
%     pm0 ~ R*,  so  md0 = y0 - b0 + q0*b1.

    nq = length(y1_nodes);

    % Solve Period 1 for each quadrature node
    D_vals       = zeros(nq, 1);
    lambda1_vals = zeros(nq, 1);
    mf1_vals     = zeros(nq, 1);
    dpm_dN_vals  = zeros(nq, 1);

    x0 = [];  % use default initial guess for first node
    for j = 1:nq
        sol_j = solve_period1(y1_nodes(j), b1, par, csv, x0);

        D_vals(j)       = sol_j.D;
        lambda1_vals(j) = sol_j.lambda1;
        mf1_vals(j)     = sol_j.mf;
        dpm_dN_vals(j)  = sol_j.dpm_dN;

        % Warm-start next quadrature node with current solution
        if sol_j.exitflag <= 0 || sol_j.omegabar < 1e-6
            x0 = [];
        else
            x0 = [sol_j.D; log(sol_j.omegabar); sol_j.Mf];
        end
    end

    % --- Bond price ---
    q0 = (1 / par.Rstar) * sum( weights .* (1 - D_vals / b1) );

    % --- Period 0 allocations (frictionless intermediation) ---
    %
    % Under the maintained assumption that Period 0 intermediaries are
    % well capitalised, pm0 ~ R* and dividends offset import costs,
    % yielding:  md0 = y0 - b0 + q0 * b1.
    md0 = par.y0 - par.b0 + q0 * b1;

    if md0 <= 0
        R = 1e6;  % penalise infeasible region
        return;
    end

    mf0 = md0 * ( (1 - par.alpha) / (par.alpha * par.Rstar) )^par.sigma;
    C0  = ( par.alpha * md0^par.eta ...
          + (1 - par.alpha) * mf0^par.eta )^(1/par.eta);

    % Shadow value:  lambda0 = u'(C0) * alpha * (C0/md0)^(1-eta)
    lambda0 = C0^(-par.sigma_u) * par.alpha * (C0 / md0)^(1 - par.eta);

    % --- Euler equation residual ---
    %
    %   lambda0 * q0  =  beta * E[ lambda1 * (1 + gamma * mf1 * dpm/dN) ]
    %
    % Note: dpm/dN < 0, so the term in parentheses < 1.
    euler_integrand = lambda1_vals .* ...
        (1 + par.gamma * mf1_vals .* dpm_dN_vals);

    R = lambda0 * q0 - par.beta * sum( weights .* euler_integrand );
end
