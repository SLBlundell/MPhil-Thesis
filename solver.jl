"""
solver.jl  —  Numerical Solver for the Two-Period Model of Partial Sovereign Default

Model overview:
  - Representative Household (sovereign): max u(C₀) + β E₀[u(C₁)]
  - Domestic Import Intermediaries (i ∈ [0,1]): subject to CSV financial friction
  - International Lender: risk-neutral pricing of sovereign bonds & working capital

Core mechanism:
  Sovereign default D ↑  →  bank net worth n₁ ↓  →  leverage ↑
    →  CSV friction (external finance premium) ↑  →  import price p_m ↑
    →  effective imports ↓  →  consumption C₁ ↓   (endogenous default cost)

Packages required:
  Distributions, NLsolve, Optim
  Install via: using Pkg; Pkg.add(["Distributions", "NLsolve", "Optim"])
"""

using Distributions
using NLsolve
using Optim
using Printf
using LinearAlgebra


# ==============================================================================
# 1.  PARAMETERS
# ==============================================================================

"""
All structural parameters of the model.  Defaults match a plausible calibration
for a small open emerging-market economy.
"""
Base.@kwdef struct Params
    # --- Preferences ---
    β::Float64    = 0.96    # Discount factor
    σ_u::Float64  = 2.0     # CRRA coefficient  (σ_u = 1 → log utility)

    # --- Armington CES aggregator  C = [α(mᵈ)^η + (1-α)(mᶠ)^η]^(1/η) ---
    α::Float64    = 0.70    # Weight on domestic inputs
    η::Float64    = -1.0    # Elasticity parameter  (η < 0 → strict complements)

    # --- World prices ---
    pf::Float64   = 1.0     # World price of foreign inputs (real terms-of-trade)
    Rstar::Float64 = 1.04   # World gross risk-free interest rate

    # --- Intermediary / financial friction ---
    γ::Float64    = 0.30    # Fraction of sovereign bonds held by domestic banks
    n̄::Float64    = 0.10    # Long-run intermediary net worth (equity floor)
    μ_csv::Float64 = 0.25   # CSV auditing cost fraction  μ ∈ (0,1)
    σ_ω::Float64  = 0.25    # Std dev of log idiosyncratic intermediary shock

    # --- Endowment process  log(g₁) = (1-ρ)log(μ_g) + ρ·log(g₀) + ε, ε~N(0,σ_g²) ---
    y0::Float64   = 1.00    # Period 0 endowment
    μ_g::Float64  = 1.02    # Mean endowment growth rate
    ρ::Float64    = 0.50    # AR(1) persistence in log growth
    g0::Float64   = 1.02    # Initial growth rate (lagged value for AR process)
    σ_g::Float64  = 0.05    # Std dev of growth innovation

    # --- Initial conditions ---
    b0::Float64   = 0.20    # Period 0 legacy debt (repaid within Period 0)

    # --- Numerical ---
    n_quad::Int   = 21      # Gauss-Hermite quadrature nodes for E₀ expectations
end


# ==============================================================================
# 2.  UTILITY
# ==============================================================================

u_crra(C, σ) = σ == 1.0 ? log(C) : C^(1 - σ) / (1 - σ)


# ==============================================================================
# 3.  LOG-NORMAL INTERMEDIARY SHOCK FUNCTIONS
# ==============================================================================
#
# ω ~ LN(−σ_ω²/2, σ_ω²)  so that  E[ω] = 1
#
# F(ω̄)  = Φ( (ln ω̄ + σ_ω²/2) / σ_ω )
# G(ω̄)  = ∫₀^ω̄ ω dF(ω)  = Φ( (ln ω̄ − σ_ω²/2) / σ_ω )
# Γ(ω̄)  = G(ω̄) + ω̄ [1 − F(ω̄)]        (expected gross revenue share to lender)
# Γ′(ω̄) = 1 − F(ω̄)

lnorm_F(ω̄, σ_ω)  = cdf(Normal(), (log(ω̄) + σ_ω^2/2) / σ_ω)
G_func(ω̄,  σ_ω)  = cdf(Normal(), (log(ω̄) - σ_ω^2/2) / σ_ω)
Γ_func(ω̄,  σ_ω)  = G_func(ω̄, σ_ω) + ω̄ * (1.0 - lnorm_F(ω̄, σ_ω))
Γp_func(ω̄, σ_ω)  = 1.0 - lnorm_F(ω̄, σ_ω)


# ==============================================================================
# 4.  PERIOD 1 COMPETITIVE EQUILIBRIUM SOLVER
# ==============================================================================
#
# Given intermediary net worth n₁ and household income I₁ = y₁ − (b₁ − D):
#
# Unknowns: ω̄₁  (idiosyncratic cutoff threshold)
#
# Algorithm:  for a candidate ω̄ —
#   (a) Equity-to-debt ratio:  l(ω̄) = (1−Γ) / (Γ′·ω̄)          [from leverage eq.]
#   (b) Raw intermediary imports: M_f = n₁·(1 + 1/l) / pf       [balance sheet]
#   (c) Import price: p_m = pf·R* / [(l+1)·(Γ−μG)]              [participation]
#   (d) Effective imports (supply): x_s = M_f·(1−μG)
#   (e) Effective imports (demand): x_d = r(p_m)·I₁ / (1+p_m·r) [household FOC]
#   Equilibrium: x_s = x_d  →  single equation in ω̄  solved by Newton/TR.

"""
    solve_p1_alloc(n1, I1, p) → (m_f_eff, m_d, C1, p_m, ω̄, M_f_raw)

Period 1 competitive equilibrium for given net worth `n1` and household income `I1`.
Returns effective foreign inputs, domestic inputs, consumption, import price,
cutoff threshold ω̄, and raw intermediary import volume.
"""
function solve_p1_alloc(n1::Float64, I1::Float64, p::Params)
    (; pf, Rstar, μ_csv, σ_ω, α, η) = p

    (n1 ≤ 0.0 || I1 ≤ 0.0) && return (NaN, NaN, NaN, NaN, NaN, NaN)

    # --- Feasibility check ---
    # At zero leverage (ω̄→0), supply → n1/pf and p_m → pf·R* (the frictionless price).
    # An equilibrium can only exist if min-supply < max-demand, i.e.:
    #   n1/pf  <  r_fric · I1 / (1 + pf·R*·r_fric)
    r_fric    = ((1.0 - α) / (α * pf * Rstar))^(1.0 / (1.0 - η))
    x_d_max   = r_fric * I1 / (1.0 + pf * Rstar * r_fric)   # demand at frictionless price
    n1 / pf ≥ x_d_max && return (NaN, NaN, NaN, NaN, NaN, NaN)

    # Equilibrium residual: supply of effective imports minus household demand
    function resid!(F, log_ω̄_vec)
        ω̄ = exp(log_ω̄_vec[1])
        Γ    = Γ_func(ω̄, σ_ω)
        Γp   = Γp_func(ω̄, σ_ω)
        G    = G_func(ω̄, σ_ω)
        ΓmμG = Γ - μ_csv * G

        if ΓmμG ≤ 1e-10 || Γp ≤ 1e-10
            F[1] = 1e8; return
        end

        l = (1.0 - Γ) / (Γp * ω̄)         # equity-to-debt ratio
        l ≤ 0.0 && (F[1] = 1e8; return)

        M_f  = n1 * (1.0 + 1.0/l) / pf    # raw imports (intermediary balance sheet)
        M_f ≤ 0.0 && (F[1] = 1e8; return)

        x_s  = M_f * (1.0 - μ_csv * G)    # effective supply (net of auditing losses)

        p_m  = pf * Rstar / ((l + 1.0) * ΓmμG)   # import price (participation constraint)
        p_m ≤ 0.0 && (F[1] = 1e8; return)

        # Household demand for effective foreign inputs (from intratemporal FOC)
        ratio = ((1.0 - α) / (α * p_m))^(1.0 / (1.0 - η))
        x_d   = ratio * I1 / (1.0 + p_m * ratio)

        F[1] = x_s - x_d
    end

    # Bracket the root by scanning log(ω̄) ∈ [log(0.05), log(5)]
    # and finding the sign-change interval, then passing the midpoint as initial guess.
    log_grid = range(log(0.05), log(5.0), length = 30)
    F_vals    = zeros(30)
    resid_tmp = zeros(1)
    for (k, lw) in enumerate(log_grid)
        resid!(resid_tmp, [lw])
        F_vals[k] = resid_tmp[1]
    end

    # Find first sign change
    log_init = log(0.5)              # fallback
    for k in 1:length(log_grid)-1
        if isfinite(F_vals[k]) && isfinite(F_vals[k+1]) && F_vals[k] * F_vals[k+1] < 0
            log_init = (log_grid[k] + log_grid[k+1]) / 2.0
            break
        end
    end

    sol = nothing
    for ω̄_init in [exp(log_init), 0.5, 0.3, 0.7, 1.0, 0.1, 1.5, 2.0]
        try
            cand = nlsolve(resid!, [log(ω̄_init)];
                           method     = :trust_region,
                           ftol       = 1e-10,
                           xtol       = 1e-10,
                           iterations = 500)
            if converged(cand)
                sol = cand; break
            end
        catch
            continue
        end
    end

    (sol === nothing || !converged(sol)) && return (NaN, NaN, NaN, NaN, NaN, NaN)

    ω̄    = exp(sol.zero[1])
    Γ    = Γ_func(ω̄, σ_ω)
    Γp   = Γp_func(ω̄, σ_ω)
    G    = G_func(ω̄, σ_ω)
    ΓmμG = Γ - μ_csv * G
    l    = (1.0 - Γ) / (Γp * ω̄)
    M_f  = n1 * (1.0 + 1.0/l) / pf
    p_m  = pf * Rstar / ((l + 1.0) * ΓmμG)

    # Household allocations
    ratio = ((1.0 - α) / (α * p_m))^(1.0 / (1.0 - η))
    m_d   = I1 / (1.0 + p_m * ratio)
    m_f   = ratio * m_d                      # effective foreign inputs

    # CES consumption
    inner = α * m_d^η + (1.0 - α) * m_f^η
    inner ≤ 0.0 && return (NaN, NaN, NaN, NaN, NaN, NaN)
    C1 = inner^(1.0 / η)

    return (m_f, m_d, C1, p_m, ω̄, M_f)
end


# ==============================================================================
# 5.  OPTIMAL DEFAULT IN PERIOD 1
# ==============================================================================
#
# Household maximises u(C₁) over D ∈ [0, b₁], where:
#   net worth:  n₁ = n̄ + γ(b₁ − D)
#   income:     I₁ = y₁ − (b₁ − D)
#
# KKT conditions (see paper eq. default-kkt):
#   Interior:  1 = m_f · ∂p_m/∂D
#   D = 0  if the marginal cost of default exceeds benefit at D = 0
#   D = b₁ if the marginal benefit exceeds cost at D = b₁
#
# Numerically: direct 1-D optimisation over D using Brent's method.

"""
    solve_default(y1, b1, p) → (D_opt, C1, m_f, m_d, p_m, ω̄)

Optimal default and Period 1 allocations for given endowment `y1` and debt `b1`.
"""
function solve_default(y1::Float64, b1::Float64, p::Params)
    (; n̄, γ, σ_u) = p

    function neg_welfare(D::Float64)
        repay = b1 - D
        I1    = y1 - repay
        n1    = n̄ + γ * repay
        (I1 ≤ 1e-8 || n1 ≤ 1e-8) && return 1e10

        (m_f, m_d, C1, p_m, ω̄, M_f) = solve_p1_alloc(n1, I1, p)
        (isnan(C1) || C1 ≤ 1e-10) && return 1e10
        return -u_crra(C1, σ_u)
    end

    # Feasibility: I₁ > 0  requires  D > b₁ − y₁
    D_lb = max(0.0, b1 - y1 + 1e-6)
    D_ub = b1
    D_lb ≥ D_ub && (D_lb = D_ub - 1e-8)

    try
        res   = optimize(neg_welfare, D_lb, D_ub; method = Brent(), abs_tol = 1e-10)
        D_opt = Optim.minimizer(res)
        n1    = n̄ + γ * (b1 - D_opt)
        I1    = y1 - (b1 - D_opt)
        (m_f, m_d, C1, p_m, ω̄, M_f) = solve_p1_alloc(n1, I1, p)
        return (D_opt, C1, m_f, m_d, p_m, ω̄)
    catch
        return (D_ub, NaN, NaN, NaN, NaN, NaN)
    end
end


# ==============================================================================
# 6.  GAUSS-HERMITE QUADRATURE AND EXPECTATION OVER g₁
# ==============================================================================

"""
    gh_nodes_weights(n) → (nodes, weights)

Physicist's Gauss-Hermite quadrature of order `n`, computed via eigenvalue
decomposition of the tridiagonal Jacobi matrix.

Convention:  ∫_{-∞}^{∞} f(x) e^{−x²} dx  ≈  Σᵢ wᵢ f(xᵢ),   Σwᵢ = √π
"""
function gh_nodes_weights(n::Int)
    β_sub = sqrt.(collect(1:n-1) ./ 2.0)
    J     = SymTridiagonal(zeros(n), β_sub)
    F     = eigen(J)
    idx   = sortperm(F.values)
    nodes   = F.values[idx]
    weights = sqrt(π) .* F.vectors[1, idx].^2
    return nodes, weights
end

"""
    E0_g1(f, p) → E₀[f(y₁)]

Expectation over y₁ = g₁·y₀ using Gauss-Hermite quadrature, where:
  log(g₁) ~ N( (1−ρ)log(μ_g) + ρ·log(g₀),  σ_g² )
"""
function E0_g1(f::Function, p::Params)
    (; y0, μ_g, ρ, g0, σ_g, n_quad) = p
    μ_ln = (1.0 - ρ) * log(μ_g) + ρ * log(g0)   # conditional mean of log(g₁)

    nodes, weights = gh_nodes_weights(n_quad)

    # Change of variables: log(g₁) = μ_ln + √2·σ_g·t
    # E[f(g₁)] = (1/√π) Σ wᵢ f( exp(μ_ln + √2·σ_g·tᵢ) · y₀ )
    total = sum(w * f(exp(μ_ln + sqrt(2.0) * σ_g * t) * y0)
                for (t, w) in zip(nodes, weights))
    return total / sqrt(π)
end


# ==============================================================================
# 7.  BOND PRICE SCHEDULE
# ==============================================================================

"""
    bond_price(b1, p) → q₀

Risk-neutral pricing:  q₀(y₀, b₁) = (1/R*) · E₀[1 − D(y₁, b₁)/b₁]
"""
function bond_price(b1::Float64, p::Params)
    b1 ≤ 0.0 && return 1.0 / p.Rstar

    q0 = E0_g1(p) do y1
        (D, _...) = solve_default(y1, b1, p)
        isnan(D) ? 0.0 : 1.0 - D / b1
    end
    return q0 / p.Rstar
end


# ==============================================================================
# 8.  PERIOD 0 ALLOCATION
# ==============================================================================
#
# Negligible Period 0 external finance premium (paper assumption):
#   p_{m,0} ≈ pf · R*
# Budget:  m^d₀ + p_{m,0}·m^f₀ = y₀ − b₀ + q₀·b₁

"""
    solve_p0_alloc(b1, q0, p) → (C0, m_d0, m_f0)
"""
function solve_p0_alloc(b1::Float64, q0::Float64, p::Params)
    (; y0, b0, pf, Rstar, α, η) = p
    I0 = y0 - b0 + q0 * b1
    I0 ≤ 0.0 && return (NaN, NaN, NaN)

    p_m0  = pf * Rstar                                          # negligible friction
    ratio = ((1.0 - α) / (α * p_m0))^(1.0 / (1.0 - η))
    m_d0  = I0 / (1.0 + p_m0 * ratio)
    m_f0  = ratio * m_d0

    inner = α * m_d0^η + (1.0 - α) * m_f0^η
    inner ≤ 0.0 && return (NaN, NaN, NaN)
    C0 = inner^(1.0 / η)
    return (C0, m_d0, m_f0)
end


# ==============================================================================
# 9.  HOUSEHOLD'S PERIOD 0 PROBLEM  —  choose b₁
# ==============================================================================
#
# Price-taking assumption: household takes q₀(b₁) as parametric (coordination
# failure in debt issuance, cf. Goodhart et al. 2018).
#
# V₀ = max_{b₁ > 0}  u(C₀) + β · E₀[u(C₁(y₁, b₁))]
#
# Strategy:
#   1. Coarse grid search to bracket the optimum
#   2. Brent's method for fine optimisation

"""
    solve_period0(p) → (b1_opt, q0_opt, C0_opt, m_d0, m_f0)
"""
function solve_period0(p::Params)
    (; β, σ_u) = p

    function neg_V0(b1::Float64)
        b1 ≤ 1e-6 && return 1e10

        q0 = bond_price(b1, p)
        (isnan(q0) || q0 ≤ 0.0) && return 1e10

        (C0, m_d0, m_f0) = solve_p0_alloc(b1, q0, p)
        (isnan(C0) || C0 ≤ 0.0) && return 1e10

        EU1 = E0_g1(p) do y1
            (D, C1, _...) = solve_default(y1, b1, p)
            (isnan(C1) || C1 ≤ 0.0) ? 0.0 : u_crra(C1, σ_u)
        end

        return -(u_crra(C0, σ_u) + β * EU1)
    end

    # Coarse grid search across plausible debt levels
    b1_grid  = range(0.01, 2.0, length = 40)
    V0_grid  = neg_V0.(b1_grid)
    valid    = isfinite.(V0_grid)
    b1_init  = b1_grid[valid][argmin(V0_grid[valid])]

    # Brent optimisation in a neighbourhood of the grid minimum
    b_lo = max(0.001, b1_init - 0.4)
    b_hi = b1_init + 0.4
    res  = optimize(neg_V0, b_lo, b_hi; method = Brent(), abs_tol = 1e-8)

    b1_opt = Optim.minimizer(res)
    q0_opt = bond_price(b1_opt, p)
    (C0_opt, m_d0, m_f0) = solve_p0_alloc(b1_opt, q0_opt, p)

    return (b1_opt, q0_opt, C0_opt, m_d0, m_f0)
end


# ==============================================================================
# 10.  FULL EQUILIBRIUM  —  solve and print results
# ==============================================================================

"""
    solve_equilibrium(p) → NamedTuple

Compute the full two-period equilibrium under default parameters `p`.
Prints a formatted summary table.
"""
function solve_equilibrium(p::Params = Params())
    println("=" ^ 62)
    println("  Two-Period Sovereign Default Model  —  Numerical Solver")
    println("=" ^ 62)

    println("\n[1]  Solving Period 0 household bond-issuance problem...")
    (b1, q0, C0, m_d0, m_f0) = solve_period0(p)

    spread_pp = (1.0/q0 - p.Rstar) * 100     # sovereign spread in basis points

    println("\n── Period 0 ──────────────────────────────────────────────")
    @printf("  Optimal debt issuance  b₁   = %.4f\n",   b1)
    @printf("  Bond price             q₀   = %.4f\n",   q0)
    @printf("  Sovereign spread             = %.2f pp\n", spread_pp)
    @printf("  Consumption            C₀   = %.4f\n",   C0)
    @printf("  Domestic inputs        m^d₀ = %.4f\n",   m_d0)
    @printf("  Effective foreign inp. m^f₀ = %.4f\n",   m_f0)

    println("\n[2]  Period 1 outcomes across endowment scenarios...")
    println("\n── Period 1 Scenarios ────────────────────────────────────")
    @printf("  %-5s │ %-6s │ %-6s │ %-5s │ %-6s │ %-6s │ %-5s\n",
            "g₁", "y₁", "D", "D/b₁", "C₁", "p_m₁", "ω̄₁")
    println("  " * "─" ^ 55)

    results_p1 = NamedTuple[]
    for g1 in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
        y1 = g1 * p.y0
        (D, C1, m_f1, m_d1, p_m1, ω̄1) = solve_default(y1, b1, p)
        D_ratio = isnan(D) ? NaN : D / b1
        @printf("  %.3f │ %.4f │ %.4f │ %.3f │ %.4f │ %.4f │ %.4f\n",
                g1, y1, D, D_ratio, C1, p_m1, ω̄1)
        push!(results_p1, (; g1, y1, D, D_ratio, C1, m_f1, m_d1, p_m1, ω̄1))
    end

    println("\n── Unconditional Expectations  (Gauss-Hermite, n=$(p.n_quad)) ──")
    ED  = E0_g1(y1 -> solve_default(y1, b1, p)[1], p)
    EC1 = E0_g1(y1 -> solve_default(y1, b1, p)[2], p)
    @printf("  E₀[D]    = %.4f   (%.1f%% of b₁)\n", ED, 100 * ED / b1)
    @printf("  E₀[C₁]   = %.4f\n", EC1)

    println("\n── Calibration ───────────────────────────────────────────")
    @printf("  β=%.2f  α=%.2f  η=%.2f  σ_u=%.2f  R*=%.4f\n",
            p.β, p.α, p.η, p.σ_u, p.Rstar)
    @printf("  γ=%.2f  n̄=%.2f  μ_csv=%.2f  σ_ω=%.2f\n",
            p.γ, p.n̄, p.μ_csv, p.σ_ω)
    @printf("  y₀=%.2f  b₀=%.2f  μ_g=%.3f  ρ=%.2f  σ_g=%.3f\n",
            p.y0, p.b0, p.μ_g, p.ρ, p.σ_g)
    println("=" ^ 62)

    return (; b1, q0, C0, m_d0, m_f0, spread_pp, results_p1, ED, EC1)
end


# ==============================================================================
# 11.  SENSITIVITY ANALYSES
# ==============================================================================

"""
    sensitivity_gamma(p; γ_vals)

Equilibrium outcomes across values of γ (bank home bias in sovereign bonds),
illustrating how the depth of the bank-sovereign nexus affects borrowing costs.
"""
function sensitivity_gamma(p::Params = Params();
                            γ_vals = 0.0:0.05:0.50)
    println("\n── Sensitivity: γ (bank home bias) ──────────────────────")
    @printf("  %-5s │ %-6s │ %-6s │ %-10s │ %-8s\n",
            "γ", "b₁", "q₀", "Spread(pp)", "E₀[D/b₁]")
    println("  " * "─" ^ 48)

    for γ in γ_vals
        p2 = Params(β=p.β, σ_u=p.σ_u, α=p.α, η=p.η, pf=p.pf,
                    Rstar=p.Rstar, γ=γ, n̄=p.n̄, μ_csv=p.μ_csv,
                    σ_ω=p.σ_ω, y0=p.y0, μ_g=p.μ_g, ρ=p.ρ,
                    g0=p.g0, σ_g=p.σ_g, b0=p.b0, n_quad=p.n_quad)
        try
            (b1, q0, _...) = solve_period0(p2)
            spread = (1.0/q0 - p.Rstar) * 100
            ED_rat = E0_g1(y1 -> solve_default(y1, b1, p2)[1], p2) / b1
            @printf("  %.3f │ %.4f │ %.4f │ %10.4f │ %8.4f\n",
                    γ, b1, q0, spread, ED_rat)
        catch
            @printf("  %.3f │   —    │   —    │     —        │    —\n", γ)
        end
    end
end

"""
    sensitivity_mu_csv(p; μ_vals)

Equilibrium outcomes across values of μ (CSV auditing cost), showing how
the severity of the financial friction amplifies the default penalty.
"""
function sensitivity_mu_csv(p::Params = Params();
                             μ_vals = 0.05:0.05:0.45)
    println("\n── Sensitivity: μ_csv (CSV auditing cost) ───────────────")
    @printf("  %-5s │ %-6s │ %-6s │ %-10s\n", "μ", "b₁", "q₀", "Spread(pp)")
    println("  " * "─" ^ 38)

    for μ in μ_vals
        p2 = Params(β=p.β, σ_u=p.σ_u, α=p.α, η=p.η, pf=p.pf,
                    Rstar=p.Rstar, γ=p.γ, n̄=p.n̄, μ_csv=μ,
                    σ_ω=p.σ_ω, y0=p.y0, μ_g=p.μ_g, ρ=p.ρ,
                    g0=p.g0, σ_g=p.σ_g, b0=p.b0, n_quad=p.n_quad)
        try
            (b1, q0, _...) = solve_period0(p2)
            spread = (1.0/q0 - p.Rstar) * 100
            @printf("  %.3f │ %.4f │ %.4f │ %10.4f\n", μ, b1, q0, spread)
        catch
            @printf("  %.3f │   —    │   —    │     —\n", μ)
        end
    end
end

"""
    default_policy(b1, p; n_g1 = 50)

Compute the default policy function D*(y₁; b₁) across a grid of y₁ realisations,
showing the continuous, partial nature of equilibrium default.
Returns a vector of (y1, D, D_ratio, C1, p_m1, ω̄1) NamedTuples.
"""
function default_policy(b1::Float64, p::Params; n_g1::Int = 50)
    y1_lo = exp((1.0-p.ρ)*log(p.μ_g) + p.ρ*log(p.g0) - 3p.σ_g) * p.y0
    y1_hi = exp((1.0-p.ρ)*log(p.μ_g) + p.ρ*log(p.g0) + 3p.σ_g) * p.y0
    y1_grid = range(y1_lo, y1_hi, length = n_g1)

    rows = NamedTuple[]
    for y1 in y1_grid
        (D, C1, m_f1, m_d1, p_m1, ω̄1) = solve_default(y1, b1, p)
        D_ratio = isnan(D) ? NaN : D / b1
        push!(rows, (; y1, D, D_ratio, C1, p_m1, ω̄1))
    end
    return rows
end


# ==============================================================================
# 12.  ENTRY POINT
# ==============================================================================

p       = Params()
results = solve_equilibrium(p)

sensitivity_gamma(p)
sensitivity_mu_csv(p)

println("\n── Default Policy Function  (b₁ = $(round(results.b1, digits=4))) ──")
policy  = default_policy(results.b1, p; n_g1 = 10)
@printf("  %-6s │ %-6s │ %-5s │ %-6s │ %-6s\n", "y₁", "D", "D/b₁", "C₁", "p_m₁")
println("  " * "─" ^ 40)
for r in policy
    @printf("  %.4f │ %.4f │ %.3f │ %.4f │ %.4f\n",
            r.y1, r.D, r.D_ratio, r.C1, r.p_m1)
end
