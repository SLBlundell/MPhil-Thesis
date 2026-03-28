function csv = csv_functions(sigma_w, mu)
% CSV_FUNCTIONS  Costly State Verification function handles.
%
%   csv = csv_functions(sigma_w, mu)
%
%   Returns a struct of function handles for the CSV functions under
%   log-normal specification: log(omega) ~ N(-sigma_w^2/2, sigma_w^2).
%
%   Fields:
%     csv.Gamma(w)            - Expected gross lender share
%     csv.G(w)                - Expected revenue from defaulting intermediaries
%     csv.GammaPrime(w)       - d Gamma / d omegabar  =  1 - F(omegabar)
%     csv.muGPrime(w)         - mu * dG/d omegabar
%     csv.GammaDoublePrime(w) - d^2 Gamma / d omegabar^2  (< 0)
%     csv.Psi(w)              - Gamma - mu*G  (net lender recovery)
%     csv.PsiPrime(w)         - d Psi / d omegabar
%
%   Reference: Bernanke, Gertler & Gilchrist (1999), Appendix.

    % Standardised normal argument: z(omegabar)
    z    = @(w) (log(w) + sigma_w^2/2) / sigma_w;

    % CDF of the idiosyncratic shock distribution
    F    = @(w) normcdf(z(w));

    % G(omegabar) = int_0^{omegabar} omega dF(omega)
    G    = @(w) normcdf(z(w) - sigma_w);

    % Gamma(omegabar) = G(omegabar) + omegabar [1 - F(omegabar)]
    Gam  = @(w) G(w) + w .* (1 - F(w));

    % Gamma'(omegabar) = 1 - F(omegabar)
    Gp   = @(w) 1 - F(w);

    % mu G'(omegabar) = (mu / sigma_w) * phi(z)
    muGp = @(w) (mu / sigma_w) * normpdf(z(w));

    % Gamma''(omegabar) = -phi(z) / (omegabar * sigma_w)   [< 0 for all w > 0]
    Gpp  = @(w) -normpdf(z(w)) ./ (w * sigma_w);

    % Psi(omegabar) = Gamma(omegabar) - mu * G(omegabar)
    Psi  = @(w) Gam(w) - mu * G(w);

    % Psi'(omegabar) = Gamma'(omegabar) - mu G'(omegabar)
    Psip = @(w) Gp(w) - muGp(w);

    % Pack into output struct
    csv.z                  = z;
    csv.F                  = F;
    csv.G                  = G;
    csv.Gamma              = Gam;
    csv.GammaPrime         = Gp;
    csv.muGPrime           = muGp;
    csv.GammaDoublePrime   = Gpp;
    csv.Psi                = Psi;
    csv.PsiPrime           = Psip;
end
