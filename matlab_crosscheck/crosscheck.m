% Run this inside the MINOR-PROJECT-PASS folder (MATLAB), then compare the
% printed numbers with:  python matlab_crosscheck/crosscheck.py
% Fading is switched off so both simulators are deterministic.
P = PASS_Parameters();  P.use_fading = false;
Env = PASS_Environment(P);
Env.user_pos = [6 7 0; 12 21 0; 19 9 0; 26 24 0];
X = zeros(P.M, P.N);
for n = 1:P.N
    for m = 1:P.M
        X(m,n) = 2 + 5*(m-1) + 0.13*(n-1);
    end
end
[~, ~, h_eff] = PASS_ChannelModel(P, Env, X);
[~, pw] = PASS_ZFBeamformer(P, h_eff);
fprintf('|Psi(1,1)|   = %.6e\n', abs(h_eff(1,1)));
fprintf('|Psi(3,2)|   = %.6e\n', abs(h_eff(3,2)));
fprintf('angle Psi(1,1) = %.4f rad\n', angle(h_eff(1,1)));
fprintf('ZF Tx power  = %.4f dBm\n', 10*log10(pw*1e3));
