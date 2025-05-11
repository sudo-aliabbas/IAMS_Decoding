% IAMS LDPC Decoder Simulation
clc; clear;

% Parameters
q = 3;                   % Quantization bits
Q = 2^(q-1)-1;           % Quantization range
mu = 3;                  % Gain factor for quantization
lambda = 0.8;            % Damping factor
D = 2;                   % Degree threshold for adaptation
Imax = 30;               % Max number of iterations

% Parity-check matrix (3x7)
H = [1 0 1 0 1 0 1;
     0 1 1 0 0 1 1;
     0 0 0 1 1 1 1];

% Generator matrix consistent with H
G = [1 0 0 0 1 1 0;
     0 1 0 0 1 0 1;
     0 0 1 0 0 1 1;
     0 0 0 1 1 1 1];

% Original message
msg = [1 0 0 0];
codeword = mod(msg * G, 2);

% BPSK Modulation: 0 -> +1, 1 -> -1
mod_signal = 1 - 2 * codeword;

% Add Gaussian noise
snr = 3;  % dB
sigma = sqrt(1 / (2 * 10^(snr / 10)));
rx_signal = mod_signal + sigma * randn(1, length(mod_signal));

% Quantization
gamma = max(min(round(mu * rx_signal), Q), -Q);

% Decode using IAMS
decoded = IAMS_decoder(gamma, H, Imax, D, lambda, q);

% Show results
disp("Original Message:"); disp(msg);
disp("Transmitted Codeword:"); disp(codeword);
disp("Received Signal (noisy):"); disp(rx_signal);
disp("Decoded Codeword:"); disp(decoded);

decoding_success = isequal(decoded, codeword);
disp("Decoding Success:"); disp(decoding_success);

% ----- Decoder Function -----
function decoded = IAMS_decoder(gamma, H, Imax, D, lambda, q)
    [M, N] = size(H);
    Q = 2^(q-1) - 1;
    alpha = zeros(M, N);
    beta = zeros(N, M);
    gamma_tilde = gamma;

    % Build neighbors
    N_m = cell(M,1);
    M_n = cell(N,1);
    for m = 1:M
        N_m{m} = find(H(m,:) ~= 0);
    end
    for n = 1:N
        M_n{n} = find(H(:,n) ~= 0)';
    end

    for t = 1:Imax
        for l = 1:M
            m = l;
            for n = N_m{m}
                set_n = setdiff(N_m{m}, n);
                signs = prod(sign(gamma_tilde(set_n)));
                abs_vals = abs(gamma_tilde(set_n));
                if isempty(abs_vals)
                    min1 = 0; min2 = 0; idx1 = -1;
                else
                    [sorted_vals, sorted_idx] = sort(abs_vals);
                    min1 = sorted_vals(1);
                    idx1 = set_n(sorted_idx(1));
                    if length(sorted_vals) > 1
                        min2 = sorted_vals(2);
                        idx2 = set_n(sorted_idx(2));
                    else
                        min2 = min1;
                        idx2 = idx1;
                    end
                end
                beta(n, m) = signs * min1;
            end

            for n = N_m{m}
                dv = length(M_n{n});
                if l <= 4 && dv >= D
                    other_nodes = setdiff(N_m{m}, n);
                    abs_vals = abs(beta(other_nodes, m));
                    min_val = min(abs_vals);
                    val = max(min_val - lambda, 0);
                    alpha(m, n) = H(m,n) * sign(prod(sign(beta(other_nodes, m)))) * val;
                else
                    if n == idx1
                        val = min2;
                    else
                        val = min1;
                    end
                    alpha(m, n) = H(m,n) * sign(prod(sign(beta(setdiff(N_m{m}, n), m)))) * val;
                end
                gamma_tilde(n) = gamma(n) + sum(alpha(M_n{n}, n));
            end
        end
        decoded = double(gamma_tilde < 0);
        if all(mod(H * decoded', 2) == 0)
            return;
        end
    end
end
