% Complete merged code: Build LDPC matrix, decode using IAMS, simulate BER
clc; clear; close all;

%% Step 1: Define Base Matrix (Example: BG2-like small matrix)
base_matrix = [...
    0 1 -1;
    1 0 1
];

Z = 2; % Lifting size (small for example)

%% Step 2: Build Expanded H matrix and Layer Info
[H_expanded, layers, col_degrees] = build_5G_LDPC(base_matrix, Z);
[M, N] = size(H_expanded);

%% Step 3: Simulation Parameters
Imax = 100;           
D = 2;               
lambda = 0;          
q = 4;               
q_tilde = 5;        
mu = 2;               
SNR_dB = 2;           
num_frames = 1;       

%% Step 4: Simulate One Frame
snr_linear = 10^(SNR_dB/10);
sigma = sqrt(1/(2*snr_linear));

% Assume all-zero codeword for simplicity
%true_codeword = zeros(1,N);
true_codeword = randi([0 1], 1, N);

% BPSK modulation: 0 -> +1, 1 -> -1
tx_signal = 1 - 2 * true_codeword;

% Transmit over AWGN
rx_signal = tx_signal + sigma*randn(1, N);

% Quantize received signal
gamma = max(min(round(mu * rx_signal), 2^(q-1)-1), -(2^(q-1)-1))

% Decode
decoded_bits = IAMS_decoder_5G(gamma, H_expanded, layers, col_degrees, Imax, D, lambda, q, q_tilde);

%% Step 5: Results
disp('Transmitted Codeword:'); disp(true_codeword);
disp('Received Signal:'); disp(rx_signal);
disp('Decoded Bits:'); disp(decoded_bits);

errors = sum(decoded_bits ~= true_codeword);
fprintf('Number of bit errors: %d\n', errors);

%% ---- Helper Functions ----
function [H_expanded, layers, col_degrees] = build_5G_LDPC(base_matrix, Z)
    [M_b, N_b] = size(base_matrix);
    M = M_b * Z;
    N = N_b * Z;
    H_expanded = sparse(M, N);
    for r = 1:M_b
        for c = 1:N_b
            shift = base_matrix(r,c);
            if shift ~= -1
                I_Z = speye(Z);
                H_block = circshift(I_Z, [0, shift]);
                H_expanded((r-1)*Z+1:r*Z, (c-1)*Z+1:c*Z) = H_block;
            end
        end
    end
    layers = cell(M_b,1);
    for l = 1:M_b
        layers{l} = (l-1)*Z + (1:Z);
    end
    col_degrees = sum(H_expanded ~= 0, 1);
end

function decoded_bits = IAMS_decoder_5G(gamma, H, layers, col_degrees, Imax, D, lambda, q, q_tilde)
    [Nc, Nv] = size(H);
    Q = 2^(q-1) - 1;
    Q_tilde = 2^(q_tilde-1) - 1;
    Gamma = -Q:Q;
    A = -Q_tilde:Q_tilde;

    alpha = zeros(Nc, Nv);
    beta = zeros(Nv, Nc);
    APP = gamma;

    for t = 1:Imax
        for l = 1:length(layers)
            check_nodes = layers{l};
            for m = check_nodes
                connected_vars = find(H(m,:) ~= 0);
                for n = connected_vars
                    beta(n,m) = clip(round(APP(n) - alpha(m,n)), Gamma);
                end
                abs_betas = abs(beta(connected_vars,m));
                signs = sign(beta(connected_vars,m));
                [sorted_vals, idx] = sort(abs_betas);
                min1 = sorted_vals(1);
                min2 = sorted_vals(min(2,end));
                sign_prod = prod(signs);

                for idx_var = 1:length(connected_vars)
                    n = connected_vars(idx_var);
                    if l <= 4 && col_degrees(n) >= D
                        other_idx = setdiff(1:length(connected_vars), idx_var);
                        if isempty(other_idx)
                            min_other = 0;
                        else
                            min_other = min(abs_betas(other_idx));
                        end
                        temp = max(min_other - lambda, 0);
                        alpha(m,n) = clip(sign_prod * temp, Gamma);
                    else
                        if connected_vars(idx(1)) == n
                            selected_min = min2;
                        else
                            selected_min = min1;
                        end
                        alpha(m,n) = clip(sign_prod * selected_min, Gamma);
                    end
                end
            end
            % Update APPs after each layer
            for m = check_nodes
                connected_vars = find(H(m,:) ~= 0);
                for n = connected_vars
                    APP(n) = clip(round(gamma(n) + sum(alpha(find(H(:,n)),n))), A);
                end
            end
        end
        decoded_bits = double(APP < 0);
        if all(mod(H * decoded_bits',2) == 0)
            return;
        end
    end
end

function y = clip(x, alphabet)
    [~, idx] = min(abs(alphabet - x));
    y = alphabet(idx);
end
