% IAMS LDPC Decoder FER Simulation and Plotting (5G BG2 Code)
clc; clear; close all;

%% Parameters
Z = 52;                         % Lifting size
Imax = 50;                    
D_values = [6, 10];              % Two D values to test
lambda = 0;                     
q = 4;                           % Quantization bits for messages
q_tilde = 5;                     % Quantization bits for APPs
EbN0_dB = 1:0.5:4;              % SNR range
num_frames = 300;                % Frames per SNR point
mu = 2;                          % Quantization gain

%% Load small BG2 base matrix manually (for example purposes)
base_matrix_BG2 = [...
    0 1 -1;
    1 0 1
];

%% Build Expanded H matrix
[H_expanded, layers, col_degrees] = build_5G_LDPC(base_matrix_BG2, Z);
[M, N] = size(H_expanded);

%% Preallocate
FER = zeros(length(D_values), length(EbN0_dB));

%% Main Simulation Loop
for idx_D = 1:length(D_values)
    D = D_values(idx_D);
    fprintf('Simulating for D = %d\n', D);

    for idx_snr = 1:length(EbN0_dB)
        snr_db = EbN0_dB(idx_snr);
        snr_linear = 10^(snr_db/10);
        sigma = sqrt(1/(2*snr_linear));

        total_frame_errors = 0;

        for frame = 1:num_frames
            % Assume all-zero codeword
            tx_codeword = zeros(1,N);
            tx_signal = 1 - 2 * tx_codeword; % BPSK modulation
            rx_signal = tx_signal + sigma * randn(1,N);

            % Quantization
            Q = 2^(q-1) - 1;
            gamma = max(min(round(mu * rx_signal), Q), -Q);

            % Decode
            decoded_bits = IAMS_decoder_5G(gamma, H_expanded, layers, col_degrees, Imax, D, lambda, q, q_tilde);

            % Check if frame error
            if any(decoded_bits ~= tx_codeword)
                total_frame_errors = total_frame_errors + 1;
            end
        end

        FER(idx_D, idx_snr) = total_frame_errors / num_frames;
        fprintf('Eb/N0 = %.1f dB, FER = %.3e\n', snr_db, FER(idx_D, idx_snr));
    end
end

%% Plotting
figure;
colors = ['r', 'b'];
markers = ['^', 'v'];
hold on; grid on;
for idx_D = 1:length(D_values)
    semilogy(EbN0_dB, FER(idx_D,:), ['-' markers(idx_D) colors(idx_D)], 'LineWidth', 2);
end
xlabel('E_b/N_0 (dB)');
ylabel('Frame Error Rate (FER)');
title('IAMS LDPC Decoding - 5G LDPC Code (BG2, Z=52)');
legend('IAMS, D=6', 'IAMS, D=10', 'Location', 'southwest');
axis([1 4 1e-5 1]);
set(gca,'FontSize',12);
hold off;

%% --- Helper Functions ---
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