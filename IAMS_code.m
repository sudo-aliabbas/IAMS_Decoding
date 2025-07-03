clc; clear;

% === Load Base Graph 2 and expand H ===
load('official_BG2_base_graph.mat');  % should contain BG2 (42 x 52)
Z = 52;
H = expandBaseGraph(BG2, Z);          % 2184 × 2704
[M, N] = size(H);
fprintf("Expanded H size: %d x %d\n", M, N);

% === All-zeros valid codeword
codeword = zeros(N, 1);

% === Flip 1 bit to introduce a single-bit error
error_pos = 1000;   % Flip bit #1000 (you can change this)
corrupted = codeword;
corrupted(error_pos) = 1;  % Bit-flip (0 → 1)

% === Create LLR from corrupted word
tx = 1 - 2 * corrupted;
rx = tx + 0.3 * randn(size(tx));     % Simulate AWGN
llr = 2 * rx;
llr = max(min(round(llr), 31), -31);  % Quantize to 6-bit

% === Decode using IAMS
max_iter = 25;
decoded = IAMSDecoder(llr, H, max_iter);

% === Compare
fprintf("Flipped bit #%d\n", error_pos);
disp("Decoded codeword (first 20 bits):"); disp(decoded(1:20)');

% === Check if decoder corrected the error
if all(decoded == codeword)
    disp("Decoder successfully corrected the 1-bit error.");
elseif all(mod(H * decoded(:), 2) == 0)
    disp("Decoding resulted in another valid codeword (but not original).");
else
    disp("Decoder failed to correct the error.");
end



function H = expandBaseGraph(BG, Z)
    [m, n] = size(BG);
    H = zeros(m*Z, n*Z);

    for i = 1:m
        for j = 1:n
            if BG(i,j) == -1
                H((i-1)*Z+1:i*Z, (j-1)*Z+1:j*Z) = zeros(Z);
            else
                I = eye(Z);
                H((i-1)*Z+1:i*Z, (j-1)*Z+1:j*Z) = circshift(I, [0 BG(i,j)]);
            end
        end
    end
end


function decoded = IAMSDecoder(llr, H, max_iter)
    [M, N] = size(H);
    gamma = llr(:);
    gamma_tilde = gamma;
    alpha = sparse(M, N); beta = sparse(N, M);
    lambda = 1; tau = 1;

    % Build Tanner Graph + compute column degrees
    CNs = cell(M,1); VNs = cell(N,1); col_deg = zeros(1,N);
    for m = 1:M, CNs{m} = find(H(m,:) ~= 0); end
    for n = 1:N
        VNs{n} = find(H(:,n) ~= 0);
        col_deg(n) = length(VNs{n});
    end

    % Scaling factor D(n)
    D_scale = ones(1, N);
    for n = 1:N
        d = col_deg(n);
        if d == 2
            D_scale(n) = 0.9;
        elseif d == 3
            D_scale(n) = 0.85;
        elseif d >= 4
            D_scale(n) = 0.8;
        end
    end

    % Column-degree adaptation threshold D = 6
    D_thresh = 6;

    for iter = 1:max_iter
        for m = 1:M
            nlist = CNs{m}; E = length(nlist);

            % Step 1: VN → CN (beta)
            for k = 1:E
                n = nlist(k);
                beta(n,m) = gamma_tilde(n) - alpha(m,n);
            end

            % Step 2: CN → VN (min-sum)
            beta_vals = abs(beta(nlist, m));
            signs = sign(prod(beta(nlist, m))) * sign(beta(nlist, m));
            [min1, idx1] = min(beta_vals);
            temp = beta_vals; temp(idx1) = inf;
            [min2, idx2_safe] = min(temp);

            for k = 1:E
                n = nlist(k);

                % Column-degree adaptation: skip update if deg ≥ D in early iterations
                if iter < 4 && col_deg(n) >= D_thresh
                    continue;  % Skip α(m,n) update
                end

                % Normal IAMS update with scaling
                if k == idx1
                    msg = tau * min2;
                elseif k == idx2_safe
                    msg = tau * min1;
                elseif min1 == min2
                    msg = tau * max(min1 - lambda, 0);
                else
                    msg = tau * min1;
                end
                alpha(m,n) = D_scale(n) * msg * signs(k);
            end

            % Step 3: APP update
            for k = 1:E
                n = nlist(k);
                gamma_tilde(n) = beta(n,m) + alpha(m,n);
            end
        end

        decoded = gamma_tilde < 0;
        if all(mod(H * decoded(:), 2) == 0)
            return;
        end
    end
end

