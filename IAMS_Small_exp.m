clc; clear;

% === 1. Small LDPC parity-check matrix (3x6)
H = [
    1 1 0 1 0 0;
    0 1 1 0 1 0;
    1 0 1 0 0 1
];  % Defines a (6,3) code

% === 2. Known valid codeword (satisfies H * c = 0 mod 2)
codeword = [1; 1; 0; 0; 1; 1];
assert(all(mod(H * codeword, 2) == 0), "❌ Codeword is not valid under H");

% === 3. Introduce a 1-bit error (flip bit 3 for example)
error_pos = 3;
corrupted = codeword;
corrupted(error_pos) = ~corrupted(error_pos);  % flip 0 ↔ 1

fprintf("Flipping bit #%d: introducing 1-bit error...\n", error_pos);
fprintf("Original : "); disp(codeword');
fprintf("Corrupted: "); disp(corrupted');

% === 4. Convert to LLRs (simulate transmission)
tx = 1 - 2 * corrupted;
rx = tx + 0.3 * randn(size(tx));   % add some noise (moderate)
llr = 2 * rx;                      % compute LLRs

% === 5. Decode with IAMS
decoded = IAMSDecoder(llr, H, 10);

% === 6. Check results
disp("Decoded codeword:"); disp(decoded');
disp("Syndrome (H * decoded mod 2):"); disp(mod(H * decoded(:), 2));

if all(decoded == codeword)
    disp("PASS: IAMS decoder correctly corrected the 1-bit error.");
elseif all(mod(H * decoded(:), 2) == 0)
    disp("Partial Pass: Syndrome is zero, but codeword doesn't match — decoder found a different valid codeword.");
else
    disp("FAIL: Decoder did not correct the error.");
end



function decoded = IAMSDecoder(llr, H, max_iter)
    [M, N] = size(H);
    gamma = llr(:);
    gamma_tilde = gamma;
    alpha = sparse(M, N); 
    beta = sparse(N, M);
    lambda = 1; tau = 1;

    CNs = cell(M,1); VNs = cell(N,1);
    for m = 1:M, CNs{m} = find(H(m,:) ~= 0); end
    for n = 1:N, VNs{n} = find(H(:,n) ~= 0); end

    for iter = 1:max_iter
        for m = 1:M
            nlist = CNs{m}; E = length(nlist);
            for k = 1:E
                n = nlist(k);
                beta(n,m) = gamma_tilde(n) - alpha(m,n);
            end

            beta_vals = abs(beta(nlist, m));
            signs = sign(prod(beta(nlist, m))) * sign(beta(nlist, m));
            [min1, idx1] = min(beta_vals);
            temp = beta_vals; temp(idx1) = inf;
            [min2, idx2_safe] = min(temp);

            for k = 1:E
                n = nlist(k);
                if k == idx1
                    msg = tau * min2;
                elseif k == idx2_safe
                    msg = tau * min1;
                elseif min1 == min2
                    msg = tau * max(min1 - lambda, 0);
                else
                    msg = tau * min1;
                end
                alpha(m,n) = msg * signs(k);
            end

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

