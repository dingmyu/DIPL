function [ W ] = SAE(X, S, alpha)
    % SAE is Semantic Auto-encoder
    % Inputs:
    %    X: dxN data matrix.
    %    S: kxN semantic matrix.
    %    lambda: regularisation parameter.
    %
    % Return: 
    %    W: kxd projection matrix.

    A = (1-alpha)*S*S';
    B = alpha*X*X';
    C = S*X'; 
    W = sylvester(A,B,C);
end

