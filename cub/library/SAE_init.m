function [ W ] = SAE(X,S,lambda )
%SAE Summary of this function goes here
%   Detailed explanation goes here
A=S*S';
B=lambda*X*X';
C=(1+lambda)*S*X';
W=sylvester(A,B,C);

end

