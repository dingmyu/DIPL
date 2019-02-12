function [ W ] = SAE(X,S,lambda )
%SAE Summary of this function goes here
%   Detailed explanation goes here
A=S*S';
B=lambda*X*X';
C=(1+lambda)*S*X';
% A=NormalizeFea(S*S');
% B=lambda*NormalizeFea(X*X');
% C=(1+lambda)*NormalizeFea(S*X');
W=sylvester(A,B,C);

end

