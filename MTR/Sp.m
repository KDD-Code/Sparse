% subgradient of function y = x^(p/2)
%
function [S] = Sp(S,p)
temp = p/2*(S).^(p/2-1);
S = temp;