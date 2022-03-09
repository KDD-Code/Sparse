%min_||X'W+1b'-Y-Y.*M||_{2,1}+gamma||W||_{2,p}^p
%s.t. M>=0

function [W,ind] = IRW_FS (X,gt,p,gamma)
[dim,n] = size (X);
c = max(gt);
Y = zeros(n,c);
for i = 1:n
    Y(i,gt(i)) = 1;
end

%% ===================== Initialization =====================
[W, b] = least_squares_regression(X,Y',gamma);   % We use the soultion to the standard least squares regreesion as the initial solution  

%% =======================  updating  =======================
%iter = 1;
%err = 1;
%while err > 1e-3
Maxiter = 50;
for iter = 1:Maxiter
temp = (X'*W+ones(n,1)*b'-Y).*Y;
M = max(temp,zeros(n,c));
diag1 = zeros(n,1);
diag2 = zeros(dim,1);
XT = X';
for j = 1:n
    med = XT(j,:)*W+(b'-Y(j,:)-Y(j,:).*M(j,:));
    diag1(j) = 0.5*(med*med'+eps)^(-0.5);
end
for k = 1:dim
    diag2(k) = 0.5*p*(W(k,:)*W(k,:)'+eps)^(0.5*(p-2));
end
D1 = diag(diag1);
D2 = diag(diag2);
W = inv(X*D1*X'+gamma*D2)*(X*D1*(Y+Y.*M-ones(n,1)*b'));
b = (-X'*W+Y+Y.*M)'*D1*ones(n,1)/(ones(1,n)*D1*ones(n,1));
loss = X'*W + ones(n,1)*b'-Y-Y.*M;
sqW = (W.^2);
sumW = sum(sqW,2);
obj(iter) = sum(sqrt(sum(loss.*loss,2)))+gamma*sum(sumW.^(0.5*p));
if iter > 1
    err = abs(obj(iter-1)-obj(iter));
end
%fprintf('Iteration count = %d, obj.IRW_FS = %f\n', iter, obj(iter));
%iter = iter + 1; 
end

[~,ind] = sort(sumW,'descend');

plot(obj)



