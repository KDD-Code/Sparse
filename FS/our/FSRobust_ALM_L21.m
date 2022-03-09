% min_{W,M}  \lambda||X'*W+1*b'-Y-Y.*M||_21 +  \|W\|_21
function [feature_idx, W, b, obj] = FSRobust_ALM_L21(X, Y, NITER)
% X: d*n data matrix, each column is a data point
% Y: n*c label matrix, Y(i,j)=1 if xi is labeled to j, and Y(i,j)=0 otherwise
% k: number of selected features
% mu, rho: parameters in the ALM optimization method
% NITER: iteration number
% feature_idx: indices of selected features
% W: d*c embedding matrix
% b: c*1 bias vector
% obj: objective values in the iterations

% Ref:
% Xiao Cai, Feiping Nie, Heng Huang. 
% Exact Top-k Feature Selection via l2,0-Norm Constraint. 
% The 23rd International Joint Conference on Artificial Intelligence (IJCAI), 2013.

% ============ The parameters need to be tuned ======================
mu = 1e-4;
rho = 1.5;
lambda = 0.1; % the regularizer parameter over 
% ==================================================================
obj = zeros(NITER,1);
[d, n] = size(X);
c = max(Y);
Xm = X-mean(X,2)*ones(1,n);

Lambda = zeros(d,c);
Sigma = zeros(n,c);
V = rand(d,c);
E = rand(n,c);
gt= Y;
Y = zeros(n,c);
for i = 1:n
    Y(i,gt(i)) = 1;
end

[W, b] = least_squares_regression(X,Y',lambda);

inXX = Xm*inv(Xm'*Xm+eye(n));
for iter = 1:NITER
    inmu = 1/mu;
    % update M
    temp = (X'*W+ones(n,1)*b'-Y-E+inmu*Sigma).*Y;
    M = max(temp,zeros(n,c));
    YM = M.*Y;
    % update b
    tem = Y+YM+E-inmu*Sigma;
    b = mean(tem)';
    V1 = (V-inmu*Lambda+Xm*tem); 
    W = V1 - inXX*(Xm'*V1);
    %Wg = XX*W - (V-inmu*Lambda+Xm*tem); st = trace(Wg'*Wg)/trace(Wg'*(XX*Wg)); W = W - st*Wg;
    
    % update V
    WL = W+inmu*Lambda;
    for i=1:d
        wl = WL(i,:);
        la = sqrt(wl*wl');
        lam = 0;
        if la>inmu
            lam = 1-inmu/la;
        elseif la< -inmu
            lam = 1+inmu/la;
        end
        V(i,:) = lam*wl;
    end
    
    % update E
    XW = Xm'*W+ones(n,1)*b'-Y-M.*Y;
    XWY = XW+inmu*Sigma;
    for i = 1:n
        w = XWY(i,:);
        la = sqrt(w*w');
        lam = 0;
        if la > inmu*lambda
            lam = 1-inmu*lambda/la;
        elseif la < -inmu*lambda
            lam = 1+inmu*lambda/la;
        end;
        E(i,:) = lam*w;
    end;
    
    Lambda = Lambda + mu*(W-V);
    Sigma = Sigma + mu*(XW-E);   
    mu = min(10^10,rho*mu);
    err = Xm'*V+ones(n,1)*b'-Y-M.*Y;
    obj(iter) = sum(sqrt(sum(err.*err,2)))+sum(sqrt(sum(W.*W,2)));
    error1(iter) = norm(W-V);
    error2(iter) = norm(XW-E);
end;
sqW = (W.^2);
error1
error2
[~,feature_idx] = sort(sum(sqW,2),'descend');
