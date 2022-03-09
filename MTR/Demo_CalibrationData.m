
clc
close all
clear all
folder_now = pwd;  
% load data
addpath([folder_now, '\utils\']);
%% Generate data
task_num = 100;
sample_number = 400;
ni = ones(task_num, 1) * sample_number;       % sample size %
m  = length(ni);              % task number 
d  = 200;                     % dimension

XSigma = diag(0.5 * ones(d, 1)) + 0.5 * ones(d, d);  % 

sigma_max = 2;

% generated the model B
r = 3; % rank of B
sigma = 0.05;
L = randn(d,r)*sqrt(sigma);
R = randn(r,task_num)*sqrt(sigma); 
B = L*R;
% generated data
X = cell(m, 1);
Y = cell(m, 1);
opts.q = 1;
for tt = 1: m    
    % generate data matrices
    X{tt} = zeros(ni(tt), d);
    for ss = 1: ni(tt)
        X{tt}(ss, :) = mvnrnd(zeros(1, d), XSigma);
    end
    % generate targets.
    X{tt} = standardize(X{tt});
      sigma_i = 1;                  % d_1
    % sigma_i = 2^(- (tt-1)*3/100); % d_2
    % sigma_i = 2^(- (tt-1)*3/25);  % d_3
    % sigma_i = 2^(- (tt-1)/4);     % d_4
    Y{tt} = X{tt} * B(:, tt) + randn(ni(tt), 1) * sigma_i * sigma_max;
    Yt{tt} = X{tt} * B(:, tt);  % test data
    fprintf('Task %u sigma: %.4f\n', tt, sigma_i * sigma_max);
end

Repet = 10;

for j =1:Repet
% generated training data and test data
training_percent = 0.6;
% generate training data
[X_tr, Y_tr, ~, ~] = mtSplitPerc(X, Y, training_percent);
% generated testa data 
[~, ~, X_te, Y_te] = mtSplitPerc(X, Yt, training_percent);

%=======Test algorithm=====
param_range = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000];  % the range of regularization parameters
opts.maxIter = 20;  
opts.q = 0.5;

do_RLog = 1; % conduct CMTL-Log
do_Lp = 1;   % conduct CMTL-Lp

if do_RLog == 1;
    disp('============================Test RLog norm based method===================')
    for lami = 1:length(param_range)
    lambda =  param_range(lami);
    % CMTL-Log
    [W, funcVal] = TCLog_Nonconvex(X_tr, Y_tr, lambda,opts);
    CLogmse(lami) = eval_MTL_mse (Y_te, X_te, W);
    end
end
TCLogmse(j,:) = CLogmse;
Out.CRLog = TCLogmse;

if do_Lp == 1;
    
    disp('============================Test Lp norm based method===================')

    for lami = 1:length(param_range)
    lambda =  param_range(lami);
    % Lp with p = 0.01;
    opts.p =0.01;
    [W11, funcVal] = TCLP_Nonconvex(X_tr, Y_tr, lambda,opts);
    CLp001(lami) = eval_MTL_mse (Y_te, X_te, W11);
    
    % Lp with p = 0.5;
    opts.p =0.5;
    [W21, funcVal] = TCLP_Nonconvex(X_tr, Y_tr, lambda,opts);
    CLp05(lami) = eval_MTL_mse (Y_te, X_te, W21);
    
    % Lp with p = 1;
    opts.p =1;
    [W31, funcVal] = TCLP_Nonconvex(X_tr, Y_tr, lambda,opts);
    CLp1(lami) = eval_MTL_mse (Y_te, X_te, W31);
    end
end
TCLp001(j,:) = CLp001;

TCLp05(j,:) = CLp05;

TCLp1(j,:) = CLp1;
end

% Standed error

std_CRLog = std(TCLogmse);

std_TCLp001= std(TCLp001);

std_TCLp05= std(TCLp05);

std_TCLp1= std(TCLp1);