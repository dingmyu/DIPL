%% %%% AwA DEMO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Following code shows a demo for AwA dataset to reproduce the result of the paper:
% 
% Semantic Autoencoder for Zero-shot Learning. 
% 
% Elyor Kodirov, Tao Xiang, and Shaogang Gong
% To appear in CVPR 2017.
% 
%
% You are supposed to get following:
% [1] AwA ZSL accuracy [V >>> S]: 84.7%
% [2] AwA ZSL accuracy [S >>> V]: 84.0%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% We used GoogleNet features.
clc, clear all,  close all

addpath('library');

%%%%% Load the data
load('awa_demo_data.mat');
%S_te_gt = S_te_pro  10*85,  每类的语义特征
%X_tr 24295*1024 训练集的图片特征
%X_te 6180*1024  测试集的图片特征
%S_tr 24295*85  训练集的语义特征

%主成分分析  1024-〉100
X = [X_tr', X_te'];
X = X';
[cof,sco]=princomp(X);
X_tr=sco(1:length(X_tr),1:300);
X_te = sco(length(X_tr)+1:length(X),1:300);


[train_num, pic_fea_num] = size(X_tr);
[test_num, pic_fea_num] = size(X_te);
[train_num, sem_fea_num] = size(S_tr);
[classes_num, sem_fea_num] = size(S_te_pro);

%%%%% Test %%%%% 
param.HITK           = 1;
param.testclasses_id = param.testclasses_id;
param.test_labels    = param.test_labels;

%%%%% Training 
X_tr    = NormalizeFea(X_tr')';
X_te1    = NormalizeFea(X_te')';
beta = 0.0001; alpha =  0;
if alpha==0 || alpha==1
    W=pinv(S_tr)*X_tr;
else
    W = SAE(X_tr', S_tr', alpha);
end
t=1; max_iter = 4; delimit=0.02;
n=zeros(test_num,classes_num,max_iter);
while(t<=max_iter)
    A=zeros(sem_fea_num,sem_fea_num);
    B=zeros(pic_fea_num,pic_fea_num);
    C=zeros(sem_fea_num,pic_fea_num);
    l=zeros(test_num,classes_num);
    a={};b={};c={};
    if (t==1)
        X_te2 = NormalizeFea( X_te);
        %X_te2 = NormalizeFea(X_te2')';
        S_est        = X_te2 * NormalizeFea(W)';
        S_est = NormalizeFea(S_est);
        S_te_gt = NormalizeFea(S_te_gt);
        [zsl_accuracy, Y_hit5] = zsl_el((S_est), S_te_gt, param); 
        fprintf('\n[%d] AwA ZSL accuracy [V >>> S]: %.1f%%\n', t, zsl_accuracy*100);
        X_te_pro     = NormalizeFea( S_te_pro')' * NormalizeFea(W);
        X_te_pro = NormalizeFea( X_te_pro);
        [zsl_accuracy]= zsl_el(X_te2, X_te_pro, param); 
        fprintf('[%d] AwA ZSL accuracy [S >>> V]: %.1f%%\n', t, zsl_accuracy*100);
    end
    for i = 1:test_num
        l(i,:)= (1-alpha)*sum((NormalizeFea(ones(classes_num,1)*X_te(i,:))-NormalizeFea(S_te_pro*W)).^2,2)' + alpha*sum((NormalizeFea((W*X_te(i,:)'*ones(1,classes_num))')'-NormalizeFea(S_te_pro)').^2,1);
        min_l=min(l(i,:));
        min_fre=0;
        for j=1:classes_num
			if abs(min_l-l(i,j))/min_l<delimit
				min_fre=min_fre+1;
			end
        end;
        %fprintf('%d\n', min_fre);
        for j=1:classes_num
            if(abs(min_l-l(i,j))/min_l<delimit)
                n(i,j,t)=1/min_fre;
            else
                n(i,j,t)=0;
            end
        end
        a{i}=zeros(sem_fea_num,sem_fea_num);
        b{i}=zeros(pic_fea_num,pic_fea_num);
        c{i}=zeros(sem_fea_num,pic_fea_num);    
        for j=1:classes_num				
            if(n(i,j,t))
                c{i}=c{i}+n(i,j,t)*S_te_pro(j,:)'*X_te1(i,:);
                a{i}=a{i}+n(i,j,t)*S_te_pro(j,:)'*S_te_pro(j,:);
                %b{i}=b{i}+n(i,j,t)*X_te1(i,:)'*X_te1(i,:);
            end
        end
        b{i}=b{i}+X_te1(i,:)'*X_te1(i,:);
        A=A+(1-alpha)*a{i};
        B=B+alpha*b{i};
        C=C+c{i};
    end
    A = A+ beta*(1-alpha)*(S_tr'*S_tr);
    B = B+ beta*alpha*(X_tr'*X_tr);
    C = C+ beta*S_tr'*X_tr; 
    if alpha==1
        W = C*pinv(B);
    elseif alpha==0
        W = pinv(A)*C;
    else
        W = sylvester(A,B,C);
        W = NormalizeFea(W);
    end
    X_te2 = NormalizeFea( X_te);
    %X_te2 = NormalizeFea(X_te2')';
    %[F --> S], projecting data from feature space to semantic sapce 
    %用特征生成属性
    S_est        = X_te2 * W';
    S_est = NormalizeFea(S_est);
    S_te_gt = NormalizeFea(S_te_gt);
    [zsl_accuracy, Y_hit5] = zsl_el((S_est), S_te_gt, param); 
    fprintf('\n[%d] AwA ZSL accuracy [V >>> S]: %.1f%%\n', t, zsl_accuracy*100);
    %[S --> F], projecting from semantic to visual space
    %为每一类生成特征
    X_te_pro     = NormalizeFea( S_te_pro')' * W;
    X_te_pro = NormalizeFea( X_te_pro);
    [zsl_accuracy]= zsl_el(X_te2, X_te_pro, param); 
    fprintf('[%d] AwA ZSL accuracy [S >>> V]: %.1f%%\n', t, zsl_accuracy*100); 
    %filename=strcat('dmy_W',num2str(t),'.mat');
    %save(filename,'W');
    t = t+1;
end







