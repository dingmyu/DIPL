clc, clear all,  close all
%% data prepare
addpath('library');
load('te_attr.mat');
load('te_fea.mat');
load('te_labels.mat');
load ('tr_attr.mat');
load('tr_fea.mat');
load('te_class_attr.mat');
%load('sco.mat');
train_num=1;
train_acc=zeros(train_num+1,1);
avg_init_acc=zeros(train_num+1,1);
total=1;
attr_num=64;
test_class_num=12;
score_num=600;
%% main code
param.HITK           = 1;
param.testclasses_id = 21:32;
param.test_labels    = te_labels;

S_tr=tr_attr;
S_te_pro=te_class_attr;
S_te_gt=te_class_attr;
X_tr=tr_fea;
X_te=te_fea;
X = [X_tr', X_te'];
X = X';
[cof,sco]=pca(X,'NumComponents',score_num);
F_=sco(:,1:score_num);
while total<=train_num
    X_tr=F_(1:length(X_tr),1:score_num);
    X_te = F_(length(X_tr)+1:length(X),1:score_num);

[train_num, pic_fea_num] = size(X_tr);
[test_num, pic_fea_num] = size(X_te);
[train_num, sem_fea_num] = size(S_tr);
[classes_num, sem_fea_num] = size(S_te_pro);

X_tr    = NormalizeFea(X_tr')';
X_te1    = NormalizeFea(X_te')';
beta = 0.04; alpha =  0;
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
        S_est        = X_te * W';
        S_est = NormalizeFea(S_est);
        S_te_gt = NormalizeFea(S_te_gt);
        [zsl_accuracy, Y_hit5] = zsl_el((S_est), S_te_gt, param); 
        fprintf('\n[%d] AwA ZSL accuracy [V >>> S]: %.1f%%\n', t, zsl_accuracy*100);
        X_te_pro     = NormalizeFea( S_te_pro) * W;
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
                b{i}=b{i}+n(i,j,t)*X_te1(i,:)'*X_te1(i,:);
            end
        end
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
        W_old = NormalizeFea(W);
        W = sylvester(A,B,C);
        W1 = NormalizeFea(W);
        sum(sum((W1-W_old)*(W1-W_old)'))/sum(sum(W1*W1'))
        %W = NormalizeFea(W);
    end
    X_te2 = NormalizeFea( X_te);
    %X_te2 = NormalizeFea(X_te2')';
    %[F --> S], projecting data from feature space to semantic sapce 
    %用特征生成属性
    S_est        = X_te * W';
    S_est = NormalizeFea(S_est);
    S_te_gt = NormalizeFea(S_te_gt);
    [zsl_accuracy, Y_hit5] = zsl_el((S_est), S_te_gt, param); 
    fprintf('\n[%d] AwA ZSL accuracy [V >>> S]: %.1f%%\n', t, zsl_accuracy*100);
    %[S --> F], projecting from semantic to visual space
    %为每一类生成特征
    X_te_pro     = S_te_pro * W;
    X_te_pro = NormalizeFea( NormalizeFea(X_te_pro')'+0.01);
    [zsl_accuracy]= zsl_el(X_te2, X_te_pro, param); 
    fprintf('[%d] AwA ZSL accuracy [S >>> V]: %.1f%%\n', t, zsl_accuracy*100); 
    %filename=strcat('dmy_W',num2str(t),'.mat');
    %save(filename,'W');
    t = t+1;
end
    train_acc(total)=zsl_accuracy*100;
    total = total+1;
end;