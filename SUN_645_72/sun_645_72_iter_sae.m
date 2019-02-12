clc, clear all,  close all
%% data prepare
addpath('library');
load('label.mat');
load('class_attr.mat');
load ('attributeLabels_continuous.mat');
%load('images_features.mat');
load('sco_2048.mat');
img_attr=labels_cv;
total=1;
attr_num=102;
test_class_num=72;
total_tr_num=1;
total_max_acc=zeros(total_tr_num,10);
avg_init_acc=zeros(total_tr_num,1);
total_class_num=717;
% [cof,sco]=pca(F);
score_num=460;
load('rand_class_141.mat');

train_acc=zeros(total_tr_num,1);
while total<= total_tr_num;
    F_=sco(:,1:score_num);
    train_iter=1;
    while train_iter<=10
    if train_iter==10
        test_class=rand_class(end-test_class_num+1:end);
    else
        test_class=rand_class((1:test_class_num)+(train_iter-1)*test_class_num);
    end
    train_class=1:total_class_num; 
    train_class(test_class)=[];
    test_imgs=[];
    for ite=1:test_class_num
        tmp=find(Y==test_class(ite));
        test_imgs=[test_imgs;tmp];
    end
    train_imgs=1:length(Y);
    train_imgs(test_imgs)=[];
    X_tr=F_(train_imgs,:);
    X_te=F_(test_imgs,:);
    S_tr=img_attr(train_imgs,:);
    S_te_pro=class_attr(test_class,:);
    %% main code
    param.HITK           = 1;
    param.testclasses_id = test_class;
    param.test_labels    = Y(test_imgs);
    %%------------------------------------------------------Abel's code---------------------------------------------------------------------
    X = [X_tr', X_te'];
    X = X';
    [cof,sco]=princomp(X);
    X_tr=sco(1:length(X_tr),1:score_num);
    X_te = sco(length(X_tr)+1:length(X),1:score_num);

    X_tr = NormalizeFea(X_tr')';
    X_te1 = NormalizeFea(X_te')';
    [train_num, score_num] = size(X_tr);
    [test_num, score_num] = size(X_te);
filename=strcat('hit_class', num2str(train_iter + 1), '.mat');
load(filename);
%     lambda = 500; 
%     alpha = 0.05; delimit = 0.01;  % 68.78%
    lambda = 500; 
    alpha = 0.05; delimit = 0.01;


    W = SAE_old(X_tr', S_tr', lambda);
    max_iter = 5;
    t = 1;

    while t<=max_iter
        a_last = zeros(attr_num, attr_num);
        b_last = zeros(score_num, score_num);
        c_last = zeros(attr_num, score_num);
        l = zeros(test_num, test_class_num);
        n = zeros(test_num, test_class_num);
        x_te = X_te1';
        y_te = S_te_pro';
        if t==1
            % SAE
            %[S --> F], projecting from semantic to visual space
            X_te_pro     = NormalizeFea( S_te_pro')' * NormalizeFea(W);
            [zsl_accuracy]= zsl_el(X_te, X_te_pro, param); 
            fprintf('SUN ZSL accuracy [S >>> V]: %.1f%%\n', zsl_accuracy*100);
            avg_init_acc(total)=avg_init_acc(total)+zsl_accuracy*100;
        end;
        for i=1:test_num
            if isempty(hit_class{i})
                hit_class{i} = 1:total_class_num;
                fprintf('%d~~~~~~~~~~~~~\n', i);
            end
            %hit_class{i} = 1:total_class_num;
        hit_index = find(ismember(test_class,hit_class{i}));
            decoder = sum((ones(test_class_num, 1)*NormalizeFea(X_te(i,:)) - NormalizeFea( S_te_pro * W)).^2,2)';
            encoder = alpha*sum((NormalizeFea(W * X_te(i,:)')*ones(1, test_class_num) - NormalizeFea( S_te_pro)').^2,1);
            l(i,:)= decoder + encoder;
            min_l = min(l(i,hit_index));
            idx_tmp = find(l(i,hit_index)-min_l < delimit);
            min_sum = numel(idx_tmp);
            n(i, idx_tmp) = 1/min_sum;
            idx_tmp=find(n(i,hit_index)>eps);
            a = y_te(:,idx_tmp) * diag(n(i, idx_tmp)) * y_te(:,idx_tmp)';
            b = x_te(:,i) *  ones(1, numel(idx_tmp)) * diag(n(i, idx_tmp)) * ones(numel(idx_tmp), 1) * x_te(:,i)';
            c = y_te(:,idx_tmp) * diag(n(i, idx_tmp)) * ones(numel(idx_tmp), 1) * x_te(:,i)';
            a_last = a_last + a;
            b_last = b_last + alpha * b;
            c_last = c_last + c;
        end;
        a_last = a_last + alpha * (S_tr' * S_tr);
        b_last = b_last + alpha * lambda * (X_tr' * X_tr);
        c_last = c_last + alpha * (S_tr' * X_tr);
        W = sylvester(a_last, b_last, c_last);
        W = NormalizeFea(W);
        % SAE
        %[S --> F], projecting from semantic to visual space
        X_te_pro     = NormalizeFea(NormalizeFea( S_te_pro')' * W);
        [zsl_accuracy]= zsl_el(X_te1, X_te_pro, param); 
        fprintf('SUN ZSL accuracy [S >>> V]: %.1f%%\n', zsl_accuracy*100);

        t=t+1;
        end;
        train_acc(total)=train_acc(total)+zsl_accuracy*100;
        train_iter = train_iter+1;
    end;
    train_acc(total)=train_acc(total)/(train_iter-1);
    avg_init_acc(total)=avg_init_acc(total)/(train_iter-1);
    fprintf('Average Accuracy of the %d time: %.2f%%\n',total,train_acc(total));
    fprintf('Average Initial Accuracy of the %d time: %.2f%%\n',total,avg_init_acc(total));
    total=total+1;
end;