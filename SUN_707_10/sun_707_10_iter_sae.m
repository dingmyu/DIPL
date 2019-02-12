clc, clear all,  close all
%% data prepare
addpath('library');
load('label.mat');
load('class_attr.mat');
load ('attributeLabels_continuous.mat');
%load('images_features.mat');
load('sco.mat');
img_attr=labels_cv;
attr_num=102;
test_class_num=10;
total_class_num=717;
%[cof,sco]=pca(F,'NumComponents',score_num);
%load('F_200.mat');
test_class=[351,265,373,462,152,415,378,573,31,24];
train_class=1:total_class_num; 
train_class(test_class)=[];
train_acc=zeros(50,1);
avg_init_acc=zeros(50,1);
test_imgs=[];
for ite=1:test_class_num
    tmp=find(Y==test_class(ite));
    test_imgs=[test_imgs;tmp];
end
train_imgs=1:length(Y);
train_imgs(test_imgs)=[];
total=1;
while total<=1
    score_num=600;
    F_=sco(:,1:score_num);
    X_tr=F_(train_imgs,:);
    X_te=F_(test_imgs,:);
    S_tr=img_attr(train_imgs,:);
    S_te_pro=class_attr(test_class,:);
    %% main code
    param.HITK           = 1;
    param.testclasses_id = test_class;
    param.test_labels    = Y(test_imgs);
    
    X_tr = NormalizeFea(X_tr')';
    X_te1 = NormalizeFea(X_te')';
    [train_num, score_num] = size(X_tr);
    [test_num, score_num] = size(X_te);
%     lambda = 560; 
%     alpha = 0.005; delimit = 0.001;  %93% 93%
    lambda = 560; 
    alpha = 0.005; delimit = 0.0001;  %92.5% 92.5%

    W = SAE(X_tr', S_tr', lambda);
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
            %[F --> S], projecting data from feature space to semantic sapce 
            S_est        = X_te * NormalizeFea(W)'; 
            [zsl_accuracy, Y_hit5] = zsl_el((S_est), S_te_pro, param); 

            fprintf('\n[1] SUN ZSL accuracy [V >>> S]: %.1f%%\n', zsl_accuracy*100);

            %[S --> F], projecting from semantic to visual space
            X_te_pro     = NormalizeFea( S_te_pro')' * NormalizeFea(W);
            [zsl_accuracy]= zsl_el(X_te, X_te_pro, param); 
            fprintf('[2] SUN ZSL accuracy [S >>> V]: %.1f%%\n', zsl_accuracy*100);
        end;
        for i=1:test_num
            decoder = sum((ones(test_class_num, 1)*NormalizeFea(X_te(i,:)) - NormalizeFea( S_te_pro * W)).^2,2)';
            encoder = alpha*sum((NormalizeFea(W * X_te(i,:)')*ones(1, test_class_num) - NormalizeFea( S_te_pro)').^2,1);
            l(i,:)= decoder + encoder;
            min_l = min(l(i,:));
            idx_tmp = find(l(i,:)-min_l < delimit);
            min_sum = numel(idx_tmp);
            n(i, idx_tmp) = 1/min_sum;
            idx_tmp=find(n(i,:)>eps);
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
        %[F --> S], projecting data from feature space to semantic sapce 
        S_est        = NormalizeFea(X_te1 * W'); 
        [zsl_accuracy, Y_hit5] = zsl_el((S_est), S_te_pro, param); 

        fprintf('\n[1]SUN ZSL accuracy [V >>> S]: %.1f%%\n', zsl_accuracy*100);

        %[S --> F], projecting from semantic to visual space
        X_te_pro     = NormalizeFea(NormalizeFea( S_te_pro')' * W);
        [zsl_accuracy]= zsl_el(X_te1, X_te_pro, param); 
        fprintf('[2] SUN ZSL accuracy [S >>> V]: %.1f%%\n', zsl_accuracy*100);

        t=t+1;
    end;
    train_acc(total)=zsl_accuracy*100;
    total = total+1;
end;