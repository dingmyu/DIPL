clc, clear all,  close all
%% data prepare
addpath('library');
load ('img_name_label.mat');
load ('cu_class_attr.mat');
load('img_attr.mat');
load('idx_old2new');
load('rand_class_97.mat');
load('sco_resnet_feature_2048.mat');
total_train_num=1;
vec=class_attr;
test_class_num = 50;
total_class_num=200;
attr_num=312;
score_num=410;
total=1;
k_class=60;
% avg_init_acc=zeros(total_train_num,1);
F_=sco(:,1:score_num);
train_acc_1=zeros(total_train_num,1);
train_acc_2=zeros(total_train_num,1);
abel_label=label;
abel_centSet=zeros(k_class,attr_num,total_train_num);
abel_clusterFlag=zeros(total_class_num,total_train_num);
while total <= total_train_num
    [clusterFlag,centSet]=kmeans(class_attr,k_class,'EmptyAction','drop');
    abel_centSet(:,:,total)=centSet;
    abel_clusterFlag(:,total)=clusterFlag;
    for i=1:k_class
        tmpidx=find(clusterFlag==i);
        for j=1:length(tmpidx)
            abel_tmpidx=find(label+1==tmpidx(j));
            abel_label(abel_tmpidx,1)=i;
        end
    end
    train_iter=1;
    while train_iter<=4
        test_class=rand_class((1:test_class_num)+(train_iter-1)*50);
        train_class=1:200; 
        train_class(test_class)=[];
        test_imgs=[];
        for ite=1:test_class_num
            tmp=find(label+1==test_class(ite));
            test_imgs=[test_imgs;tmp];
        end  
        abel_tmp_test_label = abel_label(test_imgs);
        tmp_test_label=abel_tmp_test_label;
        unique_abel = unique(tmp_test_label);
        k_class = numel(unique_abel);
        for i=1:k_class
            tmp_test_label(abel_tmp_test_label==unique_abel(i)) = i;
        end
        param.HITK           = 1;
        param.testclasses_id = 1:k_class;
        param.test_labels    = tmp_test_label;
        train_imgs=1:length(label);
        train_imgs(test_imgs)=[];
        X_tr=F_(train_imgs,:);
        X_te=F_(test_imgs,:);
        S_tr=img_attr(idx(train_imgs),:);
        S_te_pro=centSet(unique_abel,:);
        S_te_gt = S_te_pro;
        %% main code
        X_te1 = NormalizeFea(X_te')';
        [train_num, score_num] = size(X_tr);
        [test_num, score_num] = size(X_te);
        lambda = 500000;
%         alpha = 0.06; delimit = 0.01; % 44.43% 48.39%
        alpha = 0.006; delimit = 0.01;  % 51.92% 56.36%
%         alpha = 1; delimit = 0.01;
        W = SAE(X_tr', S_tr', lambda);
        max_iter = 5;
        t = 1;
        while t<=max_iter
            a_last = zeros(attr_num, attr_num);
            b_last = zeros(score_num, score_num);
            c_last = zeros(attr_num, score_num);
            l = zeros(test_num, k_class);
            n = zeros(test_num, k_class);
            x_te = X_te1';
            y_te = S_te_pro';
            if t==1               
                %[F --> S], projecting data from feature space to semantic sapce 
                S_est        =NormalizeFea(X_te) * W'; 
                [zsl_accuracy_1] = zsl_el(S_est, S_te_pro, param);
                fprintf('\n[%d] training [%d]CUB ZSL accuracy [V >>> S][0]: %.1f%%\n',total,train_iter,zsl_accuracy_1*100);
                %[S --> F], projecting from semantic to visual space                
                X_te_pro     = NormalizeFea( S_te_pro,0)* W;
                [zsl_accuracy_2]= zsl_el(X_te, X_te_pro, param); 
                fprintf('[%d] training [%d]CUB ZSL accuracy [S >>> V][0]: %.1f%%\n', total,train_iter,zsl_accuracy_2*100);
            end
            for i=1:test_num
                decoder = sum((ones(k_class, 1)*NormalizeFea(X_te(i,:)) - NormalizeFea( S_te_pro * W)).^2,2)';
                encoder = lambda*sum((NormalizeFea(W * X_te(i,:)')*ones(1, k_class) - NormalizeFea( S_te_pro)').^2,1);
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
                b_last = b_last + b;
                c_last = c_last + c;
            end
            a_last = a_last + alpha * (S_tr' * S_tr);
            b_last = b_last + alpha * lambda * (X_tr' * X_tr);
            c_last = c_last + alpha * (S_tr' * X_tr);
%             a_last = alpha*a_last +(S_tr' * S_tr);
%             b_last = lambda*alpha*b_last + lambda * (X_tr' * X_tr);
%             c_last = (alpha+alpha*lambda)*c_last + (alpha+1) * (S_tr' * X_tr);
            W = sylvester(a_last,b_last, c_last);
            % SAE
            %[F --> S], projecting data from feature space to semantic sapce 
            S_est        =NormalizeFea(X_te) * W'; 
            [zsl_accuracy_1, Y_hit5] = zsl_el(S_est, S_te_gt, param); 
            fprintf('[%d] training [%d]CUB ZSL accuracy [V >>> S][%d]: %.1f%%\n',total,train_iter,t,zsl_accuracy_1*100);
            %[S --> F], projecting from semantic to visual space
            X_te_pro     = NormalizeFea( S_te_pro,0)* W;
            [zsl_accuracy_2]= zsl_el(X_te, X_te_pro, param); 
            fprintf('[%d] training [%d]CUB ZSL accuracy [S >>> V][%d]: %.1f%%\n', total,train_iter,t,zsl_accuracy_2*100);
            t=t+1;
        end
        train_acc_1(total)=train_acc_1(total)+zsl_accuracy_1*100;
        train_acc_2(total)=train_acc_2(total)+zsl_accuracy_2*100;
        train_iter = train_iter+1;
    end
    train_acc_1(total)=train_acc_1(total)/(train_iter-1);
    train_acc_2(total)=train_acc_2(total)/(train_iter-1);
    fprintf('Average Accuracy of the %d time: %.2f%%\n',total,train_acc_1(total));
    fprintf('Average Accuracy of the %d time: %.2f%%\n',total,train_acc_2(total));
    total=total+1;
end