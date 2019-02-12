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
%67.90%
% 66.9% 69.4%
% 67.2% 68.9%
% 64.1% 67.8%
% 64.6% 66.7%
% 66.7% 69.7%
% 64.2% 66.7%
% 69.2% 71.4%
% 60.7% 65.6%
% 67.0% 68.3%
% 64.4% 68.1%

% 65.5   68.26      
k_class=300;
abel_label=Y;
[clusterFlag,centSet]=kmeans(class_attr,k_class,'EmptyAction','drop');
    for i=1:k_class
        tmpidx=find(clusterFlag==i);
        for j=1:length(tmpidx)
            abel_tmpidx=find(Y==tmpidx(j));
            abel_label(abel_tmpidx,1)=i;
        end
    end
    
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

        abel_tmp_test_label = abel_label(test_imgs);
        
        tmp_test_label=abel_tmp_test_label;
        unique_abel = unique(tmp_test_label);
        k_class = numel(unique_abel);
        for i=1:k_class
            tmp_test_label(abel_tmp_test_label==unique_abel(i)) = i;
        end    
    
    X_tr=F_(train_imgs,:);
    X_te=F_(test_imgs,:);
    S_tr=img_attr(train_imgs,:);
    %S_te_pro=class_attr(test_class,:);
    %S_te_gt = class_attr(test_class,:);
        S_te_pro=centSet(unique_abel,:);
        S_te_gt = S_te_pro;
    %% main code
    param.HITK           = 12;
    param.testclasses_id = 1:k_class;
    param.test_labels    = tmp_test_label;
    %%------------------------------------------------------Abel's code---------------------------------------------------------------------
    X = [X_tr', X_te'];
    X = X';
    [cof,sco]=princomp(X);
    X_tr=sco(1:length(X_tr),1:score_num);
    X_te = sco(length(X_tr)+1:length(X),1:score_num);

        
[train_num, pic_fea_num] = size(X_tr);
[test_num, pic_fea_num] = size(X_te);
[train_num, sem_fea_num] = size(S_tr);
[classes_num, sem_fea_num] = size(S_te_pro);

X_tr    = NormalizeFea(X_tr')';
X_te1    = NormalizeFea(X_te')';
beta = 0.04; alpha =  0.5;
if alpha==0 || alpha==1
    W=pinv(S_tr)*X_tr;
else
    W = SAE(X_tr', S_tr', alpha);
end
t=1; max_iter = 1; delimit=0.1;
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
        W = NormalizeFea(W);
        sum(sum((W-W_old)*(W-W_old)'))/sum(sum(W*W'))
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
        train_acc(total)=train_acc(total)+zsl_accuracy*100;
        train_iter = train_iter+1;

        hit_class = {};
        for i = 1:size(Y_hit5,1)
            idx5 = [];
            for j=1:param.HITK
                all_class = find(clusterFlag==unique_abel(Y_hit5(i,j)));
                idx5=[idx5 ; all_class(ismember(all_class,test_class))];
            end
            hit_class{i} = idx5;
        end
        
        filename=strcat('hit_class', num2str(train_iter), '.mat');
        save(filename,'hit_class');

        filename=strcat('test_class', num2str(train_iter), '.mat');
        save(filename,'test_class') 
    end;
   
    train_acc(total)=train_acc(total)/(train_iter-1);
    avg_init_acc(total)=avg_init_acc(total)/(train_iter-1);
    fprintf('Average Accuracy of the %d time: %.2f%%\n',total,train_acc(total));
    fprintf('Average Initial Accuracy of the %d time: %.2f%%\n',total,avg_init_acc(total));
    total=total+1;
end;