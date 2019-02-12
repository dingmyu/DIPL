%% %%% Imagenet-2 DEMO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Following code shows a demo for Imagenet-2 dataset to reproduce the result of the paper:
% 
% Semantic Autoencoder for Zero-shot Learning. 
% 
% Elyor Kodirov, Tao Xiang, and Shaogang Gong, CVPR 2017.
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We used GoogleNet features.
% X_tr: train data Ntr by d matrix. Ntr is the number of train samples, d
% is the feature dimension size.
% X_te: test  data Nte by d matrix. Nte is the number of test samples. 
% Y: Ntr by c label matrix (consists of one-hot vector). c is the number of classes. 
% S_tr is Ntr by k matrix. k is the semantic dimension size. 

clc
clear all
warning off

% Loading the data
addpath('library')
load('data_zsl/ImNet_2_demo_data.mat')

%% Dimension reduction
W    = (X_tr'  * X_tr + 150*eye(size(X_tr'*X_tr)))^(-1)*X_tr'*(Y)  ;
X_tr = X_tr * W;
X_te = X_te * W;

%% Learn projection

%X_te = X_te(1:3000,:);
[train_num, pic_fea_num] = size(X_tr);
[test_num, pic_fea_num] = size(X_te);
[train_num, sem_fea_num] = size(S_tr);
[classes_num, sem_fea_num] = size(S_te_pro);

%%%%% Training 
%X_tr    = NormalizeFea(X_tr')';
%X_te1    = NormalizeFea(X_te')';
beta = 0.02; alpha =  0.85;
if alpha==0 || alpha==1
    W=pinv(S_tr)*X_tr;
else
    W = SAE(X_tr', S_tr', alpha);
end
t=1; max_iter = 10; delimit=0.04;
nn=zeros(test_num,classes_num);
while(t<=max_iter)
    A=zeros(sem_fea_num,sem_fea_num);
    B=zeros(pic_fea_num,pic_fea_num);
    C=zeros(sem_fea_num,pic_fea_num);
    l=zeros(test_num,classes_num);
    %a={};b={};c={};
    if (t==1)
        %%%%% Testing, nearest neighbour classification %%%%%
        %[F --> S], projecting data from feature space to semantic space 
        X_te2 = X_te;
        S_te_est = X_te2 * NormalizeFea(W)';
        dist     =  1 -  pdist2(zscore(S_te_est),zscore(S_te_pro')', 'cosine');  % 26.3%
        HITK     = 5;
        Y_hit5   = zeros(size(dist,1),HITK);
        for i  = 1:size(dist,1)
            [~, I] = sort(dist(i,:),'descend');
            Y_hit5(i,:) = X_te_cl_id(I(1:HITK));
        end

        n=0;
        for i  = 1:size(dist,1)
            if ismember(Y_te(i),Y_hit5(i,:))
                n = n + 1;
            end
        end
        zsl_accuracy = n/size(dist,1);
        fprintf('\n[1] ImageNet-2 ZSL accuracy [V >>> S]: %.1f%%\n', zsl_accuracy*100);


        %[S --> F], projecting from semantic to visual space

        dist  =  1 - (pdist2(X_te2,zscore(NormalizeFea(W)' * S_te_pro')', 'cosine')) ;    % 27.1%
        HITK   = 5;
        Y_hit5 = zeros(size(dist,1),HITK);
        for i  = 1:size(dist,1)
            [sort_dist_i, I] = sort(dist(i,:),'descend');
            Y_hit5(i,:) = X_te_cl_id(I(1:HITK));
        end
        n = 0;
        for i  = 1:size(dist,1)
            if ismember(Y_te(i),Y_hit5(i,:))
                n = n + 1;
            end
        end
        zsl_accuracy = n/size(dist,1);
        fprintf('[2] ImageNet-2 ZSL accuracy [S >>> V]: %.1f%%\n', zsl_accuracy*100);
    end
    for i = 1:test_num
        if mod(i,5000)==0
            i
        end
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
                nn(i,j)=1/min_fre;
            else
                nn(i,j)=0;
            end
        end
        a=zeros(sem_fea_num,sem_fea_num);
        b=zeros(pic_fea_num,pic_fea_num);
        c=zeros(sem_fea_num,pic_fea_num);    
        for j=1:classes_num				
            if(nn(i,j))
                c=c+nn(i,j)*S_te_pro(j,:)'*X_te(i,:);
                a=a+nn(i,j)*S_te_pro(j,:)'*S_te_pro(j,:);
                b=b+nn(i,j)*X_te(i,:)'*X_te(i,:);
            end
        end
        A=A+(1-alpha)*a;
        B=B+alpha*b;
        C=C+c;
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
    
    %%%%% Testing, nearest neighbour classification %%%%%
    %[F --> S], projecting data from feature space to semantic space 
    X_te2 = X_te;
    S_te_est = X_te2 * W';
    dist     =  1 -  pdist2(zscore(S_te_est),zscore(S_te_pro')', 'cosine');  % 26.3%
    HITK     = 5;
    Y_hit5   = zeros(size(dist,1),HITK);
    for i  = 1:size(dist,1)
        [~, I] = sort(dist(i,:),'descend');
        Y_hit5(i,:) = X_te_cl_id(I(1:HITK));
    end

    n=0;
    for i  = 1:size(dist,1)
        if ismember(Y_te(i),Y_hit5(i,:))
            n = n + 1;
        end
    end
    zsl_accuracy = n/size(dist,1);
    fprintf('\n[1] ImageNet-2 ZSL accuracy [V >>> S]: %.1f%%\n', zsl_accuracy*100);


    %[S --> F], projecting from semantic to visual space

    dist  =  1 - (pdist2(X_te2,zscore(W' * S_te_pro')', 'cosine')) ;    % 27.1%
    HITK   = 5;
    Y_hit5 = zeros(size(dist,1),HITK);
    for i  = 1:size(dist,1)
        [sort_dist_i, I] = sort(dist(i,:),'descend');
        Y_hit5(i,:) = X_te_cl_id(I(1:HITK));
    end
    n = 0;
    for i  = 1:size(dist,1)
        if ismember(Y_te(i),Y_hit5(i,:))
            n = n + 1;
        end
    end
    zsl_accuracy = n/size(dist,1);
    fprintf('[2] ImageNet-2 ZSL accuracy [S >>> V]: %.1f%%\n', zsl_accuracy*100);
    
    t = t+1;
end

