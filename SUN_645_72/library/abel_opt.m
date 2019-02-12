clc, clear all,  close all
%% data prepare
addpath('library');
load ('img_name_label.mat');
load ('cu_class_attr.mat');

vec=class_attr;
load ('rand_class.mat');
load(['F.mat' ]);
[cof,sco]=princomp(F);
F_=sco(:,1:100);

test_class=rand_class(1:50);
train_class=1:200;train_class(test_class)=[];
test_imgs=[];
for ite=1:50
    tmp=find(label+1==test_class(ite));
    test_imgs=[test_imgs;tmp];
end
train_imgs=1:length(label);
train_imgs(test_imgs)=[];
X_tr=F_(train_imgs,:);
X_te=F_(test_imgs,:);
load('img_attr.mat');
load('idx_old2new');
S_tr=img_attr(idx(train_imgs),:);

S_te_gt=vec(test_class,:);
S_te_pro=vec(test_class,:);
param.trainclasses_id=train_class;
param.testclasses_id=test_class;
param.test_labels=label(test_imgs)+1;
param.train_labels=label(train_imgs)+1;
%% main code
param.HITK           = 1;
param.testclasses_id = param.testclasses_id;
param.test_labels    = param.test_labels;
%%------------------------------------------------------Abel's code---------------------------------------------------------------------
w=load('abel_w.mat');%%w.W是傲雪师姐0.62acc的矩阵W
w=w.W;
w=w';	%%size(w)=100*312
max_length_min=10;
t=1;delta=0.1;delimit=0.0001;	%%	initialize the value of t and delta
while(max_length_min>delta)
	length_min=[];
	x_last=zeros(100,312);
	y_last=zeros(312,312);
	l=zeros(length(test_imgs),50);
	x={};s={};
	for i=1:length(test_imgs)
		x_te(:,i)=X_te(i,:)';
		for j=1:50
			y_te(:,j)=S_te_pro(j,:)';
			col_vec=x_te(:,i)-w*y_te(:,j);
			l(i,j)=dot(col_vec,col_vec);	%%计算每一张图片x_te(:,i)在50种类别y_te(:,j)当中的可能性,l(i,j)越小，可能性越大
		end
		length_min(i)=min(l(i,:));
		min_l=min(l(i,:));min_fre=0;
		for j=1:50
			if(abs(min_l-l(i,j))<delimit)
				min_fre=min_fre+1;		%%可能性最大x(;,i)出现（也就是min l出现的）的次数
			end
		end
		if t==1	%%----------------------------------------------
			for j=1:50
				if(abs(min_l-l(i,j))<delimit)
					n(i,j,t)=1/min_fre;
				else
					n(i,j,t)=0;
				end
			end
		else
			x{i}=zeros(100,312);
			s{i}=zeros(312,312);
			for j=1:50
				if(abs(min_l-l(i,j))<delimit)
					n(i,j,t)=1/min_fre;
				else
					n(i,j,t)=0;
				end
				if(n(i,j,t-1))
					x{i}=x{i}+n(i,j,t-1)*x_te(:,i)*y_te(:,j)';
					s{i}=s{i}+n(i,j,t-1)*y_te(:,j)*y_te(:,j)';%%将对应一系列矩阵相加
				end
			end
			x_last=x_last+x{i};y_last=y_last+s{i};
		end	%%----------------------------------------------
		
	end
        if(t>=2)
		%%x_last=NormalizeFea(x_last')';
		%%fprintf('function NormalizeFea has been executed!\n');
		w=x_last*pinv(y_last);
		w=NormalizeFea(w')';
	end
	max_length_min=max(length_min);
	w=w';
	X_te_pro     = NormalizeFea( S_te_pro) * NormalizeFea(w);
	[zsl_accuracy]= zsl_el(X_te, X_te_pro, param); 
	fprintf('ZSL accuracy [S >>> V]: %.1f%%\n', zsl_accuracy*100);
	fprintf('the value of max_length_min is: %.3f\n',max_length_min);
	filename=strcat('abel_w',num2str(t),'.mat');
	save(filename,'w','l');
	if(t>=2)
		w=10^(-5)*w';
	else
		w=w';
	end
	t=t+1;
end
%%------------------------------------------------------Abel's code--------------------------------------
