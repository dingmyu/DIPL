function [h_mean]= gszl_evalution(X_te, X_te_pro, seen_class, unseen_class, test_labels, param)
%GSZL_EVALUATION
%
% INPUT: 
%    X_te: ground truth image features
%    X_te_pro: predict image features
%    param: other parameters
%
% Output:  
%    harmonic mean of seen and unseen accuracy
%    zero shot learning
test_classes=1:200;
n_seen = 0;
n_unseen = 0;
gamma=0.15;
for i = 1:length(test_labels)
    if ismember(test_labels(i), seen_class)
        n_seen = n_seen + 1;
    else
        n_unseen = n_unseen + 1;
    end
end
unseen_seen = zeros(1, 2);
right_seen = 0;
right_unseen = 0;
dist = 1 - (pdist2(X_te, NormalizeFea(X_te_pro,0), 'cosine'));
min_dist=min(dist,[],2);
max_dist=max(dist,[],2);
dist=(dist-min_dist*ones(1,size(dist,2)))./((max_dist-min_dist)*ones(1,size(dist,2)));
f=dist;
% f(1,1:20)
f(:,seen_class)=dist(:,seen_class)-gamma;
[~,I]=sort(f,2,'descend');
for i = 1:size(dist,1)
     if test_labels(i) == test_classes(I(i,1:param.HITK));
        flag=ismember(test_labels(i), seen_class);
        if flag
            right_seen=right_seen+1;
        else
            right_unseen = right_unseen + 1;
        end
    end
end
unseen_seen(2) = right_seen / n_seen;
unseen_seen(1) = right_unseen / n_unseen;
% unseen_seen = sortrows(unseen_seen, 1);
x = unseen_seen(1);
y = unseen_seen(2);
fprintf('unseen-seen: %.4f %.4f\n',x,y);
h_mean=2*x*y/(x+y);
end

