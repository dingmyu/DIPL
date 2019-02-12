function [ zsl_accuracy, Y_hit5 ] = abel_zsl_el(abel_param)
% ZSL_EL calculates zero-shot classification accuracy
%
% INPUT: 
%    S_est: estimated semantic labels
%    S_te_gt: ground truth semantic labels
%    param: other parameters
%
% Output:  
%    zsl_accuracy: zero-shot classification accuracy (per-sample)
total_line = size(abel_param.S_est,1);
Y_hit5   = zeros(total_line,abel_param.HITK);
for i=1:total_line
    abel_param.testclasses_id{i,1}=intersect(abel_param.testclasses_id{i,1},abel_param.test_class);
end
if abel_param.flag %[S>>>V]
    for i=1:total_line
        test_class_attr = abel_param.class_attr((abel_param.testclasses_id{i,1})',:);
        test_class_fea  = NormalizeFea(test_class_attr)*abel_param.W;
%         test_class_fea  = test_class_attr*abel_param.W;
        test_class_fea  = NormalizeFea(test_class_fea);
        dist = 1 - (pdist2(abel_param.S_est(i,:), test_class_fea, 'cosine'));
        [~,I]=sort(dist,'descend');
        idx=abel_param.testclasses_id{i,1};
        Y_hit5(i,:) = idx(I(1:abel_param.HITK));
    end 
else         %[V>>>S]
    for i=1:total_line
        test_class_attr = abel_param.class_attr((abel_param.testclasses_id{i,1})',:);
        test_class_attr = NormalizeFea(test_class_attr);
        test_sample_fea=NormalizeFea(abel_param.S_est(i,:))*abel_param.W';
%         test_sample_fea=abel_param.S_est(i,:)*abel_param.W';
        dist = 1 - (pdist2(test_sample_fea, test_class_attr, 'cosine'));
        [~,I]=sort(dist,'descend');
        idx=abel_param.testclasses_id{i,1};         
        Y_hit5(i,:) = idx(I(1:abel_param.HITK));
    end 
end
n = 0;
for i  = 1:total_line
    if ismember(abel_param.test_labels(i),Y_hit5(i,:))
        n = n + 1;
    end
end
zsl_accuracy = n/total_line;
end