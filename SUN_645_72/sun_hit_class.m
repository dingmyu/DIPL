hit_class = {};
for i = 1:size(Y_hit5,1)
    idx5 = [];
    for j=1:5
        all_class = find(clusterFlag==unique_abel(Y_hit5(i,j)));
        idx5=[idx5 ; all_class(ismember(all_class,test_class))];
    end
    hit_class{i} = idx5;
end

filename=strcat('hit_class.mat');
save(filename,'hit_class');

filename=strcat('test_class.mat');
save(filename,'test_class')

filename=strcat('unique_abel.mat');
save(filename,'unique_abel');

filename=strcat('test_imgs.mat');
save(filename,'test_imgs')

filename=strcat('train_imgs.mat');
save(filename,'train_imgs')

filename=strcat('centSet.mat');
save(filename,'centSet')

filename=strcat('clusterFlag.mat');
save(filename,'clusterFlag')

filename=strcat('tmp_test_label.mat');
save(filename,'tmp_test_label')

% for i = 1:size(Y_hit5,1) 
%     hit_class{i}
% end