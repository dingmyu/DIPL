clc,clear all,close all
K = 50;
count=1;
load('cu_class_attr.mat');
dataSet = class_attr;
[row,col] = size(dataSet);
% 存储质心矩阵
centSet = zeros(K,col);
% 随机初始化质心
for i= 1:col
    minV = min(dataSet(:,i));
    rangV = max(dataSet(:,i)) - minV;
    centSet(:,i) = repmat(minV,[K,1]) + rangV*rand(K,1);
end
% 用于存储每个点被分配的cluster以及到质心的距离
clusterFlag = zeros(row,1);
clusterChange = true;
while clusterChange || RightCount
    clusterChange = false;
    RightCount=row;
    % 计算每个点应该被分配的cluster
    for i = 1:row
        % 这部分可能可以优化
        minDist = inf;
        minIndex = 0;
        for j = 1:K
            distCal = distEclud(dataSet(i,:) , centSet(j,:));
            if (distCal < minDist)
                minDist = distCal;
                minIndex = j;
            end
        end
        if minIndex ~= clusterFlag(i)            
            clusterChange = true;
            clusterFlag(i) = minIndex;
        else
            RightCount=RightCount-1;
        end
    end
    % 更新每个cluster 的质心
    for j = 1:K
        simpleCluster = find(clusterFlag(:,1) == j);
        centSet(j,:) = mean(dataSet(simpleCluster',:));
    end
    fprintf('the %dth training and the remainder %d\n',count,RightCount);
    count=count+1;
end