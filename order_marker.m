function [ marker ] = order_marker( Para1 )
%ORDER_MARKER 此处显示有关此函数的摘要
% find size of para1  
marker = [];
[m,n] = size(Para1);
% find the marker one;
Mpos = [Para1(:,2),1024-Para1(:,1)];
A = Mpos(:,1)+Mpos(:,2);
[P0,index] = min(A);
B(1) = Para1(index);
% calculate the distance between marker n and marker one;
Dis = [];
for i = 1:1:m
    if i ~= index
        Dis(i) = (Para1(i,1)-Para1(index,1))^2+(Para1(i,2)-Para1(index,2))^2;
    else
        Dis(i) = 0;
    end
end
[sX,index] = sort(Dis);
for ii = 1:1:m
    marker(ii,:) = Para1(index(ii),:);
end
end

