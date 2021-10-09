function [F]= fbostonh(bostonh,num,idea)
%BOSTONH 此处显示有关此函数的摘要
%   此处显示详细说明
    r=zeros(506,numel(num));
    new=bostonh(:,num);
    nidea=idea(:,num);
    max_bost=max(new);
    min_bost=min(new);
    for i=1:506
        for j=1:numel(num)
            r(i,j)=abs(new(i,j)-nidea(1,j))/(max_bost(j)-min_bost(j));
        end
    end
    mean_bost=mean(r);
    std_bost=std(r,1);
    v=std_bost./abs(mean_bost);
    w=v./sum(v);
    F=zeros(506,1);
    for i=1:506
        F(i)=sum(r(i,:).*w);
    end
end

