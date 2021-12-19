function [Hough_space,Hough_circle_result,Para] = Hough_circle(BW,Step_r,Step_angle,r_min,r_max,p)  
%---------------------------------------------------------------------------------------------------------------------------  
% input��  
% BW:��ֵͼ��;  
% Step_r:����Բ�뾶����;  
% Step_angle:�ǶȲ�������λΪ����;  
% r_min:��СԲ�뾶;  
% r_max:���Բ�뾶;  
% p:��p*Hough_space�����ֵΪ��ֵ��pȡ0��1֮�����.  
% a = x-r*cos(angle); b = y-r*sin(angle);  
%---------------------------------------------------------------------------------------------------------------------------  
% output��  
% Hough_space:�����ռ䣬h(a,b,r)��ʾԲ����(a,b)�뾶Ϊr��Բ�ϵĵ���;  
% Hough_circle:��ֵͼ�񣬼�⵽��Բ;  
% Para:��⵽��Բ��Բ�ġ��뾶.  
%---------------------------------------------------------------------------------------------------------------------------  
circleParaXYR=[];  
Para=[];  
%�õ���ֵͼ���С  
[m,n] = size(BW);  
%������뾶�ͽǶȵĲ�����ѭ������ ��ȡ������������  
size_r = round((r_max-r_min)/Step_r)+1;  
size_angle = round(2*pi/Step_angle);  
%���������ռ�  
Hough_space = zeros(m,n,size_r);  
%���ҷ���Ԫ�ص���������  
[rows,cols] = find(BW);  
%��������ĸ���  
ecount = size(rows);  
% Hough�任  
% ��ͼ��ռ�(x,y)��Ӧ�������ռ�(a,b,r)  
% a = x-r*cos(angle)  
% b = y-r*sin(angle)  
i = 1;
ecount = ecount(1);
for i=1:ecount
    for r=1:size_r %�뾶��������һ�����Ȱ�Բ���ȷ�  
        for k=1:size_angle  
            a = round(rows(i)-(r_min+(r-1)*Step_r)*cos(k*Step_angle));  
            b = round(cols(i)-(r_min+(r-1)*Step_r)*sin(k*Step_angle));  
            if (a>0&&a<=m&&b>0&&b<=n)  
                Hough_space(a,b,r)=Hough_space(a,b,r)+1;%h(a,b,r)�����꣬Բ�ĺͰ뾶  
            end  
        end  
    end  
end  
% ����������ֵ�ľۼ��㣬���ڶ��Բ�ļ�⣬��ֵҪ���Сһ�㣡ͨ������ֵ�������������Բ��Բ�ĺͰ뾶����ֵ���������������ֵ  
max_para = max(max(max(Hough_space)));  
%һ�������У����ҵ����д���max_para*p����λ��  
index = find(Hough_space>=max_para*p);  
length = size(index);%������ֵ�ĸ���  
Hough_circle_result=zeros(m,n);  
%ͨ��λ����뾶��Բ�ġ�  
length = length(1);
k = 1;
par = 1;
for i=1:ecount  
    for k=1:length  
        par3 = floor(index(k)/(m*n))+1;  
        par2 = floor((index(k)-(par3-1)*(m*n))/m)+1;  
        par1 = index(k)-(par3-1)*(m*n)-(par2-1)*m;  
        if((rows(i)-par1)^2+(cols(i)-par2)^2<(r_min+(par3-1)*Step_r)^2+5&&...  
          (rows(i)-par1)^2+(cols(i)-par2)^2>(r_min+(par3-1)*Step_r)^2-5)  
            Hough_circle_result(rows(i),cols(i)) = 1;%����Բ  
        end  
    end  
end  
% �ӳ�����ֵ��ֵ�еõ�    
for k=1:length    
    par3 = floor(index(k)/(m*n))+1;%ȡ��    
    par2 = floor((index(k)-(par3-1)*(m*n))/m)+1;    
    par1 = index(k)-(par3-1)*(m*n)-(par2-1)*m;    
    circleParaXYR = [circleParaXYR;par1,par2,par3];    
    Hough_circle_result(par1,par2)= 1; %��ʱ�õ��ö�Բ�ĺͰ뾶����ͬ��Բ��Բ�Ĵ��ۼ��ö�㣬������Ϊ������Բ���Ǳ�׼��Բ    
end   
%�����ڸ���Բ��Բ�Ĵ��ĵ�ȡƽ�����õ����ÿ��Բ�ľ�ȷԲ�ĺͰ뾶;  
while size(circleParaXYR,1) >= 1  
    num=1;  
    XYR=[];  
    temp1=circleParaXYR(1,1);  
    temp2=circleParaXYR(1,2);  
    temp3=circleParaXYR(1,3);  
    c1=temp1;  
    c2=temp2;  
    c3=temp3;  
    temp3= r_min+(temp3-1)*Step_r;  
    if size(circleParaXYR,1)>1  
        for k=2:size(circleParaXYR,1)  
            if (circleParaXYR(k,1)-temp1)^2+(circleParaXYR(k,2)-temp2)^2 > temp3^2  
                XYR=[XYR;circleParaXYR(k,1),circleParaXYR(k,2),circleParaXYR(k,3)];  %����ʣ��Բ��Բ�ĺͰ뾶λ��  
            else  
                c1=c1+circleParaXYR(k,1);  
                c2=c2+circleParaXYR(k,2);  
                c3=c3+circleParaXYR(k,3);  
                num=num+1;  
            end  
        end  
    end  
    c1=round(c1/num);  
    c2=round(c2/num);  
    c3=round(c3/num);  
    c3=r_min+(c3-1)*Step_r;  
    Para=[Para;c1,c2,c3]; %�������Բ��Բ�ĺͰ뾶��ֵ  
    circleParaXYR=XYR;  
end  
