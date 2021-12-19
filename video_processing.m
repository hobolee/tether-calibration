clear; close all; clc

M = readmatrix('VIdeos\03.csv');
% M = M(100:end, :);
M(:, 2) = smoothdata(M(:, 2), 'gaussian', 50);
M(:, 2) = M(:, 2) - mean(M(1:2000, 2));
% plot(M(:, 1), M(:, 2));

vidObj_l = VideoReader('Videos\L_03.mp4');
vidObj_r = VideoReader('Videos\R_03.mp4');

% i =0;
% vidObj_l.CurrentTime = 0
% while hasFrame(vidObj_l)
%    frame = readFrame(vidObj_l);
%    vidObj_l.CurrentTime
%    i = i + 1
%    imshow(frame)
% %    pause(10/vidObj_l.FrameRate);
% end

% 1 5.08143 - 1.6 [79 670] [68 659]
% 2 7.11617          [118 649] [112 643]
% 3  6.46125        [86 623 ] [78 615]
images_l = read(vidObj_l, [86 623]);
images_r = read(vidObj_r, [78 615]);

load('Calibration2.mat');%加载你保存的相机标定的mat
leftIntrisic = stereoParams.CameraParameters1.IntrinsicMatrix';
leftRotation = [1,0,0;0,1,0;0,0,1 ];
leftTranslation = [0,0,0];
rightIntrisic = stereoParams.CameraParameters2.IntrinsicMatrix';
rightRotation = stereoParams.RotationOfCamera2';
rightTranslation = stereoParams.TranslationOfCamera2;

length = size(images_l);
data = [];
e = 0;
for i = 1:length(4)
    I1 = images_l(:, :, :, i);
    I2 = images_r(:, :, :, i);
    %畸变
    [J1,new1] = undistortImage(I1,stereoParams.CameraParameters1);
    [J2,new2] = undistortImage(I2,stereoParams.CameraParameters2);
    J1(512:end, :, :) = 0;
    J1(:, 1:540, :) = 0;
    J2(512:end,:, :) = 0;
    J2(:,741:end, :) = 0;
%     figure(); 
%     imshow(J1,[]),title('原图（左）')
%     figure(); 
%     imshow(J2,[]),title('原图（右）')

    J1 = im2double(J1);
    J2 = im2double(J2);
    J1 = rgb2hsv(J1);
    J2 = rgb2hsv(J2);
%     dd1=((J1(:,:,1)<=10/255&J1(:,:,1)>=0&J1(:,:,2)<=1&J1(:,:,2)>=43/255&J1(:,:,3)<=1&J1(:,:,3)>=46/255)...
%         |(J1(:,:,1)<=180/255&J1(:,:,1)>=156/255&J1(:,:,2)<=1&J1(:,:,2)>=43/255&J1(:,:,3)<=1&J1(:,:,3)>=46/255));
%     dd2=((J2(:,:,1)<=10/255&J2(:,:,1)>=0&J2(:,:,2)<=1&J2(:,:,2)>=43/255&J2(:,:,3)<=1&J2(:,:,3)>=46/255)...
%         |(J2(:,:,1)<=180/255&J2(:,:,1)>=156/255&J2(:,:,2)<=1&J2(:,:,2)>=43/255&J2(:,:,3)<=1&J2(:,:,3)>=46/255));
    dd1=((J1(:,:,1)<=10/255&J1(:,:,1)>=0&J1(:,:,2)<=1&J1(:,:,2)>=123/255&J1(:,:,3)<=1&J1(:,:,3)>=100/255)...
              |(J1(:,:,1)<=180/255&J1(:,:,1)>=156&J1(:,:,2)<=1&J1(:,:,2)>=43/255&J1(:,:,3)<=1&J1(:,:,3)>=46/255));
    dd2=((J2(:,:,1)<=10/255&J2(:,:,1)>=0&J2(:,:,2)<=1&J2(:,:,2)>=123/255&J2(:,:,3)<=1&J2(:,:,3)>=100/255)...
              |(J2(:,:,1)<=180/255&J2(:,:,1)>=156&J2(:,:,2)<=1&J2(:,:,2)>=43/255&J2(:,:,3)<=1&J2(:,:,3)>=46/255));
    
%     dd1 = hsv2rgb(dd1);
%     dd2 = hsv2rgb(dd2);
    
    %根据像素值找点 
%     dd1=((J1(:,:,1)<=250&J1(:,:,1)>=150&J1(:,:,2)<=90&J1(:,:,2)>=30&J1(:,:,3)<=80&J1(:,:,3)>=10)...
%         |(J1(:,:,1)<=120&J1(:,:,1)>=80&J1(:,:,2)<=70&J1(:,:,2)>=30&J1(:,:,3)<=40&J1(:,:,3)>=20))...
%         |(J1(:,:,1)<=240&J1(:,:,1)>=220&J1(:,:,2)<=120&J1(:,:,2)>=60&J1(:,:,3)<=80&J1(:,:,3)>=40)...
%         |(J1(:,:,1)<=160&J1(:,:,1)>=120&J1(:,:,2)<=80&J1(:,:,2)>=20&J1(:,:,3)<=70&J1(:,:,3)>=00);
%     dd2=((J2(:,:,1)<=250&J2(:,:,1)>=150&J2(:,:,2)<=90&J2(:,:,2)>=30&J2(:,:,3)<=80&J2(:,:,3)>=10)...
%         |(J2(:,:,1)<=120&J2(:,:,1)>=80&J2(:,:,2)<=70&J2(:,:,2)>=30&J2(:,:,3)<=40&J2(:,:,3)>=20))...
%         |(J2(:,:,1)<=250&J2(:,:,1)>=210&J2(:,:,2)<=110&J2(:,:,2)>=50&J2(:,:,3)<=70&J2(:,:,3)>=30)...
%         |(J2(:,:,1)<=160&J2(:,:,1)>=120&J2(:,:,2)<=80&J2(:,:,2)>=20&J2(:,:,3)<=70&J2(:,:,3)>=00);
%     dd1=((J1(:,:,1)<=255&J1(:,:,1)>=210&J1(:,:,2)<=100&J1(:,:,2)>=50&J1(:,:,3)<=60&J1(:,:,3)>=20)...
%           |(J1(:,:,1)<=220&J1(:,:,1)>=160&J1(:,:,2)<=120&J1(:,:,2)>=60&J1(:,:,3)<=100&J1(:,:,3)>=40)...
%           |(J1(:,:,1)<=240&J1(:,:,1)>=190&J1(:,:,2)<=130&J1(:,:,2)>=60&J1(:,:,3)<=100&J1(:,:,3)>=30)...
%           |(J1(:,:,1)<=255&J1(:,:,1)>=220&J1(:,:,2)<=110&J1(:,:,2)>=90&J1(:,:,3)<=80&J1(:,:,3)>=30));
%     dd2=((J2(:,:,1)<=255&J2(:,:,1)>=150&J2(:,:,2)<=120&J2(:,:,2)>=50&J2(:,:,3)<=100&J2(:,:,3)>=30)...
%           |J1(:,:,1)<=150&J1(:,:,1)>=100&J1(:,:,2)<=80&J1(:,:,2)>=30&J1(:,:,3)<=80&J1(:,:,3)>=30);
    [m1,n1]=size(dd1);
    z1=zeros(m1,n1);
    J11 = (cat(3,dd1,z1,z1));
    [m2,n2]=size(dd2);
    z2=zeros(m2,n2);
    J22 = (cat(3,dd2,z2,z2));
    
    figure(1); 
    imshow(J11,[]),title('边缘（左）')
    figure(2); 
    imshow(J22,[]),title('边缘（右）')
    SE=strel('rectangle',[3 3]);
    %图形学操作
    K1=imerode(J11,SE);%腐蚀
    K2=imerode(J22,SE);%腐蚀
    L1=imdilate(K1,SE);%膨胀
    L2=imdilate(K2,SE);%膨胀
    M1=im2bw(L1,0.1); 
    M2=im2bw(L2,0.1); 
    BW1=edge(M1,'sobel');
    BW2=edge(M2,'sobel');
    Step_r = 0.2;  
    
    %角度步长0.1，单位为弧度  
    Step_angle = 0.1;  
    %最小圆半径2  
    minr =8;  
    %最大圆半径30  
    maxr = 12;  
    %以thresh*hough_space的最大值为阈值，thresh取0-1之间的数  
    thresh1 = 0.4;  
    thresh2 = 0.4;
    %-----------这个只负责提供其中一张图片的圆心坐标，这个函数是上一段代码 
    [Hough_space1,Hough_circle_result1,Para1] = Hough_circle(BW1,Step_r,Step_angle,minr,maxr,thresh1);
    %开始检测另一个圆
    [Hough_space2,Hough_circle_result2,Para2] = Hough_circle(BW2,Step_r,Step_angle,minr,maxr,thresh2);  
    % 两幅图像素差
    circleParaXYR1=order_marker(Para1);
    circleParaXYR2=order_marker(Para2);
    number1 = size(circleParaXYR1);
    number2 = size(circleParaXYR2);
    if number1(1)  < 5 | number2(1) < 5
        warning("warning: %d", i);
        e = e+1
        number1
        number2
        continue
    end
    cirtira = 20;
    circleParaXYR1 = cluster_points(circleParaXYR1, cirtira);
    circleParaXYR2 = cluster_points(circleParaXYR2, cirtira);
    figure(1)
    imshow(BW1,[]),title('边缘（左）')
    hold on;
    plot(circleParaXYR1(:,2), circleParaXYR1(:,1), 'r+');  
    % figure;
    figure(2)
    imshow(BW2,[]),title('边缘（右）')
    hold on;
    plot(circleParaXYR2(:,2), circleParaXYR2(:,1), 'r+');  
    
    x = [];
    d = [];
    for ii=1:5
        x(ii,:) = uv2xyz_1(circleParaXYR1(ii,:), circleParaXYR2(ii,:), leftIntrisic, leftTranslation, leftRotation, rightIntrisic, rightTranslation, rightRotation);
        d = [d x(ii, 1:2)];
    end
    
    t = 6.46125   + 0.2*(i - 1);
    f_size = size(M);
    f_start = 1;
    f_end = f_size(1) - 1;
    for j = f_start:f_end
        if (t - M(j, 1)) * (t - M(j+1, 1))  <= 0
            f = (M(j, 2) + M(j+1, 2)) / 2;
            f_start = j;
            break
        end
    end
    d = [d f];
    data = [data; d];
    close all
end
save('data.txt', 'data', '-ascii');





