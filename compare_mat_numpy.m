clear all;
% if tst1==1, we load image 1,4 in \test folder , and run python file
% dataAugmentationtest.py
% if tst1==2, we create matrices in matlab, write as image, run till debug
% point , run python, come back to matlab and resume from reading python
% variables
tst1= 2;  % or 2



%% (A)RUN THIS OR (B): if loading images from MNIST
if(tst1==1)
    x= imread('C:\ML\env\tf\MNIST\test\1.jpeg');
    y = imread('C:\ML\env\tf\MNIST\test\4.jpeg');
end

%% creating own 2x2 data
%If you use imwrite() to write uint8 or uint16 images, you will get back the same result
%when you read them unless you used an image format that uses lossy compression.
%JPG image format uses lossy compression and so should be avoided for this purpose.
%PNG and TIFF do not use lossy compression and so should be able to give you back exactly what you wrote.
% x=  double([0.1 0.2;0.9 0.4]);
% y = uint8([ 5 6; 7 8]);
% imwrite(x,'C:\ML\env\tf\MNIST\test\m1.jpeg');
% imwrite(y,'C:\ML\env\tf\MNIST\test\m4.jpeg');
% x1= imread('C:\ML\env\tf\MNIST\test\m1.jpeg','jpeg');
% y1 = imread('C:\ML\env\tf\MNIST\test\m4.jpeg');
%WE CAN SEE X,X1 AND Y,Y1 DIFER AS WTITE FORMAT WAS LOSSY

%this also dont work
% imwrite(mat2gray(y),'C:\ML\env\tf\MNIST\test\m4.jpeg','JPEG');
% y1 = uint8(imread('C:\ML\env\tf\MNIST\test\m4.jpeg'));

% ********This matches********** %
% y = uint8([ 5 6; 7 8]);
% imwrite(y,'C:\ML\env\tf\MNIST\test\m1.png');
% y1 = imread('C:\ML\env\tf\MNIST\test\m1.png');
%% (B) RUN THIS OR (A)
if (tst1==2)
    xa=  uint8([1 2; 3 4]);
    ya = uint8([ 5 6; 7 8]);
    imwrite(xa,'C:\ML\env\tf\MNIST\test\m1.png');
    imwrite(ya,'C:\ML\env\tf\MNIST\test\m4.png');
    x= imread('C:\ML\env\tf\MNIST\test\m1.png','png');
    y = imread('C:\ML\env\tf\MNIST\test\m4.png');
end
%% read data

N = size(x,1);
num_pix = N*N;
x= double(x);
mu_x = mean2(x);
% std_x = std2(x);
std_x = std(reshape(x,N*N,1),1);
y= double(y);
mu_y = mean2(y);
%std_y =  std2(y);
std_y = std(reshape(y,N*N,1),1);

y= double(y);
% Test below with  ImageDataGenerator(samplewise_center = True,samplewise_std_normalization=True)
x_norm_sam_mat = (x - mu_x)/std_x;
y_norm_sam_mat = (y - mu_y)/std_y;
% Test below with  ImageDataGenerator(samplewise_center = True
% x_norm_sam_mat = (x - mu_x);
% y_norm_sam_mat = (y - mu_y);

% statistics over  all images

mu_w = (mu_x+mu_y)/2;

w1  = zeros(N,N,2);% as if 2 btches
w1(:,:,1) = x;
w1(:,:,2) = y;
% sd over all the images
w = reshape(w1,N*N*2,1);
%this specifies 1/Number of obesrevations and matches more closely with
%python, if std(w) used , it is 1/No of obser-1
std_w = std(w,1);

% Test below with  ImageDataGenerator(featurewise_center= True,featurewise_std_normalization=True)
x_norm_ftr_mat = (x - mu_w )/std_w;
y_norm_ftr_mat = (y - mu_w )/std_w;
% Test below with  ImageDataGenerator(featurewise_center= True)
% x_norm_ftr_mat = (x - mu_w);
% y_norm_ftr_mat = (y - mu_w);

%  *************** load numpy data from C:\ML\env\tf\dataAugmentationTest.py******
if tst1==2
    dbstop in compare_mat_numpy at 82
end
datagen_norm_rescale = load('C:\ML\env\tf\MNIST\test\datagen_rescale.mat');
datagen_norm_sam = load('C:\ML\env\tf\MNIST\test\datagen_norm_sam_pyt_svd.mat');
datagen_norm_ftr = load('C:\ML\env\tf\MNIST\test\datagen_norm_ftr_pyt_svd.mat');
if(tst1==2)
x_rsc_np = datagen_norm_rescale.arr(:,:,1);
y_rsc_np = datagen_norm_rescale.arr(:,:,2);
x_norm_sam_np = datagen_norm_sam.arr(:,:,1);
y_norm_sam_np = datagen_norm_sam.arr(:,:,2);
x_norm_ftr_np = datagen_norm_ftr.arr(:,:,1);
y_norm_ftr_np = datagen_norm_ftr.arr(:,:,2);
else
x_rsc_np = datagen_norm_rescale.x;
y_rsc_np = datagen_norm_rescale.y;
x_norm_sam_np = datagen_norm_sam.x;
y_norm_sam_np = datagen_norm_sam.y;
x_norm_ftr_np = datagen_norm_ftr.x;
y_norm_ftr_np = datagen_norm_ftr.y;
end

%figure;imshow(x);figure;imshow(y);
norm(x - x_rsc_np)
norm(y - y_rsc_np)
norm(x_norm_sam_mat - x_norm_sam_np)
norm(y_norm_sam_mat - y_norm_sam_np)
norm(x_norm_ftr_mat - x_norm_ftr_np)
norm(y_norm_ftr_mat - y_norm_ftr_np)