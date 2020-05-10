clear all;
N= 28;
num_pix = 28*28;
%x = 0.1*rand(28);
x= imread('C:\ML\env\tf\MNIST\test\1.jpeg');
x= double(x);
mu_x = mean2(x);
std_x = std2(x);

x_norm = (x - mu_x)./std_x;


%y = 0.9 + 0.1*rand(28);
y = imread('C:\ML\env\tf\MNIST\test\4.jpeg');
y= double(y);
mu_y = mean2(y);
% compare with my own
mu1_y = mean(mean(y));

std_y =  std2(y);
% reshape to N2*1 dim
z = reshape(y,num_pix,1);
% std from all pixels
std1_y = std(z);
std2_y = sqrt((1/(num_pix-1))*sum(sum((y - mu_y).^2))); %slightly varies as Var in matlab can use 1/n or 1/n-1
std3_y = sqrt((1/(num_pix-1))* sum(sum((z-mu_y).^2)) );
y_norm = (y - mu_y)./std_y;

% statistics over  all images
w= x + y;
mu_w = mean2(w);

w1  = zeros(N,N,2);% as if 2 btches
w1(:,:,1) = x;
w1(:,:,2) = y;
% sd over all the images
w2 = reshape(w1,N*N*2,1);
std_w2 = std(w2);

x_norm_all = (x - mu_w )./std_w2;
y_norm_all = (y - mu_w )./std_w2;

%  *************** load numpy data ******

datagen_norm_sam = load('C:\ML\env\tf\MNIST\test\datagen_norm_sam.mat');
datagen_norm_ftr = load('C:\ML\env\tf\MNIST\test\datagen_norm_ftr.mat');
x_norm_sam_np = datagen_norm_sam.arr(:,:,1);
y_norm_sam_np = datagen_norm_sam.arr(:,:,2);

x_norm_ftr_np = datagen_norm_ftr.arr(:,:,1);
y_norm_ftr_np = datagen_norm_ftr.arr(:,:,2);




figure;imshow(x);figure;imshow(y);
%figure;imshow(x_norm);figure;imshow(y_norm);
% all images mean is higher than single image mean, so subtracting it would
%  make region around digit more darker
%figure;imshow(x_norm_all);figure;imshow(y_norm_all);
figure;imshow(x - x_norm);figure;imshow(y - y_norm);
figure;imshow(x - x_norm_all);figure;imshow(y - y_norm_all);
figure;imshow(x_norm - x_norm_all);figure;imshow(y_norm - y_norm_all);