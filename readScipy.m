clear all
data1 = load('C:\ML\env\tf\MNIST\test\scipyWriteMatread_v1.mat');
data2 = load('C:\ML\env\tf\MNIST\test\scipyWriteMatread_v2.mat');
data3 = load('C:\ML\env\tf\MNIST\test\scipyWriteMatread_v3.mat');
x1 = data1.x1;
x2 = data1.x2;
x3 = data1.x3;
x5 = data1.x5;

x1_v1 = x5(:,:,1);
x2_v1 = x5(:,:,2);
x3_v1 = x5(:,:,3);
x1_v2 = data2.arr(:,:,1);
x2_v2 = data2.arr(:,:,2);
x3_v2 = data2.arr(:,:,3);
x1_v3 = data3.arr(:,:,1);
x2_v3 = data3.arr(:,:,2);
x3_v3 = data3.arr(:,:,3);
