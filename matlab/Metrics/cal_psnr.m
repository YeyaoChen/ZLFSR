function q = cal_psnr(I,J)
I1 = double(I);
I2 = double(J);
[m,n,col] = size(I1);
diff = (I1-I2).^2;
mse = sum(diff(:))/(m*n*col);
q = 10*log10(1^2/mse);
return


