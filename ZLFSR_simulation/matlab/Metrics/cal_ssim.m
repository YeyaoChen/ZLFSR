function q = cal_ssim(I,J)
K = [0.01,0.03];
L = 1;
C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window =  fspecial('gaussian', 11, 1.5);	
window = window/sum(window(:));

I1 = double(I);
I2 = double(J);

mu1   = filter2(window, I1(:,:), 'same');
mu2   = filter2(window, I2(:,:), 'same');

mu1_sq = mu1.*mu1;    % E(x)^2
mu2_sq = mu2.*mu2;    % E(y)^2
mu1_mu2 = mu1.*mu2;   % E(x)*E(y)
sigma1_sq = filter2(window, I1(:,:).*I1(:,:), 'same') - mu1_sq;       % D(x)=E(x.^2)- E(x)^2
sigma2_sq = filter2(window, I2(:,:).*I2(:,:), 'same') - mu2_sq;       % D(y)=E(y.^2)- E(y)^2
sigma12 = filter2(window, I1(:,:).*I2(:,:), 'same') - mu1_mu2;        % cov(x,y)=E(x*y)- E(x)*E(y)
cal_SSIM_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2)); 

q= mean(cal_SSIM_map(:));
return
