function I = rgb2gray(X)
X = double(X);
I = 0.2989*X(:,:,1) + 0.5870*X(:,:,2) + 0.1140*X(:,:,3);