function score = edgeMSE(imgRef,imgDist)

dx = [1 0 -1; 1 0 -1; 1 0 -1]/3;
dy = dx';
IxY1 = conv2(imgRef, dx, 'same');
IyY1 = conv2(imgRef, dy, 'same');
gradientMap = sqrt(IxY1.^2 + IyY1.^2);

sigma = 3;
edgeMap = conv2(double(gradientMap), fspecial('gaussian',sigma*6,sigma),'same');
edgeMap = imresize(edgeMap,size(imgRef));

score = sum(sum((imgRef-imgDist).^2.*edgeMap))/sum(sum(edgeMap));

end