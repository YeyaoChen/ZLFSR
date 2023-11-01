function score = cornerSIM(imgRef,imgDist)
% Input : (1) img: test image
% Output: (1) score: the blockiness score
% Usage:  Given a test image img, whose dynamic range is 0-255
%         score = PSS(img);

cornersRef = detectFASTFeatures(uint8(imgRef),'MinQuality',0.001,'MinContrast',0.001);
cornersRef = cornersRef.Location;

cornersDist = detectFASTFeatures(uint8(imgDist),'MinQuality',0.001,'MinContrast',0.001);
cornersDist = cornersDist.Location;

overlap = intersect(cornersRef,cornersDist,'rows');
score = length(overlap)/(length(cornersRef)+1);

end