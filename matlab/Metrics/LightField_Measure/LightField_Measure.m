function score =  LightField_Measure(View_RefName,View_DisName)
% Input:  (1) View_RefName: file names of the reference views
%         (2) View_DisName: file names of the distorted views
% Output: (1) score: quality score
% Usage:  Given the file names of the reference and distorted views
%         score =  LightField_Measure(View_RefName,View_DisName)
        
% for i = 1:length(View_RefName)
    
%     viewRef = imread(View_RefName{1,i});
%     viewDis = imread(View_DisName{1,i});
%     viewRef = rgb2gray(double(viewRef));
%     viewDis = rgb2gray(double(viewDis));

for i = 1:size(View_RefName,3)
    viewRef = View_RefName(:,:,i);
    viewDis = View_DisName(:,:,i);
    
    cornerScore(1,i) = cornerSIM(viewRef,viewDis);
    edgeScore(1,i) = edgeMSE(viewRef,viewDis);
end

angularScore = angularAnalysis(edgeScore);

score = log(mean(cornerScore./(edgeScore.^0.5+(0.01*255)^2)*angularScore));

end