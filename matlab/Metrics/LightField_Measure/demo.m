clc
clear

LightField_RefName = 'ArtGallery2';
LightField_DisName = 'ArtGallery2_DepthInterpolation_Skip24';

for i = 1:101
    View_RefName{1,i} = fullfile(LightField_RefName,['Frame_' num2str(i-1,'%03d') '.png']);
    View_DisName{1,i} = fullfile(LightField_DisName,['Frame_' num2str(i-1,'%03d') '.png']);
end

LightField_score = LightField_Measure(View_RefName,View_DisName);

