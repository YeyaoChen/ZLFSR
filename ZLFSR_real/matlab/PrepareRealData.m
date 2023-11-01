clc
clear all;

%% path
folder1 = 'I:/Dataset/Heterogeneous_imaging_data/LFImage/';
folder2 = 'I:/Dataset/Heterogeneous_imaging_data/2DImage/';
save_path = 'RealData/';
if ~exist(save_path,'dir')
    mkdir(save_path);
end

%% parameters
H = 400;
W = 600;
angRes = 7;

%% generate data
for ind = 1:1
    savename = [save_path,'data',num2str(ind),'.h5'];
    LFI1 = zeros(1, H, W, 3, angRes, angRes, 'uint8');
    LFI2 = zeros(1, H, W, 3, angRes, angRes, 'uint8');
    count = 0;

    img1 = zeros(H, W, 3, angRes, angRes, 'uint8');  
    img2 = zeros(H, W, 3, angRes, angRes, 'uint8');     
    for p = 1:7
        for q = 1:7
            sai1 = imread([folder1,num2str(ind),'/',num2str(p),'_',num2str(q),'.png']);
            sai1 = rgb2ycbcr(sai1);
            img1(:,:,:,p,q) = sai1;

            sai2 = imread([folder2,num2str(ind),'.png']);
            sai2 = rgb2ycbcr(sai2);
            img2(:,:,:,p,q) = sai2;
        end
    end

    count = count+1;    
    LFI1(count, :, :, :, :, :) = img1;    % [N,h,w,3,ah,aw]
    LFI2(count, :, :, :, :, :) = img2;   
    %% generate data
    LFI1 = permute(LFI1,[1,4,3,2,6,5]);   %[N,h,w,3,ah,aw]--->[N,3,w,h,aw,ah]
    LFI2 = permute(LFI2,[1,4,3,2,6,5]);   %[N,h,w,3,ah,aw]--->[N,3,w,h,aw,ah]
    %% save data
    if exist(savename,'file')
      fprintf('Warning: replacing existing file %s \n',savename);
      delete(savename);
    end 

    h5create(savename,'/LFI1',size(LFI1),'Datatype','uint8');
    h5write(savename, '/LFI1', LFI1);
    h5create(savename,'/LFI2',size(LFI2),'Datatype','uint8');
    h5write(savename, '/LFI2', LFI2);

    h5disp(savename);
end