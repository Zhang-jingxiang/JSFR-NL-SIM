%% Main reconstruction workflow of the JSFR-NL-SIM
% Author: Jingxiang Zhang and Ming Lei, Xi'an Jiaotong University, 2024/08
% 
% Ref:
% [1] Wen G, Li S, Wang L, et al. High-fidelity structured illumination microscopy
% by point-spread-function engineering[J]. Light: Science & Applications, 2021, 10(1): 70.
% [2] Guizar-Sicairos M, Thurman S T, Fienup J R. Efficient subpixel image 
% registration algorithms[J]. Optics letters, 2008, 33(2): 156-158.
% [3] Cao R, Chen Y, Liu W, et al. Inverse matrix based phase estimation 
% algorithm for structured illumination microscopy[J]. Biomedical Optics Express,
% 2018, 9(10): 5037-5051.
%
% For any question, please contact: jingxiang.zhang@stu.xjtu.edu.cn and ming.lei@mail.xjtu.edu.cn
% 
% We claim a Apache liscence for JSFR-NL-SIM.
clc
clear;
close all;
addpath(genpath('.\functionsNL\'));

%% basic parameter setting
location = '.\input\RawSIMData_gt_003.tif';
save_SR_folder = '.\output\';
NA= 1.49;
lambda= 0.488;
Mag = 100;                                                                 
PixelSize = 6.5;                                                           
param.nrPhases = 5; param.nrDirs = 5;
param.nrBands = 3;
NPixel = 502;
param.NImage = param.nrPhases * param.nrDirs;
FWHM = (0.5*lambda/NA)/(PixelSize/Mag);

param = parameter_set_GPU(NPixel,PixelSize,NA,lambda,Mag,param.NImage,...
    param.nrDirs,param.nrPhases,param.nrBands);

paratype = 'dft'; 
paraprefix = 'illu_para_struct_ch';
switch paratype
    case 'default'
        parasuffix = '.mat';
    case 'fairsim'
        parasuffix = '_fairsim.mat';
    case 'hifi'
        parasuffix = '_hifi.mat';
    case 'dft'
        parasuffix = '_dft.mat';
    case 'pca'
        parasuffix = '_pca.mat';
end

saveloc = '.\input\Params\';
paraname = [paraprefix,'1',parasuffix];
load([saveloc,paraname]);


%% Reconstruction NL-SIM
Iraw = gpuArray(single(zeros(param.NPixel,param.NPixel,param.NImage)));

for j = 1:param.nrDirs*param.nrPhases
    Iraw(:,:,j) = importImages(double(imread(location,j)));
end

contrast_chs = 0.5* [0.6 0.6 0.6 0.6];  
wienerpara = [0.5 0.1 0];                 
amp_chs = [0.9 0.9 1 0.9];             
attFWHM_ch = [3 1 1 1];
sigma_chs = 0.5*param.NPixel*6.5/param.mag*attFWHM_ch;   
                       
beta = 1;                               

contrast = contrast_chs(1); 
AMP = amp_chs(1);   
SIGMA = sigma_chs(1); 
[Y,X] = ndgrid(-0.5:3*param.NPixel-1.5,-0.5:3*param.NPixel-1.5);

sectioning_mask = gpuArray(single(1 - AMP.*exp(-((X-1.5*param.NPixel).^2 + (Y-1.5*param.NPixel).^2)/SIGMA^2)));

%% calculate the filter
[filter,OTF,~] = CalcWOTF_NLv2_GPU(param,illumination);

OTF = gpuArray(single(OTF));
filter = gpuArray(single(filter));
trandiFilter = fftshift(conj(OTF)./(abs(OTF).^2+wienerpara(2)).*sectioning_mask);

%% Prepare parameters
img_set = gpuArray(single(zeros(3*param.NPixel,3*param.NPixel,param.NImage)));
for i=1:param.NImage 
    img_set(:,:,i) = imresize(Iraw(:,:,i),3,'nearest');
end

m = [0.5,0.2];

img = gpuArray(single(zeros(3*param.NPixel,3*param.NPixel)));
Factors = gpuArray(single(zeros(3*param.NPixel,3*param.NPixel,param.NImage )));
for i = 1:param.nrDirs 
    Factors(:,:,param.nrPhases*(i-1)+1:param.nrPhases*i) = GetFactorGeneralized_NL(img,m,illumination(i).phase(1),illumination(i).vector(1),illumination(i).vector(2),2*pi/param.nrPhases);         % �
end

filtered_img_set = gpuArray(single(zeros(3*param.NPixel,3*param.NPixel,param.NImage)));
for i = 1:param.NImage 
    filtered_img_set(:,:,i) = ifft2(fft2(img_set(:,:,i)).*trandiFilter);
    
end

SIM_Img = sum(Factors.*filtered_img_set,3);
SIM = fftshift(fft2(SIM_Img));
SIM_FINAL = SIM.*filter;
SIM_FINAL = real(ifft2(ifftshift(SIM_FINAL)));

SIM_crop = zeros(3*param.NPixel,3*param.NPixel);
SIM_crop(1:3*param.NPixel,1:3*param.NPixel)=SIM_FINAL(1:3*param.NPixel,1:3*param.NPixel);
SIM_crop = uint16(65535*SIM_crop/max(SIM_crop(:)));
imwrite(SIM_crop,[save_SR_folder,'JSFR_NL_SIM.tif']);

