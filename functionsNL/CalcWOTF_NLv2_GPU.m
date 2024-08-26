function [w,otf,OTF_ideal] = CalcWOTF_NLv2_GPU(param,illumination)

%  illumination --- illumination parameters
%  AMP -----amplitute of attenuation
%  SIGMA -----sigma of attenuation in pixel
%  M,N -----size of image
%  Mag -----Magnification of the system
%  NA  ---- numerical aperture of the system
%  wavelength ------ wavelength
%  apo --------------parameter for apodisation, recommend valve: 0.5, 
%                    lower sacriface the resolution
%  wienerpara ------- 1x2 vector, respectively for w1 and w2
%  beta ----------- dampening factor

%% calculate OTF
temp = param.OtfProvider.otf;
otf = placeFreqNLSIM(temp,3);
clear temp

OTF_ideal=gpuArray(single(zeros(3*param.imgSize,3*param.imgSize)));
K = sqrt(illumination(1).vector(1)^2+illumination(1).vector(2)^2);
cutoff=floor(2*K)/param.sampleLateral+1.0;
OTF_ideal=writeApoVector_vector(OTF_ideal,param.OtfProvider,cutoff);                  % Ideal OTF
OTF_ideal = gpuArray(single(OTF_ideal));
otf = gpuArray(single(otf));
%% calculate mask
Mask=gpuArray(single(ones(3*param.imgSize,3*param.imgSize)));
[x,y]=meshgrid(gpuArray(single(1:size(otf,1))),gpuArray(single(1:size(otf,2))));
cnt=[size(otf,1)/2,size(otf,2)/2];
rad = hypot((x-cnt(2)), (y-cnt(1)))*param.cyclesPerMicron;
R=3;
Int10=0.0;
Int11=0.0;

for i=1:param.nrDirs
    circ=(x-cnt(2)-(2-1)*round(illumination(i).vector(1))).^2+...
        (y-cnt(1)-(2-1)*round(illumination(i).vector(2))).^2;
    Mask(find(circ<=R^2))=Int10;
    circ=(x-cnt(2)+(2-1)*round(illumination(i).vector(1))).^2+...
        (y-cnt(1)+(2-1)*round(illumination(i).vector(2))).^2;
    Mask(find(circ<=R^2))=Int10;
    circ=(x-cnt(2)-(3-1)*round(illumination(i).vector(1))).^2+...
        (y-cnt(1)-(3-1)*round(illumination(i).vector(2))).^2;
    Mask(find(circ<=R^2))=Int11;
    circ=(x-cnt(2)+(3-1)*round(illumination(i).vector(1))).^2+...
        (y-cnt(1)+(3-1)*round(illumination(i).vector(2))).^2;
    Mask(find(circ<=R^2))=Int11;
end

%% calculate W1
amp = [0.9,0.1];
SIGMA = 1.1;
wienerpara = [0.5 0.1];
otfSim = gpuArray(single(zeros(3*param.imgSize,3*param.imgSize)));
otfSim2 = gpuArray(single(zeros(3*param.imgSize,3*param.imgSize)));
otf2 = abs(otf).^2;

secMask   = 1-amp(1).*exp(-power(rad,2)/SIGMA.^2);
secMask00 =  1- (amp(1)/1.03).*exp(-power(rad,2)/SIGMA.^2);         % 1.03
secMask01 =  1- (amp(1)/1.2).*exp(-power(rad,2)/SIGMA.^2);          % 1.2
secMask02 =  1- (amp(1)/3.5).*exp(-power(rad,2)/SIGMA.^2);          % 3.5

otfSec   = otf2.*secMask;
otfSecb1 = otf2.*secMask00;
otfSecb2 = otf2.*secMask01;
otfSecb3 = otf2.*secMask02;


iter = 30;
elapsed_time0 = zeros(iter,1);
gpu = gpuDevice;


for i = 1: param.nrDirs
    for m = 1:3
        if m == 1
            temp1N = otf2/2;
            temp1P = otf2/2;
        elseif m == 2
            temp1P = exact_shift( otfSec,[-(m-1)*illumination(i).vector(2),-(m-1)*illumination(i).vector(1)],1 );
            temp1N = exact_shift( otfSec,[(m-1)*illumination(i).vector(2),(m-1)*illumination(i).vector(1)],1 );
        else
            temp1P = exact_shift( otfSec,[-(m-1)*illumination(i).vector(2),-(m-1)*illumination(i).vector(1)],1 );
            temp1N = exact_shift( otfSec,[(m-1)*illumination(i).vector(2),(m-1)*illumination(i).vector(1)],1 );
        end

        otfSim = otfSim +  temp1N + temp1P;

    end
end
wpower = 2;
w1 = OTF_ideal./(otfSim+wienerpara(1)^wpower);


%% calculate w2
ampChs = [0.9 0.9 1 0.9];

for i = 1: param.nrDirs
    for m = 1:3
        if m == 1
            temp1N = otfSecb1/2;
            temp1P = otfSecb1/2;
        elseif m == 2
            temp1P = exact_shift( otfSecb2,[-(m-1)*illumination(i).vector(2),-(m-1)*illumination(i).vector(1)],1 );
            temp1N = exact_shift( otfSecb2,[(m-1)*illumination(i).vector(2),(m-1)*illumination(i).vector(1)],1 );
        else
            temp1P = exact_shift( otfSecb3,[-(m-1)*illumination(i).vector(2),-(m-1)*illumination(i).vector(1)],1 );
            temp1N = exact_shift( otfSecb3,[(m-1)*illumination(i).vector(2),(m-1)*illumination(i).vector(1)],1 );
        end

        otfSim2 = otfSim2 + temp1N + temp1P;

    end
end

ApoFWHM = 0.4;
A_k= apodize_gauss([size(OTF_ideal,1),size(OTF_ideal,2)], struct('rad',ApoFWHM));

w2 = A_k./(otfSim2+wienerpara(2)^wpower);
secFinal = w1.*w2.*Mask;

%% calculate w
w = w1.*w2.*Mask;
