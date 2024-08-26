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

NA= 1.4;                      
lambda= 0.488;                  
Mag = 100;                                                                            
PixelSize = 6.5;                                                                  
param.nrPhases = 5; param.nrDirs = 5;
param.nrBands = 3;
param.NImage = param.nrPhases * param.nrDirs;
FWHM = (0.5*lambda/NA)/(PixelSize/Mag);

filename = 'RawSIMData_gt_003.tif';               %  input your filename      
pathname ='.\input\';                             %  input your path                           

param_savepath = [pathname,'Params/'];                                     
result_savepath = [pathname,'Results/'];                                   
param.rawPathName = [pathname,filename];
if (exist(result_savepath,"dir") == 0)
    mkdir(result_savepath)
end
if (exist(param_savepath,"dir") == 0)
    mkdir(param_savepath)
end

paramType = 'dft';                                                         
paramPrefix = 'illu_para_struct_ch';
switch paramType
    case 'hifi'
        paramSuffix = '_hifi.mat';
    case 'cor'
        paramSuffix = '_cor.mat';
    case 'pca'
        paramSuffix = '_pca.mat';
    case 'dft'
        paramSuffix = '_dft.mat';
end
paraName = [paramPrefix,'1',paramSuffix];

for j = 1:param.nrDirs*param.nrPhases
    Iraw(:,:,j) = importImages(double(imread(param.rawPathName,j)));
end
param.NPixel = size(Iraw,1);
param = parameter_set_GPU(param.NPixel,PixelSize,NA,lambda,Mag,param.NImage,...
    param.nrDirs,param.nrPhases,param.nrBands);

param.phaOff=0;
usfac = 300;
param.phaOff=0;
param.fac=ones(1,param.nrBands);

recon = recon_set_GPU(param);

for I=1:param.NImage
    recon.IIrawFFT(:,:,I)=FFT2D(Iraw(:,:,I),false);
end

cnt=[param.imgSize/2+1,param.imgSize/2+1];
param.cutoff=1000/(0.5*param.lambda/param.NA);
[x,y]=meshgrid(1:param.imgSize,1:param.imgSize);
rad=sqrt((y-cnt(1)).^2+(x-cnt(2)).^2);
Mask=double(rad<=1.0*(param.cutoff/param.cyclesPerMicron+1));
NotchFilter0=getotfAtt(param.imgSize,param.OtfProvider.cyclesPerMicron,0.5*param.cutoff,0,0);
NotchFilter=gpuArray(single(NotchFilter0.*Mask));
Mask2=double(rad<=1.10*(param.cutoff/param.cyclesPerMicron+1));
Mask3=double(rad>=0.85*(param.cutoff/param.cyclesPerMicron+1));
NotchFilter2=NotchFilter0.*Mask2;


iterN = 33;
elapsed_time = zeros(iterN,1);

for it = 1:iterN
    D = gpuDevice;
    wait(D)
    t1 = tic;
    for I=1:param.nrDirs

        separateII=separateBands_NL_optimal(I,recon,param);


        c0=separateII(:,:,1);
        c1=separateII(:,:,2);

        c0=c0./(max(max(abs(separateII(:,:,1))))).*NotchFilter;
        c1=c1./(max(max(abs(separateII(:,:,2))))).*NotchFilter;

        c0=FFT2D(c0,false);
        c1=FFT2D(c1,false);
        tempc0 = c0; tempc1 = c1;


        c1=fftshift(c1).*conj(fftshift(c0));
        c1=c1./max(max(c1));

        CC = placeFreqNLSIM_GPU(c1,2);
        vec = ifft2(ifftshift(CC));

        temp = vec;
        temp = log(1+abs(temp));
        temp = temp./max(max(temp));
        CCfinal = temp;

        [max1,loc1] = max(CCfinal);
        [max2,loc2] = max(max1);
        rloc=loc1(loc2);cloc=loc2;
        CCmax=CCfinal(rloc,cloc);

        [m,n] = size(CCfinal); md2 = fix(m/2); nd2 = fix(n/2);
        if rloc > md2
            row_shift = rloc - m - 1;
        else
            row_shift = rloc - 1;
        end
        if cloc > nd2
            col_shift = cloc - n - 1;
        else
            col_shift = cloc - 1;
        end
        row_shift=row_shift/2;
        col_shift=col_shift/2;

        row_shift = round(row_shift*usfac)/usfac;
        col_shift = round(col_shift*usfac)/usfac;
        dftshift = fix(ceil(usfac*1.5)/2);

        CC = conj(dftups(tempc0.*conj(tempc1),ceil(usfac*1.5),ceil(usfac*1.5),usfac,...
            dftshift-row_shift*usfac,dftshift-col_shift*usfac))/(md2*nd2*usfac^2);

        [max1,loc1] = max(CC);
        [max2,loc2] = max(max1);
        rloc = loc1(loc2); cloc = loc2;

        rloc = rloc - dftshift - 1;
        cloc = cloc - dftshift - 1;
        row_shift = row_shift + rloc/usfac;
        col_shift = col_shift + cloc/usfac;
        row_shift = -row_shift;
        col_shift = -col_shift;

        p1 = getphases(recon,param.OtfProvider,separateII(:,:,1),separateII(:,:,2),col_shift,row_shift);

        param.Dir(I).px = -col_shift;
        param.Dir(I).py = -row_shift;
        param.Dir(I).phaOff = -angle(p1);
        Temp_m1 = abs(p1);
        Temp_m2 = param.m2;
        param.Dir(I).modul(1)=Temp_m1;
        param.Dir(I).modul(2)=Temp_m2;
    end
    wait(D)
    elapsed_time(it) = toc(t1);
end

illumination(1).vector = [param.Dir(1).px param.Dir(1).py];
illumination(2).vector = [param.Dir(2).px param.Dir(2).py];
illumination(3).vector = [param.Dir(3).px param.Dir(3).py];
illumination(4).vector = [param.Dir(4).px param.Dir(4).py];
illumination(5).vector = [param.Dir(5).px param.Dir(5).py];

illumination(1).phase(1) = -param.Dir(1).phaOff; illumination(1).phase(2) = illumination(1).phase(1)-2*pi/5;illumination(1).phase(3) = illumination(1).phase(1)-4*pi/5;illumination(1).phase(4) = illumination(1).phase(1)-6*pi/5;illumination(1).phase(5) = illumination(1).phase(1)-8*pi/5;
illumination(2).phase(1) = -param.Dir(2).phaOff; illumination(2).phase(2) = illumination(2).phase(1)-2*pi/5;illumination(2).phase(3) = illumination(2).phase(1)-4*pi/5;illumination(2).phase(4) = illumination(2).phase(1)-6*pi/5;illumination(2).phase(5) = illumination(2).phase(1)-8*pi/5;
illumination(3).phase(1) = -param.Dir(3).phaOff; illumination(3).phase(2) = illumination(3).phase(1)-2*pi/5;illumination(3).phase(3) = illumination(3).phase(1)-4*pi/5;illumination(3).phase(4) = illumination(3).phase(1)-6*pi/5;illumination(3).phase(5) = illumination(3).phase(1)-8*pi/5;
illumination(4).phase(1) = -param.Dir(4).phaOff; illumination(4).phase(2) = illumination(4).phase(1)-2*pi/5;illumination(4).phase(3) = illumination(4).phase(1)-4*pi/5;illumination(4).phase(4) = illumination(4).phase(1)-6*pi/5;illumination(4).phase(5) = illumination(4).phase(1)-8*pi/5;
illumination(5).phase(1) = -param.Dir(5).phaOff; illumination(5).phase(2) = illumination(5).phase(1)-2*pi/5;illumination(5).phase(3) = illumination(5).phase(1)-4*pi/5;illumination(5).phase(4) = illumination(5).phase(1)-6*pi/5;illumination(5).phase(5) = illumination(5).phase(1)-8*pi/5;

illumination(1).s = param.Dir(1).modul; 
illumination(2).s = param.Dir(2).modul;
illumination(3).s = param.Dir(3).modul;
illumination(4).s = param.Dir(4).modul;
illumination(5).s = param.Dir(5).modul;

numLayer = 1;
numChannel = 1;
M = param.imgSize; N = param.imgSize;
save([param_savepath,'illu_para_struct_ch1_dft.mat'],'illumination','numLayer','numChannel','N','M');



function out=dftups(in,nor,noc,usfac,roff,coff)

[nr,nc]=size(in);
if exist('roff')~=1, roff=0; end
if exist('coff')~=1, coff=0; end
if exist('usfac')~=1, usfac=1; end
if exist('noc')~=1, noc=nc; end
if exist('nor')~=1, nor=nr; end
kernc=exp( (-1i*2*pi/(nc*usfac))*( ifftshift( (0:nc-1) ).' - floor(nc/2) ) * ( (0:noc-1) - coff ) ); 
kernr=exp((-1i*2*pi/(nr*usfac))*( (0:nor-1).' - roff )*( ifftshift( (0:nr-1) ) - floor(nr/2)  )); 
out=kernr*in*kernc;
return
end

function phase = getphases(recon,otf,band0,band1,kx,ky)
    recon.wt0 = writeOtfVector_GPU(recon.wt0,otf,1,kx,ky);
    recon.wt1 = writeOtfVector_GPU(recon.wt1,otf,1,-kx,-ky);
    
    weightLimit = 0.05;
    siz=size(band0);
    w=siz(2);
    h=siz(1);
    cnt=[siz(1)/2+1,siz(2)/2+1];
    x=1:w;
    y=1:h;
    [x,y]=meshgrid(x,y);
    rad=sqrt((y-cnt(1)).^2+(x-cnt(2)).^2);
    max=sqrt(kx*kx+ky*ky);
    ratio=rad./max;

    mask1=(abs(recon.weight0)<weightLimit) | (abs(recon.wt0)<weightLimit);
    cutCount=length(find(mask1~=0));
    band0(mask1)=0;
    
    mask2=abs(recon.weight1)<weightLimit | abs(recon.wt1)<weightLimit;
    band1(mask2)=0;

    recon.weight0(recon.weight0==0)=1;
    recon.weight1(recon.weight1==0)=1;
    recon.wt0(recon.wt0==0)=1;
    recon.wt1(recon.wt1==0)=1;

    mask=ratio<0.15 | ratio>(1-0.15);
    band0(mask)=0;

    idx = repmat({':'}, ndims(mask), 1); 
    n = size(mask, 1 ); 
    if kx>0
        idx{2}=[round(kx)+1:n 1:round(kx)];
    else
        idx{2}=[n-round(abs(kx))+1:n 1:n-round(abs(kx))];
    end
    if ky>0
        idx{1}=[round(ky)+1:n 1:round(ky)];
    else
        idx{1}=[n-round(abs(ky))+1:n 1:n-round(abs(ky))];
    end
    mask0=mask(idx{:});
    band1(mask0)=0;

    b0 = FFT2D(band0,true);
    b1 = fourierShift(FFT2D(band1,true),kx,ky);
    b1=b1.*conj(b0);
    scal=1/sum(sum(real(b0).^2+imag(b0).^2));  
    phase=sum(sum(b1))*scal;  
end

function [ ret ] = writeOtfVector_GPU( vec,otf,band, kx, ky )
    ret=otfToVector_GPU(vec,otf,band,kx,ky,0,1);
end

function [ ret ] = otfToVector_GPU( vec,otf,band,kx,ky,useAtt,write )
    siz=size(vec);
    w=siz(2);
    h=siz(1);
    cnt=siz/2+1;
    kx=kx+cnt(2);
    ky=ky+cnt(1);

    x=gpuArray(single(1:w));
    y=gpuArray(single(1:h));  
    [x,y]=meshgrid(x,y);

    rad=hypot(y-ky,x-kx);
    cycl=rad.*otf.cyclesPerMicron;

    mask=cycl>otf.cutoff;
    cycl(mask)=0;

    val=getOtfVal(otf,band,cycl,useAtt);
    if write==0
        vec=vec.*val;
    else
        vec=val;
    end
    vec(mask)=0;
    ret=vec;

end

function [ val ]= getOtfVal(otf,band,cycl,att)
    pos=cycl./otf.cyclesPerMicron;
    cpos=pos+1;

    lpos=floor(cpos);
    hpos=ceil(cpos);
    f=cpos-lpos;

    if att==1
        retl=otf.valsAtt(lpos).*(1-f);
        reth=otf.valsAtt(hpos).*f; 
        val=retl+reth;
    else
        retl=otf.vals(lpos).*(1-f);
        reth=otf.vals(hpos).*f;
        val=retl+reth;
    end

    mask=ceil(cpos)>otf.sampleLateral;
    val(mask)=0;

end





