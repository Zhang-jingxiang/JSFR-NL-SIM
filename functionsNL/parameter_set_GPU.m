function [out] = parameter_set_GPU(NPixel,Pixelsize,NA,lambda,mag,NImage,...
                    nrDirs,nrPhases,nrBands)

    param.mag = mag;
    param.nrPhases = nrPhases;
    param.nrDirs = nrDirs;
    param.NImage = NImage;                                                 % Image stack
    param.imgSize = NPixel;                                                % Image size
    param.NPixel = NPixel;
    param.micronsPerPixel = Pixelsize/mag;                                 % Pixel size  unit mm 0.065mm  
    param.cyclesPerMicron = 1/(NPixel*param.micronsPerPixel);              % Frequency domain units
    param.NA = NA;                                                         % Objective NA
    param.lambda = lambda*1000;                                            % Wavelength (nm)
    param.cutoff = 1000/(0.5*param.lambda/param.NA);                       % Cutoff frequency 2NA
    param.sampleLateral = ceil(param.cutoff/param.cyclesPerMicron)+1;      % Cutoff frequency radius
    param.nrBands = nrBands;                                                     % Bands
    param.phaOff=0;                                                        % Phase offset initial value
    param.fac=ones(1,param.nrBands);                                       % Modulation initial value 
    param.attStrength = 0;
    param.OtfProvider = SimOtfProvider(param,param.NA,param.lambda,1);  % Generate approximate OTF, turning the value of a, otf was changed and if we turn it about 0.01, the size of psf increased and resolution drop.               
    % PSF = abs(otf2psf((param.OtfProvider.otf)));                           % Generate approximate PSF 
    param.OTF=param.OtfProvider.otf;                                       
    % param.psf=abs(otf2psf((param.OtfProvider.otf)));
    
    param.m2 = 0.36;                                                       % 0.16~0.5 Default: 1/6; mainly used in components seperation

    out=param;
end

function [ret ] = SimOtfProvider(param,NA,lambda,a)
    ret.na = NA;
    ret.lambda=lambda;
    % ret.cutoff=1000/(0.61*lambda/NA);
    ret.cutoff=1000/(0.5*lambda/NA);
    ret.imgSize=param.imgSize;
    ret.cyclesPerMicron=param.cyclesPerMicron;
    ret.sampleLateral=ceil(ret.cutoff/ret.cyclesPerMicron)+1;
    ret.estimateAValue=a;
    ret.maxBand=2;
    ret.attStrength=param.attStrength;
    ret.attFWHM=1.0;
    ret.useAttenuation=1;
    ret=fromEstimate(ret);
    
    ret.otf=gpuArray(single(zeros(param.imgSize,param.imgSize)));
    ret.otfatt=gpuArray(single(zeros(param.imgSize,param.imgSize)));
    ret.onlyatt=gpuArray(single(zeros(param.imgSize,param.imgSize)));

    ret.otf=writeOtfVector(ret.otf,ret,1,0,0);
    ret.onlyatt=getonlyatt(ret,0,0);
    ret.otfatt=ret.otf.*ret.onlyatt;

end

function [va]=valIdealOTF(dist)
    if dist<0 || dist>1
        va=0;
        return;
    end
    va=(1/pi)*(2*acos(dist)-sin(2*acos(dist)));                            % HiFi actually used otf.
end                                                                        % HiFi sup equ 13. originally from born principle of optics.

function [va]= valAttenuation(dist,str,fwhm)
    va=(1-str*(exp(-power(dist,2)/(power(0.5*fwhm,2)))).^1);
end

function [ret] = fromEstimate(ret)
    ret.isMultiband=0;
    ret.isEstimate=1;
    vals1=gpuArray(single(zeros(1,ret.sampleLateral)));
    valsAtt=gpuArray(single(zeros(1,ret.sampleLateral)));
    valsOnlyAtt=gpuArray(single(zeros(1,ret.sampleLateral)));


    for I=1:ret.sampleLateral
        v=abs(I-1)/ret.sampleLateral;
        r1=valIdealOTF(v)*power(ret.estimateAValue,v);                     % modified otf by utilzing dampling factor. as shown above ret.a.
        vals1(I)=r1;
    end

    for I=1:ret.sampleLateral
        dist=abs(I-1)*ret.cyclesPerMicron;
        valsOnlyAtt(I)=valAttenuation(dist,ret.attStrength,ret.attFWHM);
        valsAtt(I)=vals1(I)*valsOnlyAtt(I);
    end

    ret.vals=vals1;
    ret.valsAtt=valsAtt;
    ret.valsOnlyAtt=valsOnlyAtt;

end

function [ onlyatt ] = getonlyatt( ret,kx,ky )
    w=ret.imgSize;
    h=ret.imgSize;
    siz=[h w];
    cnt=siz/2+1;
    kx=kx+cnt(2);
    ky=ky+cnt(1);
    onlyatt=zeros(h,w);

    y=gpuArray(single(1:h));
    x=gpuArray(single(1:w));
    [x,y]=meshgrid(x,y);
    rad=hypot(y-ky,x-kx);
    cycl=rad.*ret.cyclesPerMicron;
    onlyatt=valAttenuation(cycl,ret.attStrength,ret.attFWHM);
end
