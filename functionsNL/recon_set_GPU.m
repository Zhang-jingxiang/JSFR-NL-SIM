function [recon] = recon_set_GPU(param)

recon.IIrawFFT = gpuArray(single(zeros(param.NPixel,param.NPixel,param.NImage)));
recon.image_shifted = gpuArray(single(zeros(param.NPixel,param.NPixel,param.NImage)));
recon.phases_final = gpuArray(single(zeros(param.nrDirs,param.nrPhases)));

recon.weight0 = gpuArray(single(zeros(param.NPixel,param.NPixel)));
recon.weight1 = gpuArray(single(zeros(param.NPixel,param.NPixel)));
recon.wt0 = gpuArray(single(zeros(param.NPixel,param.NPixel)));
recon.wt1 = gpuArray(single(zeros(param.NPixel,param.NPixel)));

recon.weight0=gpuArray(single(writeOtfVector(recon.weight0,param.OtfProvider,1,0,0)));
recon.weight1=gpuArray(single(writeOtfVector(recon.weight1,param.OtfProvider,1,0,0)));

recon.W = getinvMatrix(param.nrDirs,param.phaOff,param.nrBands,0,0);


end


function [W] = getinvMatrix(length_matrix,phaOff,bands,fac,m2)

phaPerBand=length_matrix;
phases=zeros(1,phaPerBand);

for p=1:phaPerBand
    phases(p)=(2*pi*(p-1))/phaPerBand+phaOff;                                                                              
end

W=separateBands_final(length_matrix,phases,bands,fac,m2);


end


function [ W ] = separateBands_final( length_matrix,phases,bands,fac,m2)
   tempfac=[1,2/3,1/6];        
%    tempfac=[1,0.495,0.14];   
% tempfac=[1,0.30,0.2];        

if fac==0
    fac=tempfac;
else
    fac(2)=fac(2);
    fac(3)=m2;
    modLowLimit1=0.4;                   %% High SNR
    if fac(2)<modLowLimit1
        fac(2)=modLowLimit1;
    end
end


comp=zeros(1,bands*2-1);
comp(1)=0;
for I=2:bands
    comp((I-1)*2)=I-1;
    comp((I-1)*2+1)=-(I-1);
end
compfac=zeros(1,bands*2-1);
compfac(1)=fac(1);
for I=2:bands
    compfac((I-1)*2)=fac(I);
    compfac((I-1)*2+1)=fac(I);
end

length=length_matrix;
W=exp(1i*phases'*comp);
for I=1:length
    W(I,:)=W(I,:).*compfac;
end

end
