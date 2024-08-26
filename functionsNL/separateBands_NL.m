function [ separate ] = separateBands_NL( IrawFFT,phaOff,bands,fac,m2)
%UNTITLED 此处显示有关此函数的摘要
%   phaOff：相位偏移量
%   bands： 分量级次
%   fac：   调制度
%   m2:    非线性阶次调制度


phaPerBand=size(IrawFFT,3);
phases=zeros(1,phaPerBand);

for p=1:phaPerBand
    phases(p)=(2*pi*(p-1))/phaPerBand+phaOff;                                                                              
end

separate=separateBands_final(IrawFFT,phases,bands,fac,m2);
end


function [ separate ] = separateBands_final( IrawFFT,phases,bands,fac,m2)
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

length=size(IrawFFT,3);
W=exp(1i*phases'*comp);
for I=1:length
    W(I,:)=W(I,:).*compfac;
end


siz=size(IrawFFT(:,:,1));

S=reshape(IrawFFT,[prod(siz),length])*pinv(W');
Sk=reshape(S,[siz,bands*2-1]);
separate=Sk;
end


