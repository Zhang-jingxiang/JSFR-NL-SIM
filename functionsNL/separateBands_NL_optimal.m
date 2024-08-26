function [separate] = separateBands_NL_optimal(I,recon,param)

siz=size(recon.IIrawFFT(:,:,1));
length = param.nrDirs;
S=reshape(recon.IIrawFFT(:,:,(I-1)*(param.nrDirs)+1:(I-1)*(param.nrDirs)+param.nrPhases),[prod(siz),length])*pinv(recon.W');
Sk=reshape(S,[siz,param.nrBands*2-1]);
separate=Sk;


end