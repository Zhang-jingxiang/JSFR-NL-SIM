function [ out ] = placeFreqNLSIM_GPU( in,t )

siz=size(in);
w=siz(2);
h=siz(1);
out=gpuArray(single(zeros(t*w,t*h)));

out((t-1)*h/2+1:(t+1)*h/2,(t-1)*w/2+1:(t+1)*w/2)=in;


end

