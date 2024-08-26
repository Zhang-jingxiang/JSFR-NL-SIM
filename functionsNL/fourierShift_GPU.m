% This script aims to shift hight-order-components in NL-SIM to specific
% direction.
function [ comshift ] = fourierShift_GPU( vec, kx, ky  )
    % invFFT = ifft2(ifftshift(vec));
    siz=size(vec);
    [x,y]=meshgrid(gpuArray(single(0:siz(2)-1)),gpuArray(single(0:siz(1)-1)));
    x=x/siz(2);
    y=y/siz(1);
    comshift=vec.*exp(2*pi*1i*(ky*y+kx*x));
    % out = fftshift(fft2(comshift));
end

