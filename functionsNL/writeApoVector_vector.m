function [ret] = writeApoVector_vector(vec,otf,cutoff)
    cutoff_cpu = gather(cutoff);
    [h,w]=size(vec);
    cnt=[h/2+1,w/2+1];
    [x,y]=meshgrid(1:h,1:w);
    rad=hypot(y-cnt(1),x-cnt(2));
    cycl=rad*otf.cyclesPerMicron;
    frac=cycl/(otf.cutoff*cutoff_cpu);
    
    valIdealotf = (1/pi)*(2*acos(frac)-sin(2*acos(frac)));
    valIdealotf(frac<0 | frac>1) = 0;


    ret = valIdealotf;
end
