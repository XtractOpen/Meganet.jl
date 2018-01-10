export nImgIn, nImgOut, nFeatIn, nFeatOut, nTheta

abstract type abstractConvKernel end

## All convKernel types are assumed to have fields nImage (size of the image) and sK (size of the Convolution Kernel)

function nImgIn(this::abstractConvKernel)
    return [this.nImg[1], this.nImg[2], this.sK[3]]
end

function nImgOut(this::abstractConvKernel)
   return [this.nImg[1], this.nImg[2], this.sK[4]]
end

function nFeatIn(this::abstractConvKernel)
    return prod(nImgIn(this));
end
function nFeatOut(this::abstractConvKernel)
   return prod(nImgOut(this));
end

function nTheta(this::abstractConvKernel)
   return prod(this.sK);
end