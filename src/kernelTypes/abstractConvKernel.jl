export nImgIn, nImgOut, nFeatIn, nFeatOut, nTheta, getOp, initTheta

abstract type abstractConvKernel{T} <: AbstractMeganetElement{T} end

## All convKernel types are assumed to have fields nImage (size of the image) and sK (size of the Convolution Kernel)

function nImgIn(this::abstractConvKernel)
    return [this.nImg[1]; this.nImg[2]; this.sK[3]]
end

function nImgOut(this::abstractConvKernel)
   return [this.nImg[1]; this.nImg[2]; this.sK[4]]
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

function getOp(this::abstractConvKernel{T},theta::Array{T}) where {T <: Number}

    m = prod(nImgOut(this))
    n = prod(nImgIn(this))

    A = LinearOperator{T}(m,n,false,false,
                        v -> Amv(this,theta,v),
                        Nullable{Function}(),
                        w -> ATmv(this,theta,w))
    return A
end

function initTheta(this::abstractConvKernel)

    sd    = 0.1;
    theta = sd*randn(T,prod(this.sK));
    #id1 = find(theta>2*sd);
    #theta(id1[:]) = randn(numel(id1),1);

    #id2 = find(theta< -2*sd);
    #theta(id2(:)) = randn(numel(id2),1);
    #theta = max(min(2*sd, theta),-2*sd);
    return theta
end