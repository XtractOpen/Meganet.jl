export convCircKernel ,getConvCircKernel
## For the functions nImgIn, nImgOut, nFeatIn, nFeatOut, nTheta, getOp, initTheta : see abstractConvKernel.jl
## All convKernel types are assumed to have fields nImage and sK
mutable struct convCircKernel{T} <: abstractConvKernel{T}
    nImg :: Array{Int,1}
    sK   :: Array{Int,1}
    iK   :: Array{Int,1}
end

function nImgOut(this::convCircKernel)
    return [this.nImg[1]; this.nImg[2]; this.sK[3]]
 end

function getConvCircKernel(TYPE::Type,nImg,sK)
 
    n3 = [nImg[1],nImg[2],sK[3]]
    Ind = zeros(n3[1],n3[2],n3[3])
    mid = div.(n3,2)
    midK = div.(sK[1:2],2)
    Ind[mid[1]-midK[1]:mid[1]+midK[1],
        mid[2]-midK[2]:mid[2]+midK[2],:] = reshape(collect(1:prod(sK)),sK[1],sK[2],sK[3])
    Ind = circshift(Ind,1-mid)

    
    iK = find(Ind)
	return convCircKernel{TYPE}(nImg,sK,iK)
end



export Amv
function Amv(this::convCircKernel{T},theta::Array{T},Yin::Array{T}) where {T<:Number}

    nex   = div(length(Y),prod(nImgIn(this)))

    n     = nImgIn(this);
    
    Y     = reshape(Yin,(n[1],n[2],this.sK[3],nex))
    # compute convolution
    K = zeros(n[1],n[2],this.sK[3])
    
    K[this.iK] = theta[:];
    
    Kh = fft(K,(1,2,3))
    Yh = fft(Y,(1,2,3))
    
    Z  = real(ifft(Kh.*Yh,(3,2,1)))
    return Z             
end

export ATmv
function ATmv(this::convCircKernel{T},theta::Array{T},Zin::Array{T}) where {T<:Number}
    
    nex   =  div(numel(Z),prod(nImgIn(this)))
    n     = nImgIn(this)
    Z     = reshape(Zin,(n[1],n[2],this.sK[3],nex));
    # compute convolution T
    K = zeros(n[1],n[2],this.sK[3])
    K[this.iK] = theta
    
    Kh = fft(K,(1,2,3))
    Zh = fft(Z,(1,2,3))
    
    Y  = real(ifft(conj(Kh).*Zh,(3,2,1)))
    return Y
end

function Jthetamv(this::convCircKernel{T},dtheta::Array{T},dummy::Array{T},Yin::Array{T},temp=nothing) where {T<:Number}

    nex    =  div(numel(Y),nFeatIn(this))
    Y      = reshape(Yin,:,nex)
    Z      = Amv(this,dtheta,Y)
    return Z
end

function JthetaTmv(this::convCircKernel{T},Zin::Array{T},dummy::Array{T},Yin::Array{T}) where {T<:Number}
    #  derivative of Z*(A(theta)*Y) w.r.t. theta

    nex  =  div(numel(Y),nFeatIn(this))

    n  = nImgIn(this)
    
    Z     = reshape(Zin,n[1],n[2],this.sK[3],nex)
    Y     = reshape(Yin,n[1],n[2],this.sK[3],nex)
    
    Zh = fft(Z,(1,2,3))
    Yh = fft(Y,(1,2,3))
    YZ = sum(conj(Yh).*Zh,4)
    K  = real(ifft(YZ,(3,2,1)))
    dtheta = K[this.iK];

end
