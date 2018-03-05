export convMKLKernel,Amv,ATmv,getConvMKLKernel
include(Pkg.dir("Meganet")*"/src/mkl/callmkldnn.jl")
using DistributedArrays
mutable struct convMKLKernel{T} <: AbstractConvKernel{T}
    nImg    		:: Array{Int,1}
    sK      		:: Array{Int,1}
	shiftX  		:: Array{Int,1}
	shiftT  		:: Array{Int,1}
	aux_sk3 		:: Array{T, 3}
	aux_sk4 		:: Array{T, 3}
	KK 				:: Array{Array{T,2}}

end
function getConvMKLKernel(TYPE::Type,nImg,sK)

	if sK[1] == 1 && sK[2] == 1
		shiftX = [0;0];
		shiftT = [0;0];
	elseif sK[1] == 3 && sK[2] == 3
		shiftX = [0;-1;0;0;1;0];
		shiftT = [1;0;0;0;0;-1];
	else
		error("Code only supports 1X1 and 3X3 convolutions");
	end

	aux_sk3 = zeros(TYPE,nImg[1],nImg[2],sK[3]);
	aux_sk4 = zeros(TYPE,nImg[1],nImg[2],sK[4]);

	KK = Array{Array{TYPE,2}}(sK[1],sK[2]);

	return convMKLKernel{TYPE}(copy(nImg),copy(sK),shiftX,shiftT,aux_sk3,aux_sk4,KK);
end

function Amv(this::convMKLKernel{T},theta::Array{T},Y::Array{T}) where {T<:Number}
    ## We assume that the data Y is held in the order XYCN.
	sK = this.sK;

    nImg    = this.nImg;
    nex     = div(numel(Y),prod(nImgIn(this)))
    Y       = reshape(Y,nImg[1],nImg[2],this.sK[3],nex);
    K       = reshape(theta, sK[1], sK[2], sK[3], sK[4])

    AY      = convMKL(K,Y);
    AY_out  = reshape(AY,nImg[1]*nImg[2]*sK[4], nex);

	return AY_out
end

function ATmv(this::convMKLKernel{T},theta::Array{T},Zin::Array{T}) where {T<:Number}
	nImg  = this.nImg;
	sK    = this.sK;
    nex   =  div(numel(Zin),prod(nImgOut(this)));
    K     = reshape(theta, sK[1], sK[2], sK[3], sK[4]);
    Z     = reshape(Zin,nImg[1],nImg[2],sK[4],nex);

    ATZ = convTMKL(K,Z);
    ATZ_out = reshape(ATZ,:,nex);

    return ATZ_out
end

function Jthetamv(this::convMKLKernel{T},dtheta::Array{T},dummy::Array{T},Y::Array{T},temp=nothing) where {T<:Number}
    nex    =  div(numel(Y),nFeatIn(this));
    Z      = Amv(this,dtheta,Y);
    return Z
end

function JthetaTmv(this::convMKLKernel{T}, Zin::Array{T}, dummy::Array{T}, Yin::Array{T}) where {T<:Number}
	sK     = this.sK
	nImg   = this.nImg
    nex    = div(numel(Yin),prod(nImgIn(this)))
    K      = zeros(T, sK[1], sK[2], sK[3], sK[4])
    Y      = reshape(Yin,nImg[1],nImg[2],this.sK[3],nex);
    Z      = reshape(Zin,nImg[1],nImg[2],sK[4],nex);

    dtheta = convDerivMKL(K,Z,Y);
    dtheta_out = reshape(dtheta, sK[1], sK[2], sK[3], sK[4])

    return dtheta_out
end
