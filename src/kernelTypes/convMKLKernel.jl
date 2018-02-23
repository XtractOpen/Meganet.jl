export convMKLKernel,Amv,ATmv,getConvMKLKernel
include("/home/moumita/.julia/v0.6/test/callmkldnn.jl")
# include("/home/moumita/.julia/v0.6/Meganet/src/kernelTypes/convGEMMKernel.jl")
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
	println("AMV")
	sK = this.sK;
   
    nImg = this.nImg;
    nex   = div(numel(Y),prod(nImgIn(this)))
    
    if sK[1]!=3
        
        AY = Amv(this,theta,Y,1)
    else
        
        Y     = reshape(Y,nImg[1],nImg[2],this.sK[3],nex);
        K = reshape(theta, sK[1], sK[2], sK[3], sK[4])
        AY = convMKL(K,Y);
    end
    AY_out = reshape(AY,nImg[1]*nImg[2]*sK[4], nex);
    
	return AY_out
end
function Amv(this::convMKLKernel{T},theta::Array{T},Y::Array{T}, hack::Int) where {T<:Number}
## We assume that the data Y is held in the order XYCN.

sK = this.sK;
nImg = this.nImg;
nex   = div(numel(Y),prod(nImgIn(this)))

KK = this.KK

# compute convolution
Y     = reshape(Y,nImg[1],nImg[2],this.sK[3],nex);


AY    = Array{T, 3}(nImg[1]*nImg[2],this.sK[4],nex);


aux   = this.aux_sk3;
AYk   = reshape(this.aux_sk4,nImg[1]*nImg[2],sK[4]);
### reshape the kernels for gemm!:
K = reshape(theta, sK[1], sK[2], sK[3], sK[4])
    
for k1 = 1:sK[1]
    for k2 = 1:sK[2]
        @inbounds KK[k1,k2] = K[k1,k2,:,:]';
    end
end

for k = 1:nex
    AYk[:] = zero(T)
    AYk = multConv2Dblock(Y,KK, AYk,aux,this.shiftX,this.shiftT,k);
    @inbounds AY[:,:,k] = AYk;
end
AY_out = reshape(AY,:,nex)

return AY_out
end

function ATmv(this::convMKLKernel{T},theta::Array{T},Zin::Array{T}) where {T<:Number}
	println("ATMV")
	nImg  = this.nImg;
	sK    = this.sK;
    nex   =  div(numel(Zin),prod(nImgOut(this)));
    
    if sK[1]!=3
       
        ATZ = ATmv(this,theta,Zin,1)
    else
        
        K     = reshape(theta, sK[1], sK[2], sK[3], sK[4]);
       
        Z     = reshape(Zin,nImg[1],nImg[2],sK[4],nex);
        ATZ = convTMKL(K,Z);
    end
    ATZ_out = reshape(ATZ,:,nex);
   
    return ATZ_out
end

function ATmv(this::convMKLKernel{T},theta::Array{T},Zin::Array{T},hack::Int) where {T<:Number}
	
	nImg  = this.nImg;
	sK    = this.sK;
    nex   =  div(numel(Zin),prod(nImgOut(this)));
    K     = reshape(theta, sK[1], sK[2], sK[3], sK[4]);
	Z     = reshape(Zin,nImg[1],nImg[2],sK[4],nex);
	aux   = this.aux_sk4;
	# ATZ = this.ATZ
	ATZk  = reshape(this.aux_sk3,nImg[1]*nImg[2],sK[3]); 
	
	
	ATZ   = zeros(T,nImg[1]*nImg[2],sK[3],nex);
	
	### reshape the kernels for gemm!:
	KK = this.KK
	
	for k1 = 1:sK[1]
		for k2 = 1:sK[2]
			@inbounds KK[k1,k2] = K[k1,k2,:,:];
		end
	end
	## flipping:
	KK = flipdim(flipdim(KK,2),1);
	for k = 1:nex
		ATZk[:] = zero(T)
		ATZk = multConv2Dblock(Z,KK, ATZk,aux,this.shiftX,this.shiftT,k);
		@inbounds ATZ[:,:,k] = ATZk;
	end
	ATZ_out = reshape(ATZ,:,nex);
	
	return ATZ_out
end

function Jthetamv(this::convMKLKernel{T},dtheta::Array{T},dummy::Array{T},Y::Array{T},temp=nothing) where {T<:Number}
    nex    =  div(numel(Y),nFeatIn(this));
    Z      = Amv(this,dtheta,Y);
    return Z
end

function JthetaTmv(this::convMKLKernel{T}, Zin::Array{T}, dummy::Array{T}, Yin::Array{T}) where {T<:Number}
	println("JThetaTMV")
	sK = this.sK
	nImg = this.nImg
    nex   = div(numel(Yin),prod(nImgIn(this)))
    K     = zeros(T, sK[1], sK[2], sK[3], sK[4])
    
    if sK[1]!=3
        
        dtheta = JthetaTmv(this,Zin,K,Yin,1)
    else
    Y     = reshape(Yin,nImg[1],nImg[2],this.sK[3],nex);
    
    
    Z     = reshape(Zin,nImg[1],nImg[2],sK[4],nex);
    
    dtheta = convDerivMKL(K,Z,Y);
    end
    dtheta_out = reshape(dtheta, sK[1], sK[2], sK[3], sK[4])
    
    return dtheta_out
end

function JthetaTmv(this::convMKLKernel{T}, Zin::Array{T}, dummy::Array{T}, Yin::Array{T}, hack::Int) where {T<:Number}
    # derivative of Z*(A(theta)*Y) w.r.t. theta
   
    sK = this.sK
   nImg = this.nImg
   nex   = div(numel(Yin),prod(nImgIn(this)))
   # compute convolution
   
   Y     = reshape(Yin, nImg[1], nImg[2], this.sK[3], nex)
   Z	  = reshape(Zin, nImg[1]*nImg[2], this.sK[4], nex)
   
   Zk    = reshape(this.aux_sk4, nImg[1]*nImg[2], this.sK[4]);
   aux   = this.aux_sk3;
   dtheta = zeros(T, sK[1], sK[2], sK[3], sK[4])
   ### reshape the kernels for gemm!:
   KK = this.KK
   for k1 = 1:sK[1]
       for k2 = 1:sK[2]
           @inbounds KK[k1, k2] = zeros(T, sK[3], sK[4])
       end
   end
   for k = 1:nex
       getColumn!(Z, Zk, k)
       multConv2Dblock(Y, KK,  Zk, aux, this.shiftX, this.shiftT, k, doDerivative = 1)
   end
   ### Assemble the kernels from gemm!:
   for k1 = 1:sK[1]
       for k2 = 1:sK[2]
           @inbounds dtheta[k1, k2, :, :] = KK[k1, k2]
       end
   end
   dtheta_out = reshape(dtheta, sK[1], sK[2], sK[3], sK[4])

   return dtheta_out
end
