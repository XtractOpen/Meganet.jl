export convGEMMKernel,Amv,ATmv,transposeTest,getConvGEMMKernel

mutable struct convGEMMKernel{T} <: abstractConvKernel{T}
    nImg
    sK
end
function getConvGEMMKernel(TYPE::Type,nImg,sK)
	return convGEMMKernel{TYPE}(copy(nImg),copy(sK));
end

function Amv(this::convGEMMKernel{T},theta::Array{T},Y::Array{T}) where {T}
    ## We assume that the data Y is held in the order XYCN.
	sK = this.sK;
	nImg = this.nImg;
	nex   = div(numel(Y),prod(nImgIn(this)))
    # compute convolution
	Y     = reshape(Y,nImg[1],nImg[2],this.sK[3],nex);
    AY    = zeros(T,nImg[1]*nImg[2],this.sK[4],nex);
	aux   = zeros(T,nImg[1],nImg[2],this.sK[3]);
    AYk   = zeros(T,nImg[1]*nImg[2],this.sK[4]);
	### reshape the kernels for gemm!:
	K = reshape(theta,tuple(sK...));
	KK = Array{Array{T,2}}(sK[1],sK[2]);
	for k1 = 1:sK[1]
		for k2 = 1:sK[2]
			@inbounds KK[k1,k2] = K[k1,k2,:,:]';
		end
	end
	shiftX = [0;-1;0;0;1;0];
	shiftT = [1;0;0;0;0;-1];

    for k = 1:nex
		AYk = multConv2Dblock(Y,KK, AYk,aux,shiftX,shiftT,k);
		@inbounds AY[:,:,k] = AYk;
		AYk[:] = zero(T)
	end
    AY = reshape(AY,:,nex);
    return AY
end

function ATmv(this::convGEMMKernel{T},theta::Array{T},Z::Array{T}) where {T}
	nImg  = this.nImg;
	sK    = this.sK;
    nex   =  div(numel(Z),prod(nImgOut(this)));
    K     = reshape(theta,tuple(sK...));
	Z     = reshape(Z,nImg[1],nImg[2],sK[4],nex);
	aux     = zeros(T,nImg[1],nImg[2],sK[4]);
	ATZ   = zeros(T,nImg[1]*nImg[2],sK[3],nex);
	ATZk  = zeros(T,nImg[1]*nImg[2],sK[3]);

	### reshape the kernels for gemm!:
	KK = Array{Array{T,2}}(sK[1],sK[2]);
	for k1 = 1:sK[1]
		for k2 = 1:sK[2]
			@inbounds KK[k1,k2] = K[k1,k2,:,:];
		end
	end
	## flipping:
	KK = flipdim(flipdim(KK,2),1);
	shiftX = [0;-1;0;0;1;0];
	shiftT = [1;0;0;0;0;-1];
    for k = 1:nex
		ATZk = multConv2Dblock(Z,KK, ATZk,aux,shiftX,shiftT,k);
		@inbounds ATZ[:,:,k] = ATZk;
		ATZk[:] = zero(T)
	end
    ATZ = reshape(ATZ,:,nex);
    return ATZ
end

function Jthetamv(this::convGEMMKernel{T},dtheta::Array{T},dummy,Y::Array{T},temp=nothing) where {T}
    nex    =  div(numel(Y),nFeatIn(this));
    Z      = Amv(this,dtheta,Y);
    return Z
end

function JthetaTmv(this::convGEMMKernel{T},Z::Array{T},dummy,Y::Array{T}) where {T}
     # derivative of Z*(A(theta)*Y) w.r.t. theta
	sK = this.sK;
	nImg = this.nImg;
	nex   = div(numel(Y),prod(nImgIn(this)))
    # compute convolution
	Y     = reshape(Y,nImg[1],nImg[2],this.sK[3],nex);
	Z	  = reshape(Z,nImg[1]*nImg[2],this.sK[4],nex);
	Zk    = zeros(T,nImg[1]*nImg[2],this.sK[4]);
	aux     = zeros(T,nImg[1],nImg[2],this.sK[3]);
	### reshape the kernels for gemm!:
	dtheta = zeros(T,tuple(sK...));
	KK = Array{Array{T,2}}(sK[1],sK[2]);
	for k1 = 1:sK[1]
		for k2 = 1:sK[2]
			@inbounds KK[k1,k2] = zeros(T,sK[3],sK[4]);
		end
	end
	shiftX = [0;-1;0;0;1;0];
	shiftT = [1;0;0;0;0;-1];
    for k = 1:nex
		getColumn!(Z,Zk,k);
		multConv2Dblock(Y,KK, Zk,aux,shiftX,shiftT,k,doDerivative = 1);
	end
	### Assemble the kernels from gemm!:
	for k1 = 1:sK[1]
		for k2 = 1:sK[2]
			@inbounds dtheta[k1,k2,:,:] = KK[k1,k2];
		end
	end
    dtheta = reshape(dtheta,tuple(this.sK...));
    return dtheta
end



function getColumn!(Z::Array{T},Zk::Array{T},k::Int64) where {T}
for c=1:size(Z,2)
	for j=1:size(Z,1)
		@inbounds	Zk[j,c] = Z[j,c,k];
	end
end
end

function multConv2Dblock(x::Array{T},K::Array, y::Array{T}, t::Array{T},shiftX,shiftT,imIdx;doDerivative = 0) where {T}
## y = K*x
## K - 3X3 array of Arrays
## x - a vector of length |nImgag+2|*cin (zero padded)
## y - a vector of length |nImgag|*cout

nImg1 = size(x,1);
nImg2 = size(x,2);
cin = size(x,3);
cout = size(y,2);
OneType = one(T);

kernelWidth = size(K,1);
# y = reshape(y,nImg1*nImg2,cout); # it is supposed to be of this shape...
k=1;
jt=0;it=0;jt=0;jx=0;
for p = 1:2:2*kernelWidth
	for q = 1:2:2*kernelWidth
		t = reshape(t,nImg1,nImg2,cin);
		for cc = 1:cin
			jx = 1+shiftX[p];
			jt = 1+shiftT[p];
			if jt > 1
				@inbounds t[:,1:(jt-1),cc] = 0.0;
			end
			while jt < nImg2+shiftT[p+1]
				it = 1+shiftT[q];
				ix = 1+shiftX[q];
				if it > 1
					@inbounds t[1:(it-1),jt,cc] = 0.0;
				end
				while it < nImg1+shiftT[q+1]
					@inbounds t[it,jt,cc] = x[ix,jx,cc,imIdx];
					it+=1;ix+=1;
				end
				if it <= nImg1
					@inbounds t[it:nImg1,jt,cc] = 0.0;
				end
				jt+=1;jx+=1;
			end
			if jt <= nImg2
				@inbounds t[:,jt:nImg2,cc] = 0.0;
			end
		end
		t = reshape(t,nImg1*nImg2,cin);
		if doDerivative == 0
			BLAS.gemm!('N','T',OneType,t,K[k],OneType,y);
		else
			BLAS.gemm!('T','N',OneType,t,y,OneType,K[k]);
		end
		k+=1;
	end
end
return y;
end


function transposeTest()
	nImage = [16,16];
	sK = [3,3,2,4];
	TYPE = Float64;
	K = randn(TYPE,tuple(sK...));
	Y = randn(TYPE,nImage[1],nImage[2],sK[3],2);
	Z = randn(TYPE,nImage[1],nImage[2],sK[4],2);
	Kernel2 = convGEMMKernel(nImage,sK);
	AY = Amv(Kernel2,K,Y);
	ATZ = ATmv(Kernel2,K,Z);
	println(vecdot(Z,AY));
	println(vecdot(ATZ,Y));

	println(vecdot(Z,Jthetamv(Kernel2,K,[],Y)));
	println(vecdot(K,JthetaTmv(Kernel2,Z,[],Y)));

end
