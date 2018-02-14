export convGEMMKernel,Amv,ATmv,transposeTest,getConvGEMMKernel

mutable struct convGEMMKernel{T} <: AbstractConvKernel{T}
    nImg    :: Array{Int,1}
    sK      :: Array{Int,1}
	shiftX  :: Array{Int,1}
	shiftT  :: Array{Int,1}
	aux_sk3 :: Array{T, 3}
	aux_sk4 :: Array{T, 3}
end
function getConvGEMMKernel(TYPE::Type,nImg,sK)

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
	
	return convGEMMKernel{TYPE}(copy(nImg),copy(sK),shiftX,shiftT,aux_sk3,aux_sk4);
end

function Amv(this::convGEMMKernel{T},theta::Array{T},Y::Array{T}) where {T<:Number}
    ## We assume that the data Y is held in the order XYCN.
	sK = this.sK;
	nImg = this.nImg;
	nex   = div(numel(Y),prod(nImgIn(this)))
    # compute convolution
	Y     = reshape(Y,nImg[1],nImg[2],this.sK[3],nex);
    AY    = Array{T, 3}(nImg[1]*nImg[2],this.sK[4],nex);
	aux   = this.aux_sk3;
    AYk   = reshape(this.aux_sk4,nImg[1]*nImg[2],sK[4]);
	### reshape the kernels for gemm!:
	K = reshape(theta, sK[1], sK[2], sK[3], sK[4])
	KK = Array{Array{T,2}}(sK[1],sK[2]);
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
    AY_out = reshape(AY,:,nex);
    return AY_out
end

function ATmv(this::convGEMMKernel{T},theta::Array{T},Zin::Array{T}) where {T<:Number}
	nImg  = this.nImg;
	sK    = this.sK;
    nex   =  div(numel(Zin),prod(nImgOut(this)));
    K     = reshape(theta, sK[1], sK[2], sK[3], sK[4]);
	Z     = reshape(Zin,nImg[1],nImg[2],sK[4],nex);
	aux   = this.aux_sk4;
	ATZ   = zeros(T,nImg[1]*nImg[2],sK[3],nex);
	ATZk  = reshape(this.aux_sk3,nImg[1]*nImg[2],sK[3]);

	### reshape the kernels for gemm!:
	KK = Array{Array{T,2}}(sK[1],sK[2]);
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

function Jthetamv(this::convGEMMKernel{T},dtheta::Array{T},dummy::Array{T},Y::Array{T},temp=nothing) where {T<:Number}
    nex    =  div(numel(Y),nFeatIn(this));
    Z      = Amv(this,dtheta,Y);
    return Z
end

function JthetaTmv(this::convGEMMKernel{T}, Zin::Array{T}, dummy::Array{T}, Yin::Array{T}) where {T<:Number}
     # derivative of Z*(A(theta)*Y) w.r.t. theta
	sK = this.sK
	nImg = this.nImg
	nex   = div(numel(Yin),prod(nImgIn(this)))
    # compute convolution
	Y     = reshape(Yin, nImg[1], nImg[2], this.sK[3], nex)
	Z	  = reshape(Zin, nImg[1]*nImg[2], this.sK[4], nex)
	Zk    = reshape(this.aux_sk4, nImg[1]*nImg[2], this.sK[4]);
	aux   = this.aux_sk3;
	### reshape the kernels for gemm!:
	dtheta = zeros(T, sK[1], sK[2], sK[3], sK[4])
	KK = Array{Array{T, 2}}(sK[1], sK[2])
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



function getColumn!(Z::Array{T},Zk::Array{T},k::Int64) where {T<:Number}
for c=1:size(Z,2)
	for j=1:size(Z,1)
		@inbounds	Zk[j,c] = Z[j,c,k];
	end
end
end

function multConv2Dblock(x::Array{T},K::Array{Array{T,2},2}, y::Array{T}, tin::Array{T},shiftX,shiftT,imIdx;doDerivative = 0) where {T<:Number}
	## y = K*x
	## K - 3X3 array of Arrays
	## x - a vector of length |nImgag+2|*cin (zero padded)
	## y - a vector of length |nImgag|*cout

	nImg1 = size(x,1);
	nImg2 = size(x,2);
	cin = size(x,3);
	cout = size(y,2);
	OneType = one(T);
	t = reshape(tin,nImg1,nImg2,cin);
	kernelWidth = size(K,1);
	# y = reshape(y,nImg1*nImg2,cout); # it is supposed to be of this shape...
	k=1;
	jt=0;it=0;jt=0;jx=0;
	
	for p = 1:2:2*kernelWidth
		for q = 1:2:2*kernelWidth
			lower = nImg2+shiftT[p+1]  # Move outside of the forloop for increased speed
			upper = nImg1+shiftT[q+1]  # Move outside of the forloop for increased speed
			for cc = 1:cin
				jx = 1+shiftX[p];  # Moving these outside didn't seem to help
				jt = 1+shiftT[p];
				if jt > 1
					###################### Dirichlet #######################
					@inbounds t[:,1:(jt-1),cc] = zero(T);
					###################### Periodic #######################
					# ix = 1+shiftX[q];
					# if shiftT[q] > 0
						#@inbounds t[1,1,cc] = x[end,end,cc,imIdx];
					# end
					# for it = (1+shiftT[q]):upper 
						#@inbounds t[it,1,cc] = x[ix,end,cc,imIdx];
						# ix +=1;
					# end
					# if shiftT[q+1] < 0
						#@inbounds t[end,1,cc] = x[1,end,cc,imIdx];
					# end
					###################### End Periodic #######################
				end
				while jt <= lower 
					it = 1+shiftT[q];
					ix = 1+shiftX[q];
					if it > 1
						for ii = 1:(it-1)
							###################### Dirichlet #######################
							@inbounds t[ii,jt,cc] = zero(T)   #@inbounds t[1:(it-1),jt,cc] = 0.0 - faster unvectorized
							###################### Periodic #######################
							#@inbounds t[ii,jt,cc] = x[end,jx,cc,imIdx];   
						end							
					end
					while it <= upper
						@inbounds t[it,jt,cc] = x[ix,jx,cc,imIdx];
						it+=1;ix+=1;
					end
					if it <= nImg1
						for ii = it:nImg1
							###################### Dirichlet #######################
							@inbounds t[ii,jt,cc] = zero(T)	#@inbounds t[it:nImg1,jt,cc] = 0.0 - faster unvectorized
							###################### Periodic #######################
							# @inbounds t[ii,jt,cc] = x[1,jx,cc,imIdx];	
						end
					end
					jt+=1;jx+=1;

				end
				if jt <= nImg2
					###################### Dirichlet #######################
					@inbounds t[:,jt:nImg2,cc] = zero(T);
					###################### Periodic #######################
					# if shiftT[q] > 0
						# @inbounds t[1,end,cc] = x[end,1,cc,imIdx];
					# end
					# ix = ix = 1+shiftX[q];
					# for it = (1+shiftT[q]):upper 
						# @inbounds t[it,end,cc] = x[ix,1,cc,imIdx];
						# ix +=1;
					# end
					# if shiftT[q+1] < 0
						# @inbounds t[end,end,cc] = x[1,1,cc,imIdx];
					# end
					###################### End Periodic #######################
				end
			end
			if doDerivative == 0
				BLAS.gemm!('N','T',OneType,reshape(t,nImg1*nImg2,cin),K[k],OneType,y);
			else
				BLAS.gemm!('T','N',OneType,reshape(t,nImg1*nImg2,cin),y,OneType,K[k]);
			end
			k+=1;
		end
	end
	return y;
end
