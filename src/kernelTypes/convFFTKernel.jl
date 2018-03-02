export convFFTKernel, getConvFFTKernel
## For the functions nImgIn, nImgOut, nFeatIn, nFeatOut, nTheta, getOp, initTheta : see AbstractConvKernel.jl
## All convKernel types are assumed to have fields nImage and sK
mutable struct convFFTKernel{T} <: AbstractConvKernel{T}
    nImg :: Array{Int,1}
    sK   :: Array{Int,1}
	Kp   :: Array{T}
	I    :: Array{Int}
end

function getConvFFTKernel(TYPE::Type,nImg,sK)
	return convFFTKernel{TYPE}(nImg,sK,(TYPE)[],(Int64)[])
end

function getKp(this::convFFTKernel{T}) where {T<:Number}
	# setup the padded convolution kernel and get the indices of non-zeros
	if isempty(this.Kp)
		theta = reshape(T.(collect(1:prod(this.sK))),this.sK[1],this.sK[2],prod(this.sK[3:4]))
		Kp    = zeros(T,this.nImg[1],this.nImg[2],size(theta,3))
		Kp[1:this.sK[1],1:this.sK[2],:] = theta;
		center = (this.sK[1:2]+1)./2
		Kp    = circshift(Kp,1-center);
		I = find(Kp)
		this.Kp = zero(T)*Kp;
		idp     = sortperm(Kp[I])
		this.I  = I[idp];
	end
	return this.Kp,this.I
end
function getK1(this::convFFTKernel{T},theta::Array{T}) where {T<:Number}
	# get first columns of convolultion matrices
	Kp,I = getKp(this)
	for k=1:length(I)
		Kp[I[k]]  = theta[k]
	end
	return Kp
end

# methods for A*x
function multRed!(Zkh::Array{Complex{T},2},S::Array{Complex{T},4},Yh::Array{Complex{T},3},k::Int) where {T<:Number}
	# compute  Zkh[i1,i2,k] = S[i1,i2,:,k]'*Yh[i1,i2,:]
	Zkh[:]=Complex128(0.0)
	for i3=1:size(Yh,3)
		for i2=1:size(Zkh,2)
			for i1=1:size(Zkh,1)
				@inbounds Zkh[i1,i2] += S[i1,i2,i3,k].*Yh[i1,i2,i3]
			end
		end
	end
end

function Amv!(this::convFFTKernel{T},Z::AbstractArray{T,3},S::Array{Complex{T}},Y::AbstractArray{T,3},Yh::Array{Complex{T},3},Zkh::Array{Complex{T},2}) where {T<:Number}
	# 2D convolution for a single example. 
	Yh[:]=Y; ifft2!(Yh)
    for k=1:this.sK[4]
		multRed!(Zkh,S,Yh,k)
        Z[:,:,k]  = real(fft2!(Zkh))
    end
	return Z
end
function Amv(this::convFFTKernel{T},theta::Array{T},Y::Array{T}) where {T<:Number}
		nex   = div(length(Y),prod(nImgIn(this)))
		Z = zeros(T,tuple([nImgOut(this); nex]...))
	    Amv!(this,Z,theta,Y)
		return Z
end
function Amv!(this::convFFTKernel{T},Z::Array{T},theta::Array{T},Y::Array{T}) where {T<:Number}
    nex   = div(length(Y),prod(nImgIn(this)))
	Y     = reshape(Y,tuple([nImgIn(this);nex]...))
	# pre-allocation for temps
	Ykh = zeros(Complex{T},this.nImg[1],this.nImg[2],this.sK[3])
	Zk  = zeros(T, this.nImg[1],this.nImg[2], this.sK[4])
    Zik = zeros(Complex{T}, this.nImg[1],this.nImg[2])
	
	# get kernel
	S = reshape( fft2(getK1(this,theta)), this.nImg[1],this.nImg[2], this.sK[3], this.sK[4])

    # compute convolution
    for k=1:nex
		Amv!(this,view(Z,:,:,:,k),S,view(Y,:,:,:,k),Ykh,Zik)
	end
	Z = reshape(Z,:,nex)
	return Z
end

# methods for A'*x
function multRedT!(Ykh::Array{Complex{T},2},S::Array{Complex{T},4},Zh::Array{Complex{T},3},j::Int) where {T<:Number}
	# compute  Ykh[i1,i2,k] = S[i1,i2,j,:]'*Zh[i1,i2,:]
	Ykh[:]=Complex{T}(0.0)
	for i3=1:size(Zh,3)
		for i2=1:size(Ykh,2)
			for i1=1:size(Ykh,1)
				@inbounds Ykh[i1,i2] += S[i1,i2,j,i3].*Zh[i1,i2,i3]
			end
		end
	end
	return Ykh
end

function ATmv!(this::convFFTKernel{T},Y::AbstractArray{T,3},S::Array{Complex{T}},Z::AbstractArray{T,3},Zh::Array{Complex{T},3},Ykh::Array{Complex{T},2}) where {T<:Number}
	# 2D convolution for a single example. 
	Zh[:]=Z; Zh =fft2!(Zh)
    for j=1:this.sK[3]
		multRedT!(Ykh,S,Zh,j)
        Y[:,:,j]  = real(ifft2!(Ykh))
    end
end
function ATmv(this::convFFTKernel{T},theta::Array{T},Z::Array{T}) where {T<:Number}
		nex = div(length(Z),prod(nImgOut(this)))
		Y   = zeros(T,tuple([nImgIn(this); nex]...))
	    ATmv!(this,Y,theta,Z)
		return Y
end

function ATmv!(this::convFFTKernel{T},Y::Array{T},theta::Array{T},Z::Array{T}) where {T<:Number}
    nex   = div(length(Y),prod(nImgIn(this)))
	Z     = reshape(Z,tuple([nImgOut(this);nex]...))
	
	# pre-allocation for temps
	Zkh = zeros(Complex{T},this.nImg[1],this.nImg[2],this.sK[4])
	Yk  = zeros(T, this.nImg[1],this.nImg[2], this.sK[3])
    Yik = zeros(Complex{T}, this.nImg[1],this.nImg[2])
	
	# get kernel
	S = reshape( fft2(getK1(this,theta)), this.nImg[1],this.nImg[2], this.sK[3], this.sK[4])

    # compute convolution
    for k=1:nex
		ATmv!(this,view(Y,:,:,:,k),S,view(Z,:,:,:,k),Zkh,Yik)
	end
	Y = reshape(Y,:,nex)
	return Y
end

function Jthetamv(this::convFFTKernel{T},dtheta::Array{T},dummy::Array{T},Y::Array{T},temp=nothing) where {T<:Number}
    return Amv(this,dtheta,Y)
end

function JthetaTmv(this::convFFTKernel{T},Z::Array{T},dummy::Array{T},Y::Array{T}) where {T<:Number}
	nex    =  div(length(Y),nFeatIn(this))
	
	Y    = reshape(Y,this.nImg[1],this.nImg[2], this.sK[3],nex)
	Z    = reshape(Z,this.nImg[1],this.nImg[2], this.sK[4],nex)
	
	# temps 
	Yh   = zeros(Complex{T},this.nImg[1],this.nImg[2],this.sK[3])
	Zh   = zeros(Complex{T},this.nImg[1],this.nImg[2],this.sK[4])
	tt   = zeros(Complex{T},this.nImg[1],this.nImg[2],this.sK[3],this.sK[4])
	
	for k=1:nex
		Yh[:] = Y[:,:,:,k]
		Zh[:] = Z[:,:,:,k]
		
		ifft2!(Yh)
		fft2!(Zh)
		
		for i4=1:this.sK[4]
			for i3=1:this.sK[3]
				for i2=1:this.nImg[2]
					for i1=1:this.nImg[1]
						@inbounds tt[i1,i2,i3,i4] += Yh[i1,i2,i3].*Zh[i1,i2,i4]
					end
				end
			end
		end
	end
	tt = real(fft2(reshape(tt,this.nImg[1],this.nImg[2],:)))
	return tt[this.I]
end
