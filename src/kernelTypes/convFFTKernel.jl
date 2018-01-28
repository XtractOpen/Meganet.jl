export convFFTKernel, getEigs,getConvFFTKernel
## For the functions nImgIn, nImgOut, nFeatIn, nFeatOut, nTheta, getOp, initTheta : see abstractConvKernel.jl
## All convKernel types are assumed to have fields nImage and sK
mutable struct convFFTKernel{T} <: abstractConvKernel{T}
    nImg :: Array{Int,1}
    sK   :: Array{Int,1}
    S    :: Array{Complex{T},2}
end

function getConvFFTKernel(TYPE::Type,nImg,sK)
	S = getEigs(Complex{TYPE},nImg,sK)
	return convFFTKernel{TYPE}(nImg,sK,S)
end

function getEigs(TYPE,nImg,sK)
    S = zeros(TYPE,prod(nImg),prod(sK[1:2]));
    for k=1:prod(sK[1:2])
        Kk = zeros(sK[1],sK[2]);
        Kk[k] = 1;
        Ak = getConvMatPeriodic(TYPE,Kk,[nImg[1],nImg[2], 1]);
		Akk = full(convert(Array{TYPE},Ak[:,1]));
        S[:,k] = vec(fft2(reshape(Akk,nImg[1],nImg[2]) ));
    end
    return S
end

export Amv
function Amv(this::convFFTKernel{T},theta::Array{T},Y::Array{T}) where {T<:Number}

    nex   = div(numel(Y),prod(nImgIn(this)))

    # compute convolution
    AY    = zeros(Complex{T},tuple([nImgOut(this);  nex]...));
    theta = reshape(theta, tuple([prod(this.sK[1:2]); this.sK[3:4]]...));
    Yh    = ifft2(reshape(Y,tuple([nImgIn(this); nex]...)));
    #### allocate stuff for the loop
    Sk = zeros(Complex{T},tuple(nImgOut(this)...))
    #T  = zeros(Complex{eltype(Y)},tuple(nImgOut(this)...))
    nn = nImgOut(this); nn[3] = 1;
    sumT = zeros(Complex{T},tuple([nn;nex]...))
    ####
       
    for k=1:this.sK[4]
		Sk = reshape(this.S*theta[:,:,k],tuple(nImgIn(this)...));
        #T  = Sk .* Yh;
        #sumT = sum(T,3)
        sumT = hadamardSum(sumT,Yh,Sk)
        AY[:,:,k,:]  = sumT[:,:,1,:];
    end
    AY = real(fft2(AY));
    Y  = reshape(AY,:,nex);
    return Y
end

function ATmv(this::convFFTKernel{T},theta::Array{T},Z::Array{T}) where {T<:Number}
    
    nex   =  div(numel(Z),prod(nImgOut(this)));
    ATY   = zeros(Complex{T},tuple([nImgIn(this); nex]...));
    theta = reshape(theta, prod(this.sK[1:2]),this.sK[3],this.sK[4]);
    #### allocate stuff for the loop
    Sk = zeros(Complex{T},tuple(nImgOut(this)...))
    #T  = zeros(Complex{eltype(Z)},tuple(nImgOut(this)...))
    nn = nImgOut(this); nn[3] = 1;
    sumT = zeros(Complex{T},tuple([nn;nex]...))
    ####
    
    Yh = fft2(reshape(Z,tuple([nImgOut(this); nex]...)));
    for k=1:this.sK[3]
        tk = theta[:,k,:]
        #if size(this.S,2) == 1
        #    tk = reshape(tk,1,:);
        #end
        Sk = reshape(this.S*tk,tuple(nImgOut(this)...));
        #T  = Sk.*Yh;
        #sumT = sum(T,3)
        sumT = hadamardSum(sumT,Yh,Sk)
        ATY[:,:,k] = sumT[:,:,1];
    end
    ATY = real(ifft2(ATY));
    ATY = reshape(ATY,:,nex);
    return ATY
end

function Jthetamv(this::convFFTKernel{T},dtheta::Array{T},dummy::Array{T},Y::Array{T},temp=nothing) where {T<:Number}

    nex    =  div(numel(Y),nFeatIn(this));
    Y      = reshape(Y,:,nex);
    Z      = Amv(this,dtheta,Y);
    return Z
end

function JthetaTmv(this::convFFTKernel{T},Z::Array{T},dummy::Array{T},Y::Array{T}) where {T<:Number}
    #  derivative of Z*(A(theta)*Y) w.r.t. theta

    nex  =  div(numel(Y),nFeatIn(this));

    dth1  = zeros(this.sK[1]*this.sK[2],this.sK[3],this.sK[4]);
    Y     = permutedims(reshape(Y,tuple([nImgIn(this); nex ]...)),[1 2 4 3]);
    Yh    = reshape(fft2(Y),prod(this.nImg[1:2]),nex*this.sK[3]);
    Zh    = permutedims(ifft2(reshape(Z,tuple([nImgOut(this); nex]...))),[1 2 4 3]);
    Zh    = reshape(Zh,:, this.sK[4]);

    for k=1:prod(this.sK[1:2])
        temp = conj(this.S[:,k]) .* Yh
        temp = reshape(temp,:,this.sK[3])
        dth1[k,:,:] = real(conj(temp)'*Zh);
    end

    dtheta = reshape(dth1,tuple(this.sK...));
    return dtheta
end

function hadamardSum(sumT::Array{T},Yh::Array{T},Sk::Array{T}) where {T<:Number}
    sumT .= 0.0;
    for i4 = 1:size(Yh,4)
        for i3 = 1:size(Yh,3)
            for i2 = 1:size(Yh,2)
                for i1 = 1:size(Yh,1)
                    @inbounds tt = Sk[i1,i2,i3]
                    @inbounds sumT[i1,i2,1,i4] += tt * Yh[i1,i2,i3,i4]
                end
            end
        end
    end        
    return sumT
end
