export DoubleSymLayer,getDoubleSymLayer

"""
 Implementation of symmetric double layer model

 Y(theta,Y0) = K(th1)'(activation( K(th1)\*Y0 + trafo.Bin\*th2))) + trafo.Bout\*th3
"""
mutable struct DoubleSymLayer{T, TK <: AbstractConvKernel{T}, TN <: Union{batchNormNN{T}, normLayer{T}}} <: AbstractMeganetElement{T}
    activation  :: Function   # activation function
    activation! :: Function # in place activation function
    K           :: TK   # Kernel model, e.g., convMod
    nLayer      :: TN   # normalization layer
    Bin         :: Array{T}   # Bias inside the nonlinearity
    Bout        :: Array{T}   # bias outside the nonlinearity
end


function getDoubleSymLayer(TYPE::Type,K,nLayer::AbstractMeganetElement{T};
                           Bin=zeros(nFeatOut(K),0),Bout=zeros(nFeatIn(K),0),
                           activation=tanhActivation,activation_inplace=tanhActivation!) where {T <: Number}
    BinT = convert.(T, Bin)
    BoutT = convert.(T, Bout)
    return DoubleSymLayer(activation,activation_inplace,K,nLayer,BinT,BoutT);

end

function splitWeights(this::DoubleSymLayer{T},theta::Array{T}) where {T<:Number}

    th1 = theta[1:nTheta(this.K)::Int]
    cnt = length(th1)
    th2 = theta[cnt+(1:size(this.Bin,2))]
    cnt = cnt + length(th2)
    th3 = theta[cnt+(1:size(this.Bout,2))]
    cnt = cnt + length(th3)

    th4 = theta[cnt+1:end];

    return th1, th2, th3, th4
end


function apply(this::DoubleSymLayer{T},theta::Array{T},Yin::Array{T,2},tmp,doDerivative=true) where {T<:Number}

    nex = div(length(Yin),nFeatIn(this))::Int
    Y   = reshape(Yin,:,nex)

    theta1,theta2,theta3,theta4 = splitWeights(this,theta)
    Kop    = getOp(this.K,theta1)
    KY     = Kop*Y
    if doDerivative
        if isempty(tmp)
            tmp = copy(KY)
        else
            tmp .= KY
        end
    end
    KY, = apply(this.nLayer,theta4,KY,[],false)

    if !isempty(theta2)
     KY .+= this.Bin*theta2
    end

    Z::Array{T,2},      = this.activation!(KY,[],false)

    Z      = -(Kop'*Z)
    if !isempty(theta3)
        Z  .+= this.Bout*theta3
    end
    return Z, Z, tmp
end

function nTheta(this::DoubleSymLayer)
    return nTheta(this.K) + size(this.Bin,2)+ size(this.Bout,2) + nTheta(this.nLayer)
end

function nFeatIn(this::DoubleSymLayer)
    return nFeatIn(this.K)
end

function nFeatOut(this::DoubleSymLayer)
    return nFeatIn(this.K)
end

function nDataOut(this::DoubleSymLayer)
    return nFeatIn(this)
end

function initTheta(this::DoubleSymLayer{T})  where {T<:Number}
    theta = [vec(initTheta(this.K));
    T(0.01)*ones(T,size(this.Bin,2),1);
    T(0.01)*ones(T,size(this.Bout,2),1);
    initTheta(this.nLayer)];
    return theta
end

function Jthetamv(this::DoubleSymLayer{T},dtheta::Array{T},theta::Array{T},Y::Array{T},tmp::Array{T})  where {T<:Number}

    th1, th2,th3,th4    = splitWeights(this,theta)
    dth1,dth2,dth3,dth4 = splitWeights(this,dtheta)

	# re-compute derivative of activation
	KY              = copy(tmp);
    KY,dummy,tmpNL  = apply(this.nLayer,th4,KY,[],true)
    if !isempty(th2)
     KY .+= this.Bin*th2
    end
    A,dA   = this.activation(KY,true)


	Kop    = getOp(this.K,th1)
    dKop   = getOp(this.K,dth1)
    dY     = dKop*Y

    dY   = Jmv(this.nLayer,dth4,dY,th4,tmp,tmpNL)[2]
	if !isempty(dth2)
    	dY .+= this.Bin*dth2
	end
    dY = -(Kop'*(dA.*dY) + dKop'*A)
	if !isempty(dth3)
		dY .+= this.Bout*dth3
	end
    return dY, dY
end

function JYmv(this::DoubleSymLayer{T},dY::Array{T},theta::Array{T},Y::Array{T},tmp::Array{T})  where {T<:Number}

    nex = div(length(dY),nFeatIn(this))
    dY  = reshape(dY,:,nex)
    Y   = reshape(Y,:,nex)
    th1, th2,th3,th4  = splitWeights(this,theta)

	# re-compute derivative of activation
	KY              = copy(tmp);
    KY,dummy,tmpNL  = apply(this.nLayer,th4,KY,[],true)
    if !isempty(th2)
     KY .+= this.Bin*th2
    end
    A,dA   = this.activation(KY,true)


    Kop = getOp(this.K,th1)
    dY = Kop*dY
    dY = JYmv(this.nLayer,dY,th4,tmp,tmpNL)[2]
    dZ = -(Kop'*(dA.*dY))
    return dZ, dZ
end

function Jmv(this::DoubleSymLayer{T},dtheta::Array{T},dY::Array{T},theta::Array{T},Y::Array{T},tmp::Array{T})  where {T<:Number}
    nex = div(length(Y),nFeatIn(this))

    th1, th2,th3,th4    = splitWeights(this,theta)
    dth1,dth2,dth3,dth4 = splitWeights(this,dtheta)

	# re-compute derivative of activation
	KY              = copy(tmp);
    KY,dummy,tmpNL  = apply(this.nLayer,th4,KY,[],true)
    if !isempty(th2)
     KY .+= this.Bin*th2
    end
    A,dA   = this.activation(KY,true)

	Kop    = getOp(this.K,th1)
    dKop   = getOp(this.K,dth1)
    if length(dY)>1
        dY  = reshape(dY,:,nex)
        KdY = Kop*dY
    else
        KdY = 0
    end
    dY = dKop*Y+KdY
    dY = Jmv(this.nLayer,dth4,dY,th4,tmp,tmpNL)[2]

    dY = reshape(dY,:,nex)
    if !isempty(dth2)
        dY .+= this.Bin*dth2
    end

    dY = -(Kop'*(dA.*dY) + dKop'*A)
    if !isempty(dth3)
        dth3 .+= this.Bout*dth3
    end

    return dY, dY
end


function JthetaTmv(this::DoubleSymLayer{T},Z::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp::Array{T})  where {T<:Number}

    nex       = div(length(Y),nFeatIn(this))
    Z         = reshape(Z,:,nex)
    th1,th2,th3,th4  = splitWeights(this,theta)
    Kop       = getOp(this.K,th1)

	# re-compute derivative of activation
	KY              = copy(tmp);
    KY,dummy,tmpNL  = apply(this.nLayer,th4,KY,[],true)
    if !isempty(th2)
     KY .+= this.Bin*th2
    end
    A,dA   = this.activation(KY,true)

	dth3      = vec(sum(this.Bout'*Z,2))
    dAZ       = dA.*(Kop*Z)
    dth2      = vec(sum(this.Bin'*dAZ,2))

    dth4,dAZ  = JTmv(this.nLayer,dAZ,zeros(T,0),th4,tmp,tmpNL)

    dth1      = JthetaTmv(this.K,A,zeros(T,0),Z)
    dth1     += JthetaTmv(this.K,dAZ,zeros(T,0),Y)
    dtheta    = [-vec(dth1); -vec(dth2); vec(dth3); -vec(dth4)]
    return dtheta
end

function JYTmv(this::DoubleSymLayer{T},Zin::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp::Array{T})  where {T<:Number}

    nex       = div(length(Y),nFeatIn(this))
    Z         = reshape(Zin,:,nex)
    th1,th2,th3,th4  = splitWeights(this,theta)
    Kop       = getOp(this.K,th1)

	# re-compute derivative of activation
	KY              = copy(tmp);
    KY,dummy,tmpNL  = apply(this.nLayer,th4,KY,[],true)
    if !isempty(th2)
     KY .+= this.Bin*th2
    end
    A,dA   = this.activation(KY,true)



    dAZ       = dA.*(Kop*Z)
    dAZ       = JYTmv(this.nLayer,dAZ,(T)[],th4,tmp,tmpNL)
    dAZ_out   = reshape(dAZ,:,nex)
    dY  = -(Kop'*dAZ_out)
    return dY
end

function JTmv(this::DoubleSymLayer{T}, Zin::Array{T}, dummy::Array{T},
                                            theta::Array{T}, Yin::Array{T}, tmp::Array{T}) where {T<:Number}

    nex  = div(length(Yin),nFeatIn(this))
    Z         = reshape(Zin, :, nex)
    Y         = reshape(Yin,:,nex)
    th1, th2, th3, th4  = splitWeights(this,theta)
    #Kop       = getOp(this.K,th1)

	# re-compute derivative of activation
	KY              = copy(tmp);
    KY,dummy,tmpNL  = apply(this.nLayer,th4,KY,[],true)
    if !isempty(th2)
     KY .+= this.Bin*th2
    end
    A,dA   = this.activation(KY,true)


    dth3      = vec(sum(this.Bout'*Z,2))

    KopZ = Amv(this.K, th1, Z)
    dAZ1       = dA.*KopZ

    dth2      = vec(sum(this.Bin'*dAZ1,2))
    dth4, dAZ2  = JTmv(this.nLayer,dAZ1,zeros(T,0),th4,tmp,tmpNL)
    dth1      = JthetaTmv(this.K,dAZ2,zeros(T,0),Y)
    dth1 += JthetaTmv(this.K,A,(T)[],Z)
    dtheta    = [-vec(dth1); -vec(dth2); vec(dth3);-vec(dth4)]

    dAZ_out = reshape(dAZ2,:,nex)
    KopTdAZ = ATmv(this.K, th1, dAZ_out)
    dY  = -KopTdAZ
    return dtheta, dY
end
