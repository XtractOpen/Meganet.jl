export DoubleSymLayer,getDoubleSymLayer

"""
 Implementation of symmetric double layer model

 Y(theta,Y0) = K(th1)'(activation( K(th1)\*Y0 + trafo.Bin\*th2))) + trafo.Bout\*th3
"""
mutable struct DoubleSymLayer{T} <: AbstractMeganetElement{T}
    activation  :: Function   # activation function
    K              # Kernel model, e.g., convMod
    nLayer      :: Union{NN{T}, normLayer{T}, AffineScalingLayer{T}}   # normalization layer
    Bin         :: Array{T}   # Bias inside the nonlinearity
    Bout        :: Array{T}   # bias outside the nonlinearity
end


function getDoubleSymLayer(TYPE::Type,K,nLayer::AbstractMeganetElement{T},
                           Bin=zeros(nFeatOut(K),0),Bout=zeros(nFeatIn(K),0),
                           activation=tanhActivation) where {T <: Number}
    BinT = convert.(T, Bin)
    BoutT = convert.(T, Bout)
    return DoubleSymLayer{TYPE}(activation,K,nLayer,Bin,Bout);

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

function apply(this::DoubleSymLayer{T},theta::Array{T},Y::Array{T},doDerivative=true)  where {T<:Number}

    #QZ = []
    tmp = Array{Any}(2) # TODO: Should this be type T?
    nex = div(length(Y),nFeatIn(this))
    Y   = reshape(Y,:,nex)

    theta1,theta2,theta3,th4 = splitWeights(this,theta)
    Kop    = getOp(this.K,theta1)
    KY     = Kop*Y

    KY,dummy,tmp[1] = apply(this.nLayer,th4,KY)

    Yt     = KY
    if !isempty(theta2)
     Yt .+= this.Bin*theta2
    end
    tmp[2] = copy(Yt)
    Z,      = this.activation(Yt,doDerivative)
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
    T(0.1)*ones(T,size(this.Bin,2),1);
    T(0.1)*ones(T,size(this.Bout,2),1);
    initTheta(this.nLayer)];
    return theta
end

function Jthetamv(this::DoubleSymLayer{T},dtheta::Array{T},theta::Array{T},Y::Array{T},tmp)  where {T<:Number}

    A,dA = this.activation(tmp[2],true)
    th1, th2,th3,th4    = splitWeights(this,theta)
    dth1,dth2,dth3,dth4 = splitWeights(this,dtheta)

	Kop    = getOp(this.K,th1)
    dKop   = getOp(this.K,dth1)
    dY     = dKop*Y

    dY = Jmv(this.nLayer,dth4,dY,th4,Kop*Y,copy(tmp[1]))[2]
    dY = dY .+ this.Bin*dth2

    dY = -(Kop'*(dA.*dY) + dKop'*A) .+ this.Bout*dth3
    return dY, dY
end

function JYmv(this::DoubleSymLayer{T},dY::Array{T},theta::Array{T},Y::Array{T},tmp)  where {T<:Number}

    dA = this.activation(tmp[2],true)[2]

    nex = div(length(dY),nFeatIn(this))
    dY  = reshape(dY,:,nex)
    Y   = reshape(Y,:,nex)
    th1, th2,th3,th4  = splitWeights(this,theta)

    Kop = getOp(this.K,th1)
    dY = Kop*dY
    dY = JYmv(this.nLayer,dY,th4,Kop*Y,copy(tmp[1]))[2]
    dZ = -(Kop'*(dA.*dY))
    return dZ, dZ
end

function Jmv(this::DoubleSymLayer{T},dtheta::Array{T},dY::Array{T},theta::Array{T},Y::Array{T},tmp)  where {T<:Number}
    A,dA = this.activation(copy(tmp[2]),true)
    nex = div(length(Y),nFeatIn(this))

    th1, th2,th3,th4    = splitWeights(this,theta)
    dth1,dth2,dth3,dth4 = splitWeights(this,dtheta)

    Kop    = getOp(this.K,th1)
    dKop   = getOp(this.K,dth1)
    if length(dY)>1
        dY  = reshape(dY,:,nex)
        KdY = Kop*dY
    else
        KdY = 0
    end
    dY = dKop*Y+KdY
    dY = Jmv(this.nLayer,dth4,dY,th4,Kop*Y,tmp[1])[2]

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


function JthetaTmv(this::DoubleSymLayer{T},Z::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp)  where {T<:Number}

    nex       = div(length(Y),nFeatIn(this))
    Z         = reshape(Z,:,nex)
    th1,th2,th3,th4  = splitWeights(this,theta)
    Kop       = getOp(this.K,th1)
    A,dA      = this.activation(tmp[2],true)

    dth3      = vec(sum(this.Bout'*Z,2))
    dAZ       = dA.*(Kop*Z)
    dth2      = vec(sum(this.Bin'*dAZ,2))

    dth4,dAZ  = JTmv(this.nLayer,dAZ,zeros(T,0),th4,Kop*Y,tmp[1])

    dth1      = JthetaTmv(this.K,A,zeros(T,0),Z)
    dth1     += JthetaTmv(this.K,dAZ,zeros(T,0),Y)
    dtheta    = [-vec(dth1); -vec(dth2); vec(dth3); -vec(dth4)]
    return dtheta
end

function JYTmv(this::DoubleSymLayer{T},Zin::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp)  where {T<:Number}

    nex       = div(length(Y),nFeatIn(this))
    Z         = reshape(Zin,:,nex)
    th1,th2,th3,th4  = splitWeights(this,theta)
    Kop       = getOp(this.K,th1)
    A,dA      = this.activation(tmp[2],true)

    dAZ       = dA.*(Kop*Z)
    dAZ       = JYTmv(this.nLayer,dAZ,(T)[],th4,Kop*Y,tmp[1])
    dAZ_out       = reshape(dAZ,:,nex)
    dY  = -(Kop'*dAZ_out)
    return dY
end

function JTmv(this::DoubleSymLayer{T}, Zin::Array{T}, dummy::Array{T},
                                            theta::Array{T}, Yin::Array{T}, tmp) where {T<:Number}

    nex::Int  = div(length(Yin),nFeatIn(this))
    Z         = reshape(Zin, :, nex)
    Yt        = reshape(tmp[2]::Array{T,2},:,nex)
    Y         = reshape(Yin,:,nex)
    th1, th2, th3, th4  = splitWeights(this,theta)
    Kop       = getOp(this.K,th1)::SparseMatrixCSC{T, Int64}
    A,dA    = this.activation(Yt,true)

    dth3      = vec(sum(this.Bout'*Z,2))
    dAZ       = dA.*(Kop*Z)
    dth2      = vec(sum(this.Bin'*dAZ,2))
    dth4,dAZ  = JTmv(this.nLayer,dAZ,zeros(T,0),th4,Kop*Y,tmp[1])

    dth1      = JthetaTmv(this.K,dAZ,zeros(T,0),Y)

    dth1      = dth1 + JthetaTmv(this.K,A,(T)[],Z)
    dtheta    = [-vec(dth1); -vec(dth2); vec(dth3);-vec(dth4)]

    dAZ_out = reshape(dAZ,:,nex)
    dY  = -(Kop'*dAZ_out)
    return dtheta,dY
end
