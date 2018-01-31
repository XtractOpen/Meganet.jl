export singleLayer,getSingleLayer

mutable struct singleLayer{T} <: AbstractMeganetElement{T}
        activation :: Function # activation function
        K           # transformation type
        nLayer     :: Union{NN{T}, normLayer{T}, AffineScalingLayer{T}} # normalization layer
        Bin        :: Array{T} # bias inside nonlinearity
        Bout       :: Array{T} # bias outside nonlinearity
        # singleLayer{T}(K,nLayer;Bin=zeros(T,nFeatOut(K),0),Bout=zeros(T,nFeatOut(K),0),activation=tanhActivation) =

end

function getSingleLayer(TYPE::Type, K,nLayer;Bin=zeros(TYPE,nFeatOut(K),0),Bout=zeros(TYPE,nFeatOut(K),0),activation=tanhActivation)
	singleLayer{TYPE}(activation,K,nLayer,Bin,Bout);
end


function splitWeights(this::singleLayer{T},theta::Array{T}) where {T <: Number}
    th1 = theta[1:nTheta(this.K)]
    cnt = length(th1)
    th2 = theta[cnt+(1:size(this.Bin,2))]
    cnt += length(th2)
    th3 = theta[cnt+(1:size(this.Bout,2))]
    cnt += length(th3)
    th4 = theta[cnt+(1:nTheta(this.nLayer))]
    cnt += length(th4)
    if cnt!=length(theta); error("splitWeights: length does not match"); end
    return th1, th2, th3, th4
end

function apply(this::singleLayer{T},theta::Array{T},Y::Array{T},doDerivative=false,tmp=[]) where {T <: Number}
    if isempty(tmp)
        tmp = Array{Any}(2)
    end
    nex = div(length(Y),nFeatIn(this))
    Y   = reshape(Y,:,nex)
    th1,th2,th3,th4 = splitWeights(this,theta)
    Y     =  getOp(this.K,th1)*Y .+ this.Bin * th2
    Y,dummy,tmp[1] = apply(this.nLayer,th4,Y,doDerivative)
    Y,tmp[2]  = this.activation(Y,doDerivative)
    Y     = Y .+ this.Bout*th3
    Ydata = Y
    return Ydata, Y, tmp
end

function nTheta(this::singleLayer)
    return nTheta(this.K)+size(this.Bin,2) + size(this.Bout,2) + nTheta(this.nLayer)
end

function nFeatIn(this::singleLayer)
    return nFeatIn(this.K)
end

function nFeatOut(this::singleLayer)
    return nFeatOut(this.K)
end

function nDataOut(this::singleLayer)
    return nFeatOut(this.K)
end

function initTheta(this::singleLayer{T}) where {T <: Number}
    return [vec(initTheta(this.K)); convert(T,0.1)*ones(T,size(this.Bin,2),1) ; convert(T,0.1)*ones(T,size(this.Bout,2),1); initTheta(this.nLayer) ]
end


function Jthetamv(this::singleLayer{T},dtheta::Array{T},theta::Array{T},Y::Array{T},tmp) where {T <: Number}
    dA             = tmp[2]
    nex            = div(length(Y),nFeatIn(this))
    Y              = reshape(Y,:,nex)

    th1,th2,th3,th4     = splitWeights(this,theta)
    dth1,dth2,dth3,dth4 = splitWeights(this,dtheta)
    dZ = Jthetamv(this.K,dth1,th1,Y) .+ this.Bin*dth2
    Kop = getOp(this.K,th1)
    dZ  = Jmv(this.nLayer,dth4,dZ,th4,Kop*Y.+this.Bin*th2,tmp[1])[2]
    dZ = dA.*dZ .+ this.Bout*dth3;
    return dZ, dZ
end

function JYmv(this::singleLayer{T},dY::Array{T},theta::Array{T},Y::Array{T},tmp) where {T <: Number}
    dA  = tmp[2]
    nex = div(length(dY),nFeatIn(this))
    th1,th2,th3,th4 = splitWeights(this,theta)
    Kop         = getOp(this.K,th1)
    dY          = reshape(dY,:,nex)
    dZ          = Kop*dY
    dZ          = JYmv(this.nLayer,dZ,th4,Kop*Y.+this.Bin*th2,tmp[1])[2]
    dZ          = dA.*dZ
    return dZ,dZ
end

function Jmv(this::singleLayer{T},dtheta::Array{T},dY::Array{T},theta::Array{T},Y::Array{T},tmp) where {T <: Number}
    dA  = tmp[2]
    nex = div(length(Y),nFeatIn(this))
    th1,th2,th3,th4 = splitWeights(this,theta)
    dth1,dth2,dth3,dth4 = splitWeights(this,dtheta)

    dY = reshape(dY,:,nex);
    Kop = getOp(this.K,th1)
    dZ = Kop*dY;

    Y   = reshape(Y,:,nex);
    dZ += Jthetamv(this.K,dth1,th1,Y) .+ this.Bin*dth2
    dZ = Jmv(this.nLayer,dth4,dZ,th4,Kop*Y.+this.Bin*th2,tmp[1])[2]

    dZ = dA.*dZ .+ this.Bout*dth3
    return dZ,dZ
end

function JTmv(this::singleLayer{T},Z::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp) where {T <: Number}
    dA   = tmp[2]
    nex  = div(length(Y),nFeatIn(this))
    Z    = reshape(Z,:,nex)
    th1,th2,th3,th4  = splitWeights(this,theta)
    Kop = getOp(this.K,th1)

    dth3      = vec(sum(this.Bout'*Z,2))
    dAZ       = dA.*Z
    dth4,dAZ  = JTmv(this.nLayer,dAZ,zeros(T,0),th4,Kop*Y.+this.Bin*th2,tmp[1])
    dth2      = vec(sum(this.Bin'*reshape(dAZ,:,nex),2))
    dth1      = JthetaTmv(this.K, dAZ,theta,Y)

    dY   = Kop'*reshape(dAZ,:,nex)
    dtheta = [vec(dth1); vec(dth2); vec(dth3); vec(dth4)]

    return dtheta, dY

end

function JthetaTmv(this::singleLayer{T},Z::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp) where {T <: Number}
    dA        = tmp[2]
    nex       = div(length(Z),nFeatOut(this))
    th1,th2,th3,th4  = splitWeights(this,theta)

    Z         = reshape(Z,:,nex);
    dAZ       = dA.*Z;
    dth3      = vec(sum(this.Bout'*Z,2));
    Kop       = getOp(this.K,th1)
    dth4,dAZ  = JTmv(this.nLayer,dAZ,(T)[],th4,Kop*Y.+this.Bin*th2,tmp[1])
    dth1      = JthetaTmv(this.K,dAZ,theta,Y);
    dth2      = vec(sum(this.Bin'*reshape(dAZ,:,nex),2));
    return [vec(dth1); vec(dth2); vec(dth3); vec(dth4)];
end

function JYTmv(this::singleLayer{T},Z::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp) where {T <: Number}
    dA   = tmp[2]
    nex  = div(length(Y),nFeatIn(this))
    th1,th2,th3,th4 = splitWeights(this,theta)
    Kop = getOp(this.K,th1)
    Z    = reshape(Z,:,nex)
    dAZ  = dA.*Z
    dAZ = JYTmv(this.nLayer,dAZ,(T)[],th4,Kop*Y.+this.Bin*th2,tmp[1])
    return Kop'*reshape(dAZ,:,nex)
end
