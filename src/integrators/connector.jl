export Connector,getConnector

mutable struct Connector{T} <: AbstractMeganetElement{T}
    K
    b
    outTimes
    Q
end

nTheta(this::Connector) = 0
nFeatIn(this::Connector) = size(this.K,2)
nFeatOut(this::Connector) = size(this.K,1)
nDataOut(this::Connector) = ((this.Q==I) ? nFeatOut(this) : size(this.Q,1))
initTheta(this::Connector{T}) where {T <: Number} = zeros(T,0)

function getConnector(TYPE::Type, K; b = zero(TYPE),outTimes=0,Q=I)
	return Connector{TYPE}(K,b,outTimes,Q);
end


function apply(this::Connector{T},theta::Array{T},Y0::Array{T},doDerivative=true) where {T <: Number}
    nex = div(length(Y0),nFeatIn(this))
    Y0  = reshape(Y0,:,nex)
    Y = this.K*Y0 .+ this.b
    if this.outTimes==1
        Ydata = this.Q*Y
    else
        Ydata = Array{T, 2}(0, 0)
    end
    tmp = Y0;
    return Ydata, Y, tmp
end

function Jmv(this::Connector{T},dtheta::Array{T},dY::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T <: Number}

    nex = div(length(dY),nFeatIn(this))
    dY  = reshape(dY,:,nex)
    dY = this.K*dY
    if this.outTimes==1
        dYdata = this.Q*dY
    else
        dYdata = []
    end
    return dYdata,dY
end

function JTmv(this::Connector{T},Wdata::Array{T},W::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T <: Number}
    nex = div(length(Y),nFeatIn(this))
    if length(W)==0
        W = zero(T);
    else
        W     = reshape(W,:,nex);
    end

    if length(Wdata)>0
        Wdata = reshape(Wdata,:,nex);
        W     = W+ this.Q'*Wdata;
    end

    dtheta = zeros(T,0);
    W   = this.K'*W;

    return dtheta,W
end
