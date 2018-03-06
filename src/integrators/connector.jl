export Connector,getConnector

mutable struct Connector{T,TQ <: Union{Array{T,2},UniformScaling{Int}}, TK <: Union{Function, Array{T,2},SparseMatrixCSC{T,Int}}} <: AbstractMeganetElement{T}
    K::TK
    KT::Nullable{Function}
    b::T
    outTimes::Int
    Q::TQ # ???
    nFeatIn::Int
    nFeatOut::Int
end

nTheta(this::Connector) = 0
nFeatIn(this::Connector) = this.nFeatIn
nFeatOut(this::Connector) = this.nFeatOut
nDataOut(this::Connector) = ((this.Q==I) ? nFeatOut(this) : size(this.Q,1))
initTheta(this::Connector{T}) where {T <: Number} = zeros(T,0)

function getConnector(TYPE::Type, K::AbstractArray; b = zero(TYPE), outTimes=0, Q=I)
	return Connector(K, Nullable{Function}(), b, outTimes, Q, size(K,2), size(K,1))
end
function getConnector(TYPE::Type, K::Function, KT::Function, nFeatIn::Int, nFeatOut::Int; b = zero(TYPE), outTimes=0, Q=I)
	return Connector(K, Nullable{Function}(KT), b, outTimes, Q, nFeatIn, nFeatOut)
end

function apply(this::Connector{T, <:Any, <:Any},theta::Array{T},Y0::Array{T},tmp,doDerivative=true) where {T <: Number}
    nex = div(length(Y0),nFeatIn(this))
    Y0  = reshape(Y0,:,nex)

    if doDerivative
        if isempty(tmp)
            tmp = copy(Y0)
        else
            tmp .= Y0
        end
    end

    Y = this.K*Y0 .+ this.b
    Ydata::Array{T,2} = Array{T, 2}(0, 0) # Temporary fix until we know what type Q is
    if this.outTimes==1
        Ydata = this.Q*Y
    end

    return Ydata, Y, tmp
end

function apply(this::Connector{T, <:Any, <:Function},theta::Array{T},Y0::Array{T},tmp,doDerivative=true) where {T <: Number}
    nex = div(length(Y0),nFeatIn(this))
    Y0  = reshape(Y0,:,nex)

    if doDerivative
        if isempty(tmp)
            tmp = copy(Y0)
        else
            tmp .= Y0
        end
    end

    Y = this.K(Y0) .+ this.b
    Ydata::Array{T,2} = Array{T, 2}(0, 0) # Temporary fix until we know what type Q is
    if this.outTimes==1
        Ydata = this.Q*Y
    end

    return Ydata, Y, tmp
end

function Jmv(this::Connector{T, <:Any, <:Any},dtheta::Array{T},dY::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T <: Number}
    # ??? This doesn't seem to get used?
    nex = div(length(dY),nFeatIn(this))
    dY  = reshape(dY,:,nex)
    dY = this.K*dY
    dYdata::Array{T,2} = Array{T, 2}(0, 0) # Temporary fix until we know what type Q is
    if this.outTimes==1
        dYdata = this.Q*dY
    end

    return dYdata,dY
end

function Jmv(this::Connector{T, <:Any, <:Function},dtheta::Array{T},dY::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T <: Number}
    # ??? This doesn't seem to get used?
    nex = div(length(dY),nFeatIn(this))
    dY  = reshape(dY,:,nex)
    dYout = this.K(dY)
    dYdata::Array{T,2} = Array{T, 2}(0, 0) # Temporary fix until we know what type Q is
    if this.outTimes==1
        dYdata = this.Q*dYout
    end

    return dYdata,dYout
end

function JTmv(this::Connector{T, <:Any, <:Any},Wdata::Array{T},Win::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T <: Number}
    nex = div(length(Y),nFeatIn(this))
    if length(Win)==0
        W = zeros(T,1,1);
    else
        W = reshape(Win,:,nex);
    end

    if length(Wdata)>0
        Wdata = reshape(Wdata,:,nex);
        W = W .+ this.Q'*Wdata
    end

    dtheta = zeros(T,0);
    Wout   = this.K'*W;

    return dtheta,Wout
end

function JTmv(this::Connector{T, <:Any, <:Function},Wdata::Array{T},Win::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T <: Number}
    nex = div(length(Y),nFeatIn(this))
    if length(Win)==0
        W = zeros(T,1,1)
    else
        W = reshape(Win,:,nex)
    end

    if length(Wdata)>0
        Wdata = reshape(Wdata,:,nex)
        W = W .+ this.Q'*Wdata
    end

    dtheta = zeros(T,0)
    Wout   = get(this.KT)(W)

    return dtheta,Wout
end
