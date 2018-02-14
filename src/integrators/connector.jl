export Connector,getConnector

mutable struct Connector{T,TQ <: Union{Array{T,2},UniformScaling{Int}}, TK <: Union{Array{T,2},SparseMatrixCSC{T,Int}}} <: AbstractMeganetElement{T}
    K::TK
    b::T
    outTimes::Int
    Q::TQ # ???
end

nTheta(this::Connector) = 0
nFeatIn(this::Connector) = size(this.K,2)
nFeatOut(this::Connector) = size(this.K,1)
nDataOut(this::Connector) = ((this.Q==I) ? nFeatOut(this) : size(this.Q,1))
initTheta(this::Connector{T}) where {T <: Number} = zeros(T,0)

function getConnector(TYPE::Type, K; b = zero(TYPE),outTimes=0,Q=I)
	return Connector(K,b,outTimes,Q);
end


function apply(this::Connector{T},theta::Array{T},Y0::Array{T},doDerivative=true) where {T <: Number}
    # tic()
    nex = div(length(Y0),nFeatIn(this))
    Y0  = reshape(Y0,:,nex)
    Y = this.K*Y0 .+ this.b
    Ydata::Array{T,2} = Array{T, 2}(0, 0) # Temporary fix until we know what type Q is
    if this.outTimes==1
        Ydata = this.Q*Y
    end
    tmp = Y0;
    # println("Measuring time for connector ",toc())
    return Ydata, Y, tmp
end

function Jmv(this::Connector{T},dtheta::Array{T},dY::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T <: Number}
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

function JTmv(this::Connector{T},Wdata::Array{T},Win::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T <: Number}
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
    W   = this.K'*W;

    return dtheta,W
end
