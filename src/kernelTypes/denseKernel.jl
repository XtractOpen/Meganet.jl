export DenseKernel, getOp, getDenseKernel

mutable struct DenseKernel{T} <: AbstractConvKernel{T}
    nK :: Array{Int, 1}
end

function getDenseKernel(TYPE::Type,nK)
	return DenseKernel{TYPE}(copy(nK));
end

function nTheta(kernel::DenseKernel)
    return prod(kernel.nK)
end

function nFeatIn(kernel::DenseKernel)
    return kernel.nK[2]
end

function nFeatOut(kernel::DenseKernel)
    return kernel.nK[1]
end

function initTheta(kernel::DenseKernel{T}) where {T<:Number}
    return randn(T,nTheta(kernel))
end

function getOp(kernel::DenseKernel{T},theta::Array{T}) where {T<:Number}
    return reshape(theta,kernel.nK[1],kernel.nK[2])
end

function Amv(this::DenseKernel, theta::Array{T}, Y::Array{T}) where {T<:Number}
    K = getOp(this, theta)
    AY = K * Y
    return AY
end

function ATmv(this::DenseKernel, theta::Array{T}, Y::Array{T}) where {T<:Number}
    K = getOp(this, theta)
    AY = K' * Y
    return AY
end

function Jthetamv(kernel::DenseKernel{T},dtheta::Array{T},theta::Array{T},Yin::Array{T},tmp=nothing) where {T<:Number}
    nex    =  div(length(Yin),nFeatIn(kernel))
    Y      = reshape(Yin,:,nex)
    return  getOp(kernel,dtheta)*Y
end


function JthetaTmv(kernel::DenseKernel{T},Zin::Array{T},theta::Array{T},Yin::Array{T},tmp=nothing) where {T<:Number}
# Jacobian transpose matvec.
    nex    =  div(length(Yin),nFeatIn(kernel))
    Y      = reshape(Yin,:,nex)
    Z      = reshape(Zin,:,nex)
    return  Z*Y'
end
