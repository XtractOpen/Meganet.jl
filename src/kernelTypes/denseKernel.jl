export DenseKernel, getOp, getDenseKernel

mutable struct DenseKernel{T}
    nK
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

function Jthetamv(kernel::DenseKernel{T},dtheta::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T<:Number}
    nex    =  div(length(Y),nFeatIn(kernel))
    Y      = reshape(Y,:,nex)
    return  getOp(kernel,dtheta)*Y
end


function JthetaTmv(kernel::DenseKernel{T},Z::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T<:Number}
# Jacobian transpose matvec.
    nex    =  div(length(Y),nFeatIn(kernel))
    Y      = reshape(Y,:,nex)
    Z      = reshape(Z,:,nex)
    return  Z*Y'
end
