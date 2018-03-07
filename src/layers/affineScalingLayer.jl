export AffineScalingLayer, getAffineScalingLayer

"""
Scales and shifts the 3D feature tensor along each dimension. This is
useful, e.g., in batch normalization.

kron(s3,kron(s2,s1)) * vec(Y) + kron(b3,kron(e2,e1)) +
kron(e3,kron(b2,e1)) + kron(e3,kron(e2,b1));
"""
mutable struct AffineScalingLayer{T} <: AbstractMeganetElement{T}
    nData  :: Array{Int,1}     # describe size of data, at least first two dim must be correct.
end
function getAffineScalingLayer(TYPE::Type, nData)
    return AffineScalingLayer{TYPE}(nData)
end

function splitWeights(this::AffineScalingLayer{T},theta_in::Array{T}) where {T <: Number}
    theta = reshape(theta_in,:,2)
    s2    = theta[:,1]
    b2    = theta[:,2]
    return s2, b2
end

function scaleChannels!(Y::Array{T},s::Array{T},b::Array{T}) where {T <: Number}
    for i=1:length(s)
        Y[:,i,:] .= Y[:,i,:].*s[i] .+ b[i]
    end
end

function apply(this::AffineScalingLayer{T},theta::Array{T},Yin::Array{T},dA,doDerivative=false) where {T <: Number}

    Y   = reshape(Yin,this.nData[1], this.nData[2],:)
    dA  = Array{T,2}(0,0)
    nex = size(Y,3)

    s2,b2 = splitWeights(this,theta);

    s = reshape(s2, 1, :, 1)
    b = reshape(b2, 1, :, 1)
    # scaleChannels!(Y,s2,b2);
    Y .= Y .* s .+ b

    Yout = reshape(Y,:,nex)
    Ydata = Yout
    return Ydata, Yout, dA
end

function nTheta(this::AffineScalingLayer)
    return 2*this.nData[2]
end

function nFeatIn(this::AffineScalingLayer)
    return prod(this.nData[1:2])
end

function nFeatOut(this::AffineScalingLayer)
   return prod(this.nData[1:2])
end

function nDataOut(this::AffineScalingLayer)
    return nFeatOut(this);
end

function initTheta(this::AffineScalingLayer{T}) where {T <: Number}
    s2,b2 = splitWeights(this,ones(T,nTheta(this)))
    return  [vec(s2); convert(T,0.)*vec(b2)]
end

function Jthetamv(this::AffineScalingLayer{T},dtheta::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T <: Number}
    Y   = reshape(copy(Y),this.nData[1], this.nData[2],:)
    nex = size(Y,3)

    ds2,db2 = splitWeights(this,dtheta)

    scaleChannels!(Y,ds2,db2)

    dY = reshape(Y,:,nex)
    dYdata = dY
    return dYdata, dY
end

function JthetaTmv(this::AffineScalingLayer{T},Z::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T <: Number}
    Y   = reshape(Y,this.nData[1], this.nData[2],:)
    Z   = reshape(Z,this.nData[1], this.nData[2],:)

    W = Y.*Z

    dtheta = vec(sum(sum(W,1),3))
    return  [dtheta; vec(sum(sum(Z,1),3))]
end

function JYmv(this::AffineScalingLayer{T},dY::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T <: Number}
    dY   = reshape(copy(dY),this.nData[1], this.nData[2],:);
    nex = size(dY,3)

    s2,b2 = splitWeights(this,theta);
    scaleChannels!(dY,s2,b2*0)

    dY     = reshape(dY,:,nex)
    dYdata = dY
    return dYdata, dY
end

function JYTmv(this::AffineScalingLayer{T},Z::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp=nothing) where {T <: Number}
    Z   = reshape(copy(Z),this.nData[1], this.nData[2],:)
    nex = size(Z,3)

    s2,b2 = splitWeights(this,theta)
    scaleChannels!(Z,s2,b2*0)
    return reshape(Z,:,nex)
end
