export normLayer, getBatchNormLayer, getTVNormLayer, getNormLayer

mutable struct normLayer{T} <: AbstractMeganetElement{T}
    nData   ::Array{Int,1}     # size of data #pixels x #channels x #examples
    doNorm  ::Int  # specifies dimensions along which to normalize
    eps		::T          # smoothing factor
    # normLayer(nData::Array{Int},doNorm=2,eps=convert(T,1e-3)) = new{Float32}(nData,doNorm,eps)
end

function getNormLayer(TYPE::Type, nData,doNorm,eps = convert(TYPE,1e-3))
    return  normLayer{TYPE}(nData,doNorm,eps)
end

function getBatchNormLayer(TYPE::Type, nData; eps = convert(TYPE,1e-3),isTrainable::Bool=true)

    L =  normLayer{TYPE}(nData,3,eps)
    if isTrainable
        SL = AffineScalingLayer{TYPE}(nData)
        temp_var = getbatchNormNN((L,SL))
        return temp_var
    else
        temp_var = L
        return temp_var
    end

end

function getTVNormLayer(TYPE::Type,nData;eps = convert(TYPE,1e-3),isTrainable::Bool=true)
    L =  normLayer{TYPE}(nData,2,eps)
    if isTrainable
        SL = AffineScalingLayer{TYPE}(nData)
        return getbatchNormNN((L,SL))
    else
        return L
    end
end

function apply(this::normLayer{T},theta::Array{T},Yin::Array{T,2},dA,doDerivative=true) where {T <: Number}

     # first organize Y with channels
    nf  = this.nData[2]::Int
    nex = div(length(Yin),nFeatIn(this))::Int
    Y = reshape(Yin,:,nf,nex)

    dA = Array{T,2}(0,0)

    # subtract mean across pixels
    m = mean(Y, this.doNorm)
    Y .-= m

    # normalize
    ep = this.eps
    mean!(x -> x^2, m, Y)
    m .= sqrt.(m .+ ep)
    Y .= Y ./ m

    Yout = reshape(Y,:,nex)
    return Yout, Yout, dA
end

function nTheta(this::normLayer)
    return 0
end

function nFeatIn(this::normLayer)
    return prod(this.nData[1:2]);
end

function nFeatOut(this::normLayer)
    return prod(this.nData[1:2]);
end

function nDataOut(this::normLayer)
    return nFeatOut(this)
end

function initTheta(this::normLayer{T}) where {T <: Number}
    return zeros(T,0)
end


function Jthetamv(this::normLayer,dtheta::Array{T},theta::Array{T},Y::Array{T},dA) where {T <: Number}
    return zeros(T,size(Y)), zeros(T,size(Y))
end

function JYmv(this::normLayer,dYin::Array{T},theta::Array{T},Yin::Array{T},dA=nothing) where {T <: Number}

    nex = div(length(dYin),nFeatIn(this))
    nf  = this.nData[2]
    dY   = reshape(dYin,:,nf,nex)
    Y    = reshape(Yin,:,nf,nex)

    Ya  = mean(Y,this.doNorm)
    Yout = Y .- Ya
    dYa = mean(dY,this.doNorm)
    dYout  = dY .- dYa

    S2y = mean(Yout.^2,this.doNorm);
    den = sqrt.(S2y+this.eps);
    tmp = mean(Yout.*dYout,this.doNorm)
    dYout ./= den

    den .^= 3
    Yout .= Yout.*tmp ./den

    dZ = reshape(dYout-Yout,:,nex)
    return dZ,dZ
end

function Jmv(this::normLayer{T},dtheta::Array{T},dY::Array{T},theta::Array{T},Y::Array{T},dA::Array{T}) where {T <: Number}
    return JYmv(this,dY,theta,Y,dA)
end

function  JTmv(this::normLayer{T},Z::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},dA) where {T <: Number}
    dtheta = zeros(T,0)
    dY     = JYTmv(this,Z,dummy,theta,Y,dA)
    return dtheta, dY
end

function JthetaTmv(this::normLayer{T},Z::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},dA) where {T <: Number}
    return zeros(T,0)
end

function JYTmv(this::normLayer{T},Zin::Array{T},dummy::Array{T},theta::Array{T},Yin::Array{T},dA=nothing) where {T <: Number}
    # Can overwrite Zin
    nex = div(length(Yin),nFeatIn(this))
    nf  = this.nData[2]
    Zout   = reshape(Zin,:,nf,nex)
    Y   = reshape(Yin,:,nf,nex)

    m = mean(Y, this.doNorm)
    Yout  = Y .- m
    mean!(m, Zout)
    Zout .= Zout .- m
    mean!(x -> x^2, m, Yout)
    den = sqrt.(m .+ this.eps)
    tmp = mean(Yout.*Zout,this.doNorm)

    # Combine into one broadcast
    # Zout .= Zout ./ den
    # Yout .= Yout .* tmp ./ den.^3
    # Zout .= Zout .- Yout <-- dY = Zout .- Yout
    Zout .= Zout./den .- Yout.*tmp./den.^3

    mean!(m, Zout)
    Zout .= Zout .- m
    return reshape(Zout,:,nex)
end
