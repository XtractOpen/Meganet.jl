export normLayer, getBatchNormLayer, getTVNormLayer

type normLayer
    nData        # size of data #pixels x #channels x #examples
    doNorm::Int  # specifies dimensions along which to normalize
    eps          # smoothing factor
    normLayer(nData::Array{Int},doNorm=2,eps=1e-3) = new(nData,doNorm,eps)
end

function getBatchNormLayer(nData;eps=1e-3,isTrainable::Bool=true)
    L =  normLayer(nData,3,eps)
    if isTrainable
        SL = AffineScalingLayer(nData)
        return NN([L;SL])
    else
        return L
    end
end

function getTVNormLayer(nData;eps=1e-3,isTrainable::Bool=true)
    L =  normLayer(nData,2,eps)
    if isTrainable
        SL = AffineScalingLayer(nData)
        return NN([L;SL])
    else
        return L
    end
end

function apply(this::normLayer,theta,Y,doDerivative=true)

    # first organize Y with channels
    nf  = this.nData[2]
    nex = div(length(Y),nFeatIn(this))
    Y = reshape(Y,:,nf,nex)

    dA = []

    # subtract mean across pixels
    Ya = mean(Y,this.doNorm)
    Y  = Y.-Ya
    # normalize
    S2 = mean(Y.^2,this.doNorm)
    Y  = Y ./ sqrt.(S2+this.eps)

    Y = reshape(Y,:,nex)

    return Y, Y, dA
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

function initTheta(this::normLayer)
    return zeros(0)
end


function Jthetamv(this::normLayer,dtheta,theta,Y,dA)
    return zeros(size(Y)), zeros(size(Y))
end

function JYmv(this::normLayer,dY,theta,Y,dA=nothing)

    nex = div(length(dY),nFeatIn(this))
    nf  = this.nData[2]
    dY   = reshape(dY,:,nf,nex)
    Y    = reshape(Y,:,nf,nex)

    Ya  = mean(Y,this.doNorm)
    Y   = Y .- Ya
    dYa = mean(dY,this.doNorm)
    dY  = dY .- dYa
    S2y = mean(Y.^2,this.doNorm);
    den = sqrt.(S2y+this.eps);

    tmp = mean(Y.*dY,this.doNorm)
    dY  = dY ./ den

    Y   = Y .* tmp
    Y  =  Y ./ den.^3
    dZ = reshape(dY-Y,:,nex)
    return dZ,dZ
end

function Jmv(this::normLayer,dtheta,dY,theta,Y,dA)
    return JYmv(this,dY,theta,Y,dA)
end

function  JTmv(this::normLayer,Z,dummy,theta,Y,dA)
    dtheta = zeros(0)
    dY     = JYTmv(this,Z,dummy,theta,Y,dA)
    return dtheta, dY
end

function JthetaTmv(this::normLayer,Z,dummy,theta,Y,dA)
    return zeros(0)
end

function JYTmv(this::normLayer,Z,dummy,theta,Y,dA=nothing)

    nex = div(length(Y),nFeatIn(this))
    nf  = this.nData[2]

    Z   = reshape(Z,:,nf,nex)
    Y    = reshape(Y,:,nf,nex)

    Ya = mean(Y,this.doNorm)
    Y  = Y .- Ya
    Za = mean(Z,this.doNorm)
    Z  = Z .- Za
    S2y = mean(Y.^2,this.doNorm)
    den = sqrt.(S2y+this.eps)

    tmp = mean(Y.*Z,this.doNorm)
    Z =  Z ./ den
    Y  = Y .* tmp
    Y =  Y ./ den.^3
    dY = Z-Y
    dYa = mean(dY,this.doNorm)
    dY = dY .- dYa
    return reshape(dY,:,nex)
end
