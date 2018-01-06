export Connector

type Connector <: AbstractMeganetElement
    K
    b
    outTimes
    Q
    Connector(K;b=0.0,outTimes=0,Q=I) = new(K,b,outTimes,Q)
end

nTheta(this::Connector) = 0
nFeatIn(this::Connector) = size(this.K,2)
nFeatOut(this::Connector) = size(this.K,1)
nDataOut(this::Connector) = ((Q==I) ? nFeatOut(this) : size(Q,1))
initTheta(this::Connector) = zeros(0)

function apply(this::Connector,theta,Y0,doDerivative=true)
    nex = div(length(Y0),nFeatIn(this))
    Y0  = reshape(Y0,:,nex)
    Y = this.K*Y0 .+ this.b
    if this.outTimes==1
        Ydata = this.Q*Y
    else
        Ydata = []
    end
    tmp = Y0;
    return Ydata, Y, tmp
end

function Jmv(this::Connector,dtheta,dY,theta,Y,tmp=nothing)

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

function JTmv(this::Connector,Wdata,W,theta,Y,tmp=nothing)
    nex = div(length(Y),nFeatIn(this))
    if isempty(W)
        W = 0
    else
        W     = reshape(W,:,nex);
    end

    if !isempty(Wdata)
        Wdata = reshape(Wdata,:,nex);
        W     = W+ this.Q'*Wdata;
    end

    dtheta = zeros(0);
    W   = this.K'*W;

    return dtheta,W
end
