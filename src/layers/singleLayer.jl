export singleLayer

type singleLayer <: AbstractMeganetElement
        activation  # activation function
        K           # transformation type
        nLayer      # normalization layer
        Bin         # bias inside nonlinearity
        Bout        # bias outside nonlinearity
        singleLayer(K,nLayer;Bin=zeros(nFeatOut(K),0),Bout=zeros(nFeatOut(K),0),activation=tanhActivation) =
        new(activation,K,nLayer,Bin,Bout)
end

function splitWeights(this::singleLayer,theta)
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

function apply(this::singleLayer,theta,Y,doDerivative=false)
    tmp = Array{Any}(2)
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

function initTheta(this::singleLayer)
    return [vec(initTheta(this.K)); 0.1*ones(size(this.Bin,2),1) ; 0.1*ones(size(this.Bout,2),1); initTheta(this.nLayer) ]
end


function Jthetamv(this::singleLayer,dtheta,theta,Y,tmp)
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

function JYmv(this::singleLayer,dY,theta,Y,tmp)
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

function Jmv(this::singleLayer,dtheta,dY,theta,Y,tmp)
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

function JTmv(this::singleLayer,Z,dummy,theta,Y,tmp)
    dA   = tmp[2]
    nex  = div(length(Y),nFeatIn(this))
    Z    = reshape(Z,:,nex)
    th1,th2,th3,th4  = splitWeights(this,theta)
    Kop = getOp(this.K,th1)

    dth3      = vec(sum(this.Bout'*Z,2))
    dAZ       = dA.*Z
    dth4,dAZ  = JTmv(this.nLayer,dAZ,[],th4,Kop*Y.+this.Bin*th2,tmp[1])
    dth2      = vec(sum(this.Bin'*reshape(dAZ,:,nex),2))
    dth1      = JthetaTmv(this.K, dAZ,theta,Y)

    dY   = Kop'*reshape(dAZ,:,nex)
    dtheta = [vec(dth1); vec(dth2); vec(dth3); vec(dth4)]

    return dtheta, dY

end

function JthetaTmv(this::singleLayer,Z,dummy,theta,Y,tmp)
    dA        = tmp[2]
    nex       = div(length(Z),nFeatOut(this))
    th1,th2,th3,th4  = splitWeights(this,theta)

    Z         = reshape(Z,:,nex);
    dAZ       = dA.*Z;
    dth3      = vec(sum(this.Bout'*Z,2));
    Kop       = getOp(this.K,th1)
    dth4,dAZ  = JTmv(this.nLayer,dAZ,[],th4,Kop*Y.+this.Bin*th2,tmp[1])
    dth1      = JthetaTmv(this.K,dAZ,theta,Y);
    dth2      = vec(sum(this.Bin'*reshape(dAZ,:,nex),2));
    return [vec(dth1); vec(dth2); vec(dth3); vec(dth4)];
end

function JYTmv(this::singleLayer,Z,dummy,theta,Y,tmp)
    dA   = tmp[2]
    nex  = div(length(Y),nFeatIn(this))
    th1,th2,th3,th4 = splitWeights(this,theta)
    Kop = getOp(this.K,th1)
    Z    = reshape(Z,:,nex)
    dAZ  = dA.*Z
    dAZ = JYTmv(this.nLayer,dAZ,[],th4,Kop*Y.+this.Bin*th2,tmp[1])
    return Kop'*reshape(dAZ,:,nex)
end
