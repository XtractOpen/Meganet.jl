export singleLayer,getSingleLayer

mutable struct singleLayer{T, TK <: AbstractConvKernel{T}, TN <: Union{batchNormNN{T}, normLayer{T}}} <: AbstractMeganetElement{T}
        activation :: Function # activation function
        K          :: TK # transformation type
        nLayer     :: TN # normalization layer
        Bin        :: Array{T} # bias inside nonlinearity
        Bout       :: Array{T} # bias outside nonlinearity

end

function getSingleLayer(TYPE::Type, K,nLayer;Bin=zeros(TYPE,nFeatOut(K),0),Bout=zeros(TYPE,nFeatOut(K),0),activation=tanhActivation!)
	singleLayer(activation,K,nLayer,Bin,Bout);
end


function splitWeights(this::singleLayer{T},theta::Array{T}) where {T <: Number}
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

function apply(this::singleLayer{T},theta::Array{T},Yin::Array{T},tmp,doDerivative=false) where {T <: Number}

    if isempty(tmp)
        tmp = Array{Any}(2)
        tmp[1] = Array{Any}(0,0)
        tmp[2] = Array{Any}(0)
    end
    nex = div(length(Yin),nFeatIn(this))
    Y   = reshape(Yin,:,nex)
    th1,th2,th3,th4 = splitWeights(this,theta)

    Yout::Array{T,2}     =  getOp(this.K,th1)*Y 
    Yout .+= this.Bin * th2
    Yout,dummy,tmp[1] = apply(this.nLayer,th4,Yout,tmp[1],doDerivative)

    Yout,tmp[2]  = this.activation(Yout,tmp[2],doDerivative)

    Yout .+= this.Bout*th3
    Ydata = Yout
    return Ydata, Yout, tmp
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

function initTheta(this::singleLayer{T}) where {T <: Number}
    return [vec(initTheta(this.K)); convert(T,0.01)*ones(T,size(this.Bin,2),1) ; convert(T,0.01)*ones(T,size(this.Bout,2),1); initTheta(this.nLayer) ]
end


function Jthetamv(this::singleLayer{T},dtheta::Array{T},theta::Array{T},Yin::Array{T},tmp) where {T <: Number}
    dA::Array{T,2}             = tmp[2]
    nex            = div(length(Yin),nFeatIn(this))
    Y              = reshape(Yin,:,nex)

    th1,th2,th3,th4     = splitWeights(this,theta)
    dth1,dth2,dth3,dth4 = splitWeights(this,dtheta)
    dZ::Array{T,2} = Jthetamv(this.K,dth1,th1,Y) .+ this.Bin*dth2
    Kop = getOp(this.K,th1)
    dZ  = Jmv(this.nLayer,dth4,dZ,th4,Kop*Y.+this.Bin*th2,tmp[1])[2]
    dZ .*= dA 
    dZ .+= this.Bout*dth3
    return dZ, dZ
end

function JYmv(this::singleLayer{T},dYin::Array{T},theta::Array{T},Y::Array{T},tmp) where {T <: Number}
    dA  = tmp[2]
    nex = div(length(dYin),nFeatIn(this))
    th1,th2,th3,th4 = splitWeights(this,theta)
    Kop         = getOp(this.K,th1)
    dY          = reshape(dYin,:,nex)
    dZ          = Kop*dY
    dZ          = JYmv(this.nLayer,dZ,th4,Kop*Y.+this.Bin*th2,tmp[1])[2]
    # dZ          = dA.*dZ
    dZ .*= dA
    return dZ,dZ
end

function Jmv(this::singleLayer{T},dtheta::Array{T},dYin::Array{T},theta::Array{T},Yin::Array{T},tmp) where {T <: Number}
    dA::Array{T,2}  = tmp[2]
    nex = div(length(Yin),nFeatIn(this))
    th1,th2,th3,th4 = splitWeights(this,theta)
    dth1,dth2,dth3,dth4 = splitWeights(this,dtheta)

    dY = reshape(dYin,:,nex);
    Kop = getOp(this.K,th1)
    dZ::Array{T, 2} = Kop*dY;

    Y   = reshape(Yin,:,nex);
    dZ += Jthetamv(this.K,dth1,th1,Y) .+ this.Bin*dth2
    dZ = Jmv(this.nLayer,dth4,dZ,th4,Kop*Y.+this.Bin*th2,tmp[1])[2]

    dZ .*= dA 
    dZ .+= this.Bout*dth3
    return dZ,dZ
end

function JTmv(this::singleLayer{T},Zin::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp) where {T <: Number}
    dA::Array{T,2}   = tmp[2]
    nex  = div(length(Y),nFeatIn(this))
    Z    = reshape(Zin,:,nex)
    th1,th2,th3,th4  = splitWeights(this,theta)
    Kop = getOp(this.K,th1)

    dth3      = vec(sum(this.Bout'*Z,2))
    dAZ       = dA.*Z
    dth4,dAZ  = JTmv(this.nLayer,dAZ,zeros(T,0),th4,Kop*Y.+this.Bin*th2,tmp[1]) # this not type stable
    dth2      = vec(sum(this.Bin'*reshape(dAZ,:,nex),2))
    dth1      = JthetaTmv(this.K, dAZ,theta,Y) # this not type stable

    dY   = Kop'*reshape(dAZ,:,nex)
    dtheta = [vec(dth1); vec(dth2); vec(dth3); vec(dth4)]
    
    return dtheta, dY

end

function JthetaTmv(this::singleLayer{T},Zin::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp) where {T <: Number}
    dA        = tmp[2]
    nex       = div(length(Zin),nFeatOut(this))
    th1,th2,th3,th4  = splitWeights(this,theta)

    Z         = reshape(Zin,:,nex);
    dAZ       = dA.*Z;
    dth3      = vec(sum(this.Bout'*Z,2));
    Kop       = getOp(this.K,th1)
    dth4,dAZ  = JTmv(this.nLayer,dAZ,(T)[],th4,Kop*Y.+this.Bin*th2,tmp[1])
    dth1      = JthetaTmv(this.K,dAZ,theta,Y);
    dth2      = vec(sum(this.Bin'*reshape(dAZ,:,nex),2));
    return [vec(dth1); vec(dth2); vec(dth3); vec(dth4)];
end

function JYTmv(this::singleLayer{T},Zin::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp) where {T <: Number}
    dA::Array{T,2}   = tmp[2]
    nex  = div(length(Y),nFeatIn(this))
    th1,th2,th3,th4 = splitWeights(this,theta)
    Kop = getOp(this.K,th1)
    Z    = reshape(Zin,:,nex)
    dAZ::Array{T,2}  = dA.*Z
    dAZ = JYTmv(this.nLayer,dAZ,(T)[],th4,Kop*Y.+this.Bin*th2,tmp[1])
    ret::Array{T,2} = Kop'*reshape(dAZ,:,nex)
    return ret #TODO: @lars or eldad rename this variable as I'm not sure what to call it
end