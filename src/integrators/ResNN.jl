export ResNN,getResNN
import Juno

"""
Residual Neural Network block

Y_k+1 = Y_k + h*layer{k}(theta{k},Y_k)
"""
mutable struct ResNN{T} <: AbstractMeganetElement{T}
    layer
    nt
    h
    outTimes
    Q
end

function getResNN(TYPE::Type,layer,nt,h=one(TYPE),outTimes=eye(Int,nt)[:,nt],Q=I)
	h = convert(TYPE,h);
	if nFeatIn(layer)!=nFeatOut(layer)
            error("ResNN layer must be square!")
       end
    return ResNN{TYPE}(layer,nt,h,outTimes,Q)
end


function nTheta(this::ResNN)
    return this.nt*nTheta(this.layer);
end

function nFeatIn(this::ResNN)
    return nFeatIn(this.layer);
end

function nFeatOut(this::ResNN)
    return nFeatOut(this.layer);
end

function nDataOut(this::ResNN)
    if length(this.Q)==1
        n = sum(this.outTimes)*nFeatOut(this.layer)
    else
        n = sum(this.outTimes)*size(this.Q,1)
    end
    return n
end

function initTheta(this::ResNN{T}) where {T<:Number}
    return repmat(vec(initTheta(this.layer)),this.nt,1)
end

# ------- apply forward problems -----------
function  apply(this::ResNN{T},theta::Array{T},Y0::Array{T},doDerivative=true) where {T<:Number}

    nex = div(length(Y0),nFeatIn(this))
    Y   = reshape(Y0,:,nex)
    tmp = Array{Any}(this.nt+1,2)
    if doDerivative
        tmp[1,1] = Y0
    end

    theta = reshape(theta,:,this.nt)

    Ydata = zeros(T,0,nex)
    for i=1:this.nt
        Z,dummy,tmp[i,2] = apply(this.layer,theta[:,i],Y,doDerivative)
        Y +=  this.h * Z
        if doDerivative
            tmp[i+1,1] = Y
        end
        if this.outTimes[i]==1
            Ydata = [Ydata;this.Q*Y]
        end
    end
    return Ydata,Y,tmp
end

# -------- Jacobian matvecs ---------------
function JYmv(this::ResNN{T},dY::Array{T},theta::Array{T},Y::Array{T},tmp) where {T<:Number}
    # if isempty(dY)
    #     dY = 0.0;
    # elseif length(dY)>1
        nex = div(length(dY),nFeatIn(this))
        dY   = reshape(dY,:,nex)
    # end
    dYdata = zeros(T,0,nex);
    theta  = reshape(theta,:,this.nt);
    for i=1:this.nt
        dY += this.h* JYmv(this.layer,dY,theta[:,i],tmp[i,1],tmp[i,2])[2];
        if this.outTimes[i]==1
            dYdata = [dYdata; this.Q*dY];
        end
    end
    return dYdata, dY
end


function  Jmv(this::ResNN{T},dtheta::Array{T},dY::Array{T},theta::Array{T},Y::Array{T},tmp) where {T<:Number}
    nex = div(length(Y),nFeatIn(this))
    if length(dY)==0
         dY = zeros(T,size(Y))
     elseif length(dY)>1
        dY   = reshape(dY,:,nex)
    end

    dYdata = zeros(T,0,nex)
    theta  = reshape(theta,:,this.nt)
    dtheta = reshape(dtheta,:,this.nt)
    for i=1:this.nt
        dY +=  this.h* Jmv(this.layer,dtheta[:,i],dY,theta[:,i],tmp[i,1],tmp[i,2])[2]
        if this.outTimes[i]==1
            dYdata = [dYdata;this.Q*dY]
        end
    end
    return dYdata,dY
end

# -------- Jacobian transpose matvecs ----------------

function JYTmv(this::ResNN{T},Wdata::Array{T},W::Array{T},theta::Array{T},Y::Array{T},tmp) where {T<:Number}

    nex = div(length(Y),nFeatIn(this))
    if length(Wdata)>0
        Wdata = reshape(Wdata,:,sum(this.outTimes),nex)
    end
    if length(W)==0
        W = zero(T)
    else
        W     = reshape(W,:,nex)
    end
    theta  = reshape(theta,:,this.nt)

    cnt = sum(this.outTimes)
    for i=this.nt:-1:1
        if  this.outTimes[i]==1
            W += this.Q'*Wdata[:,cnt,:]
            cnt = cnt-1
        end
        dW = JYTmv(this.layer,W,zeros(T,0),theta[:,i],tmp[i,1],tmp[i,2])
        W  += this.h*dW
    end
    return W
end

function JTmv(this::ResNN{T},Wdata::Array{T},W::Array{T},theta::Array{T},Y::Array{T},tmp) where {T<:Number}

    nex = div(length(Y),nFeatIn(this))
    if !isempty(Wdata) && any(this.outTimes.!=0)
        Wdata = reshape(Wdata,:,sum(this.outTimes),nex)
    end
    if length(W)==0
        if any(this.outTimes.!=0)
            W = zero(T) #assume the Wdata is non-zero
        else
            W = Wdata
        end
    end
    if length(W)>1
        W     = reshape(W,:,nex)
    end

    theta  = reshape(theta,:,this.nt)
    cnt    = sum(this.outTimes)
    dtheta = zeros(T,size(theta))

    for i=this.nt:-1:1
        if  this.outTimes[i]==1
            W +=  this.Q'* Wdata[:,cnt,:]
            cnt = cnt-1
        end

        dmbi,dW = JTmv(this.layer,W,zeros(T,0),theta[:,i],tmp[i,1],tmp[i,2])

        dtheta[:,i]  = this.h*dmbi
        W += this.h*dW
    end

    return vec(dtheta), W
end
