export singleLayer,getSingleLayer

mutable struct singleLayer{T, TK <: AbstractConvKernel{T}, TN <: Union{batchNormNN{T}, normLayer{T}}} <: AbstractMeganetElement{T}
        activation  :: Function # activation function
        activation! :: Function # in place activation function
        K           :: TK # transformation type
        nLayer      :: TN # normalization layer
        Bin         :: Array{T} # bias inside nonlinearity
        Bout        :: Array{T} # bias outside nonlinearity

end

function getSingleLayer(TYPE::Type, K,nLayer;Bin=zeros(TYPE,nFeatOut(K),0),Bout=zeros(TYPE,nFeatOut(K),0),
                        activation=tanhActivation,activation_inplace=tanhActivation!)
	singleLayer(activation,activation_inplace,K,nLayer,Bin,Bout);
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

    nex = div(length(Yin),nFeatIn(this))
    Y   = reshape(Yin,:,nex)
    th1,th2,th3,th4 = splitWeights(this,theta)

    Yout::Array{T,2}     =  getOp(this.K,th1)*Y 
    if doDerivative
      if isempty(tmp)
        tmp = copy(Yout)
      else
        tmp .= Yout
      end
    end
  
    Yout,  = apply(this.nLayer,th4,Yout,[],false) #TODO passing empty array is a bit hacky
    Yout .+= this.Bin * th2

    Yout,  = this.activation!(Yout,[],false)

    Yout .+= this.Bout*th3
    Ydata  = Yout
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


function Jthetamv(this::singleLayer{T},dtheta::Array{T},theta::Array{T},Yin::Array{T},tmp::Array{T}) where {T <: Number}
    nex            = div(length(Yin),nFeatIn(this))
    Y              = reshape(Yin,:,nex)
	
    th1,th2,th3,th4     = splitWeights(this,theta)
    dth1,dth2,dth3,dth4 = splitWeights(this,dtheta)
	
	# re-compute derivative of activation
	Yout              = copy(tmp);
    Yout,dummy,tmpNL  = apply(this.nLayer,th4,Yout)
    Yout .+= this.Bin * th2
	A,dA   = this.activation(Yout,true)
    
	
    dZ::Array{T,2} = Jthetamv(this.K,dth1,th1,Y) 
    Kop = getOp(this.K,th1)
    dZ  = Jmv(this.nLayer,dth4,dZ,th4,tmp,tmpNL)[2]
	dZ .+= this.Bin*dth2
    dZ .*= dA 
    dZ .+= this.Bout*dth3
    return dZ, dZ
end

function JYmv(this::singleLayer{T},dYin::Array{T},theta::Array{T},Y::Array{T},tmp::Array{T}) where {T <: Number}
    nex = div(length(dYin),nFeatIn(this))
    th1,th2,th3,th4 = splitWeights(this,theta)
 
	# re-compute derivative of activation
	Yout              = copy(tmp);
    Yout,dummy,tmpNL  = apply(this.nLayer,th4,Yout)
    Yout .+= this.Bin * th2
	A,dA   = this.activation(Yout,true)
 
 
    Kop  = getOp(this.K,th1)
    dY   = reshape(dYin,:,nex)
    dZ   = Kop*dY
    dZ   = JYmv(this.nLayer,dZ,th4,tmp,tmpNL)[2]
    dZ .*= dA
    return dZ,dZ
end

function Jmv(this::singleLayer{T},dtheta::Array{T},dYin::Array{T},theta::Array{T},Yin::Array{T},tmp::Array{T}) where {T <: Number}
    nex = div(length(Yin),nFeatIn(this))
    th1,th2,th3,th4 = splitWeights(this,theta)
    dth1,dth2,dth3,dth4 = splitWeights(this,dtheta)

	# re-compute derivative of activation
	Yout              = copy(tmp);
    Yout,dummy,tmpNL  = apply(this.nLayer,th4,Yout)
    Yout .+= this.Bin * th2
	A,dA   = this.activation(Yout,true)
 
    dY = reshape(dYin,:,nex);
    Kop = getOp(this.K,th1)
    dZ::Array{T, 2} = Kop*dY;

    Y   = reshape(Yin,:,nex);
    dZ += Jthetamv(this.K,dth1,th1,Y) 
    dZ  = Jmv(this.nLayer,dth4,dZ,th4,tmp,tmpNL)[2]
	dZ .+ this.Bin*dth2

    dZ .*= dA 
    dZ .+= this.Bout*dth3
    return dZ,dZ
end

function JTmv(this::singleLayer{T},Zin::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp::Array{T}) where {T <: Number}
    nex  = div(length(Y),nFeatIn(this))
    Z    = reshape(Zin,:,nex)
    th1,th2,th3,th4  = splitWeights(this,theta)
    Kop = getOp(this.K,th1)
	
	# re-compute derivative of activation
	Yout              = copy(tmp);
    Yout,dummy,tmpNL  = apply(this.nLayer,th4,Yout)
    Yout .+= this.Bin * th2
	A,dA   = this.activation(Yout,true)
 

    dth3      = vec(sum(this.Bout'*Z,2))
    dAZ       = dA.*Z
    dth2      = vec(sum(this.Bin'*reshape(dAZ,:,nex),2))
    
	dth4,dAZ  = JTmv(this.nLayer,dAZ,zeros(T,0),th4,tmp,tmpNL) # this not type stable
    dth1      = JthetaTmv(this.K, dAZ,th1,Y) # this not type stable

    dY   = Kop'*reshape(dAZ,:,nex)
    dtheta = [vec(dth1); vec(dth2); vec(dth3); vec(dth4)]
    
    return dtheta, dY

end

function JthetaTmv(this::singleLayer{T},Zin::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp::Array{T}) where {T <: Number}
    nex       = div(length(Zin),nFeatOut(this))
    th1,th2,th3,th4  = splitWeights(this,theta)

	# re-compute derivative of activation
	Yout              = copy(tmp);
    Yout,dummy,tmpNL  = apply(this.nLayer,th4,Yout)
    Yout .+= this.Bin * th2
	A,dA   = this.activation(Yout,true)
 

    Z         = reshape(Zin,:,nex);
    dAZ       = dA.*Z;
    dth3      = vec(sum(this.Bout'*Z,2));
    dth2      = vec(sum(this.Bin'*reshape(dAZ,:,nex),2));
    dth4,dAZ  = JTmv(this.nLayer,dAZ,(T)[],th4,tmp,tmpNL)
    dth1      = JthetaTmv(this.K,dAZ,th1,Y);
    return [vec(dth1); vec(dth2); vec(dth3); vec(dth4)];
end

function JYTmv(this::singleLayer{T},Zin::Array{T},dummy::Array{T},theta::Array{T},Y::Array{T},tmp) where {T <: Number}
    
	nex  = div(length(Y),nFeatIn(this))
    th1,th2,th3,th4 = splitWeights(this,theta)
    
	# re-compute derivative of activation
	Yout              = copy(tmp);
    Yout,dummy,tmpNL  = apply(this.nLayer,th4,Yout)
    Yout .+= this.Bin * th2
	A,dA   = this.activation(Yout,true)
 
	Kop = getOp(this.K,th1)
    Z    = reshape(Zin,:,nex)
    dAZ::Array{T,2}  = dA.*Z
    dAZ = JYTmv(this.nLayer,dAZ,(T)[],th4,tmp,tmpNL)
    ret::Array{T,2} = Kop'*reshape(dAZ,:,nex)
    return ret #TODO: @lars or eldad rename this variable as I'm not sure what to call it
end