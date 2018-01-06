export AbstractMeganetElement, nTheta, nFeatIn, nFeatOut,nDataOut, splitWeights, initTheta
export JYTmv, JYmv, Jthetamv, JthetaTmv, Jmv, JTmv
export linearizeY, linearizeTheta
export apply
export getJOp, getJYOp, getJthetaOp

"""
 here go all functions that are shared among elements of a Meganet
 (i.e., transformations, layers, networks, blocks,...)

 An AbstractMeganetElement does the following

  Y_k+1 = apply(theta,Y_k)

 where 'apply' can be everything from a single affine transformation to a
 ResNet block. All these operations have a similar structure, e.g.,
 provide derivatives w.r.t. theta and Y_k, ...

 Example: Consider a Neural network consisting of two layers.
 T1  = dense([12,8]);
 T2  = dense([24,12])
 net = NN({T1, T2});

 Calling apply(net,theta,Y) results in a nested evaluation

 apply(net,theta,Y) = apply(T2, theta2, apply(T1, theta1, Y));

 This example shows that each element of the network needs the following
 functions

  split - partition the input parameters into parameters of elements
          describing this object (in our case theta -> theta1, theta2
  apply - evaluate the action (e.g., forward propagation, filtering, ..)
          in many cases this involves calling 'apply' for other objects
          (e.g., for different layers, kernels,...)
  Jthetamv  - compute the action of the Jacobian w.r.t theta on a vector
  JthetaTmv - compute the action of the transpose(Jacobian) w.r.t theta on a vector
  JYmv      - compute the action of the Jacobian w.r.t Y on a vector
  JYTmv     - compute the action of the transpose(Jacobian) w.r.t Y on a vector

 In addition, elements of this class also need to provide the folowing
 methods

  nTheta   - return the number of parameters, numel(theta) for this
              element (may have to ask lower-level elements for this)
  nFeatIn  - number of input features
  nFeatOut - number of output features
  initTheta - initialize parameters

"""
abstract type AbstractMeganetElement end

"""
return number of trainable weights
"""
nTheta(this::AbstractMeganetElement) = error("An AbstractMeganetElement must provide method nTheta")

"""
number of input features
"""
nFeatIn(this::AbstractMeganetElement) = error("An AbstractMeganetElement must provide method nFeatIn")

"""
number of data going to classifier
"""
nDataOut(this::AbstractMeganetElement) = nFeatOut(this)


"""
number of output features
"""
nFeatOut(this::AbstractMeganetElement) = error("An AbstractMeganetElement must provide method nFeatOut")

"""
split weights into parts
"""
splitWeights(this::AbstractMeganetElement,theta) = error("An AbstractMeganetElement must provide method splitWeights")

"""
initialize weights
"""
initTheta(this::AbstractMeganetElement) = randn(nTheta(this))

"""
transforms feature vector Y using the weights theta
"""
apply(this::AbstractMeganetElement,theta,Y,doDerivative)= error("An AbstractMeganetElement must provide method apply ")

"""
computes dY = transpose(J_Y(theta,Y))\*W

Input:

   W     - vector or matrix
   theta - current theta
   Y     - current Y
   tmp   - intermediates used in derivative computations
           (e.g., hidden features, activations, derivatives,
           ... )

Output:

  dY     - directional derivative, numel(dY)==numel(Y)
"""
function JYTmv(this::AbstractMeganetElement,Wdata,W,theta,Y,tmp=nothing)
    return JTmv(this,Wdata,W,theta,Y,tmp)[2]
end

"""
computes dZ = J_Y(theta,Y)\*dY

Input:

   dY    - perturbation in Y
   theta - current theta
   Y     - current Y
   tmp   - intermediates used in derivative computations
           (e.g., hidden features, activations, derivatives,
           ... )

Output:

  dZ     - directional derivative, numel(dZ)==numel(Z)
"""
function JYmv(this::AbstractMeganetElement,dY,theta,Y,tmp=nothing)
    return Jmv(this,[],dY,theta,Y,tmp);
end

"""
linearization with respect to Y, i.e.,

Z(theta,Y+dY) approx Z(theta,Y) + J\*dY

Input:

   theta - current theta
   Y     - current Y
   tmp   - intermediates used in derivative computations
           (e.g., hidden features, activations, derivatives,
           ... )

Output:

  Z     - current output features
  J     - Jacobian, LinearOperator
"""
function linearizeY(this::AbstractMeganetElement,theta,Y)
    Z,~,tmp  = apply(this,theta,Y)
    J        = getJYOp(this,theta,Y,tmp)
    return Z,J
end

"""
J = getJYOp(this,theta,Y,tmp)

constructs Jacobian w.r.t. Y around current (theta,Y)

Input:

   W     - vector or matrix
   theta - current theta
   Y     - current Y
   tmp   - intermediates used in derivative computations
           (e.g., hidden features, activations, derivatives,
           ... )

Output:

  dY     - directional derivative, numel(dY)==numel(Y)
"""
function getJYOp(this::AbstractMeganetElement,theta,Y,tmp=nothing)
    nex    = div(length(Y),nFeatIn(this))
    m      = nex*nDataOut(this)
    n      = length(Y);
    Amv    = x -> JYmv(this,x,theta,Y,tmp)
    ATmv   = x -> JYTmv(this,x,[],theta,Y,tmp)
    return LinearOperator(m,n,false,false,Amv,ATmv,ATmv);
end

"""
dZ = abstractMeganetElement.Jthetamv(this,W,theta,Y,tmp)

computes dZ = J_theta(theta,Y)\*dtheta

Input:

   dtheta- perturbation in theta
   theta - current theta
   Y     - current Y
   tmp   - intermediates used in derivative computations
           (e.g., hidden features, activations, derivatives,
           ... )

Output:

  dZ     - directional derivative, numel(dZ)==numel(Z)
"""
function Jthetamv(this::AbstractMeganetElement,dtheta,theta,Y,tmp)
    return Jmv(this,dtheta,0*Y,theta,Y,tmp);
end

"""
computes dtheta = transpose(J_theta(theta,Y))\*W

Input:

   W     - vector or matrix
   theta - current theta
   Y     - current Y
   tmp   - intermediates used in derivative computations
           (e.g., hidden features, activations, derivatives,
           ... )

Output:

  dtheta - directional derivative, numel(dtheta)==numel(theta)
"""
function JthetaTmv(this::AbstractMeganetElement,Wdata,W,theta,Y,tmp=nothing)
     return JTmv(this,Wdata,W,theta,Y,tmp)[1]
end

"""
J = abstractMeganetElement.getJthetaOp(this,theta,Y,tmp)

constructs Jacobian w.r.t. theta as LinearOperator

Input:

   theta - current theta
   Y     - current Y
   tmp   - intermediates used in derivative computations
           (e.g., hidden features, activations, derivatives,
           ... )

Output:

  J     - Jacobian, LinearOperator
"""
function getJthetaOp(this::AbstractMeganetElement,theta,Y,tmp=nothing)
    nex    = div(length(Y),nFeatIn(this))
    m      = nex*nDataOut(this)
    n      = length(theta)
    Amv    = x -> Jthetamv(this,x,theta,Y,tmp)
    ATmv   = x -> JthetaTmv(this,x,[],theta,Y,tmp)
    return LinearOperator(m,n,false,false,Amv,ATmv,ATmv)
end

"""
linearization with respect to theta, i.e.,

Z(theta+dth,Y) \approx Z(theta,Y) + J*dth

Input:

   theta - current theta
   Y     - current Y

Output:

  Z     - output features
  J     - Jacobian, LinearOperator
"""
function linearizeTheta(this::AbstractMeganetElement,theta,Y)
    Z,Z2,tmp = apply(this,theta,Y)
    J       = getJthetaOp(this,theta,Y,tmp)
    return Z,J
end

"""
computes dZ = J_theta(theta,Y)\*dtheta + J_Y(theta,Y)\*dY

Input:

   dtheta- perturbation in theta
   dY    - perturbation of input features Y
   theta - current theta
   Y     - current Y
   tmp   - intermediates used in derivative computations
           (e.g., hidden features, activations, derivatives,
           ... )

Output:

  dZ     - directional derivative, numel(dZ)==numel(Z)
"""
function Jmv(this::AbstractMeganetElement,dtheta,dY,theta,Y,tmp=nothing)

    if isempty(dtheta) || norm(vec(dtheta))==0
        dZdata = 0.0
        dZ     = 0.0
    else
        dZdata,dZ = Jthetamv(this,dtheta,theta,Y,tmp)
    end

    if !isempty(dY) && norm(vec(dY))>0
        dZdt,dZt = JYmv(this,dY,theta,Y,tmp);
        dZdata +=  dZdt;
        dZ     +=  dZt;
    end
    return dZdata, dZ
end

"""
computes [dtheta;dY] = [J_theta(theta,Y)'; J_Y(theta)']\*Z

Input:

   Z     - perturbation of output
   theta - current theta
   Y     - current Y
   tmp   - intermediates used in derivative computations
           (e.g., hidden features, activations, derivatives,
           ... )
   doDerivative - vector with two elements for theta and Y
   derivative. Important only when nargout==1. default=[1,0];

"""
function JTmv(this::AbstractMeganetElement,Wdata,W,theta,Y,tmp=nothing)
    dtheta = JthetaTmv(this,Wdata,W,theta,Y,tmp)
    dY     = JYTmv(this,Wdata,W,theta,Y,tmp)
    return vec(dtheta), vec(dY)
end

"""
constructs Jacobian J(theta,Y) such that

Z(theta+dth,Y+dY) \approx Z(theta,Y) + J*[dth; dY]

Input:

   theta - current theta
   Y     - current Y
   tmp   - intermediates used in derivative computations
           (e.g., hidden features, activations, derivatives,
           ... )

Output:

  J     - Jacobian, LinearOperator
"""
function getJOp(this::AbstractMeganetElement,theta,Y,tmp=nothing)
    nex    = div(length(Y),nFeatIn(this))
    m      = nex*nDataOut(this)
    nth    = length(theta)
    nY     = length(Y)
    Amv    = x -> Jmv(this,x[1:nth],x[nth+1:end],theta,Y,tmp)
    ATmv   = x -> JTmv(this,x,[],theta,Y,tmp)
    J      = LinearOperator(m,nth+nY,false,false,Amv,ATmv,ATmv)
    return J
end
