export tanhActivation

"""
 hyperbolic tan activation A = tanh(Y)

 Input:

   Y - array of features

 Optional Input:

   doDerivative - flag for computing derivative, set via varargin
                  Ex: tanhActivation(Y,'doDerivative',0);

 Output:

  A  - activation
  dA - derivatives
"""
function tanhActivation{T}(Y::Array{T},doDerivative::Bool=false)

A = tanh.(Y)

if doDerivative
     dA = 1-A.^2
else
    dA = zeros(T,0)
end
return A,dA
end
