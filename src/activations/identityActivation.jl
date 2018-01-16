export identityActivation

"""
 identity activation A = identity(Y)

 Input:

   Y - array of features

 Optional Input:

   doDerivative - flag for computing derivative, set via varargin
                  Ex: identityActivation(Y,true);

 Output:

  A  - activation
  dA - derivatives
"""
function identityActivation{T}(Y::Array{T},doDerivative::Bool=false)

if doDerivative
    dA = ones(T,Y);
else
    dA = zeros(T,0)
end

return A,dA
end
