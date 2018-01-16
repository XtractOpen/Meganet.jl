export reluActivation

"""
 relu activation A = relu(Y)

 Input:

   Y - array of features

 Optional Input:

   doDerivative - flag for computing derivative, set via varargin
                  Ex: reluActivation(Y,true);

 Output:

  A  - activation
  dA - derivatives
"""
function reluActivation{T}(Y::Array{T},doDerivative::Bool=false)

Y = max.(Y,0);

if doDerivative
    dA = sign(Y);
else
    dA = zeros(T,0)
end

return A,dA
end
