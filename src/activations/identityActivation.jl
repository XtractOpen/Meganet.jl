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
function identityActivation(Y::Array{T},dA,doDerivative::Bool=false) where {T}

if doDerivative
    if isempty(dA)
        dA = ones(T,Y);
    else
        dA .= ones(T,Y);
    end
end

return A,dA #Depricated? A Isnt even declared lol
end
