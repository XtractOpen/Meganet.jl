export tanhActivation

"""
 hyperbolic tan activation A = tanh(Y)

 Input:

   Y - array of features

 Optional Input:

   doDerivative - flag for computing derivative, set via varargin
                  Ex: tanhActivation(Y,true);

 Output:

  A  - activation
  dA - derivatives
"""
function tanhActivation(Y::Array{T},doDerivative::Bool=false) where {T}

    A = tanh.(Y)
    dA = similar(Y)
    if doDerivative
         dA .= 1-A.^2
    else
        fill!(dA, zero(T))
    end
    return A, dA
end
