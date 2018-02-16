export tanhActivation, tanhActivation!

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
function tanhActivation(Y::Array{T,2},doDerivative::Bool=false) where {T <: Number}

    A = tanh.(Y)
    dA = zeros(A)
    if doDerivative
         dA .= one(T) .- A.^2
    end
    return A, dA
end

function tanhActivation!(A::Array{T,2},dA,doDerivative::Bool=false) where {T <: Number}

    A .= tanh.(A)
    if doDerivative
        if isempty(dA)
            dA = one(T) .- A.^2
        else
            dA .= one(T) .- A.^2
        end
    end
    return A, dA
end
