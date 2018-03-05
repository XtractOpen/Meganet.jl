export jumping_average, jumping_averageT

function numel(Y)
    return length(Y)
end

import Base.*
*(op :: AbstractLinearOperator{T}, M :: Array{T,2}) where {T} = op.prod(M)


function lastOne(n)
    ei = zeros(n)
    ei[end] = 1
    return ei
end

function fft2(Y)
    return fft(fft(Y,1),2)
end

function ifft2(Y)
    return ifft(ifft(Y,1),2)
end


function ndgrid_fill(a, v, s, snext)
	for j = 1:length(a)
		a[j] = v[div(rem(j-1, snext), s)+1]
	end
end

function ndgrid(vs::AbstractVector{T}...) where {T}
	n = length(vs)
	sz = map(length, vs)
	out = ntuple(i->Array{T}(sz), n)
	s = 1
	for i=1:n
		a = out[i]::Array
		v = vs[i]
		snext = s*size(a,i)
		ndgrid_fill(a, v, s, snext)
		s = snext
	end
	out
end

# --- meshgrid
meshgrid(v::AbstractVector) = meshgrid(v, v)
function meshgrid(vx::AbstractVector{T}, vy::AbstractVector{T}) where {T}
	m, n = length(vy), length(vx)
	vx = reshape(vx, 1, n)
	vy = reshape(vy, m, 1)
	(repmat(vx, m, 1), repmat(vy, 1, n))
end

function meshgrid(vx::AbstractVector{T}, vy::AbstractVector{T},
	                 vz::AbstractVector{T}) where {T}
	m, n, o = length(vy), length(vx), length(vz)
	vx = reshape(vx, 1, n, 1)
	vy = reshape(vy, m, 1, 1)
	vz = reshape(vz, 1, 1, o)
	om = ones(Int, m)
	on = ones(Int, n)
	oo = ones(Int, o)
	(vx[om, :, oo], vy[:, on, oo], vz[om, on, :])
end

"""
    mean(f, A, region)

Apply the function `f` to each element of `A`, and compute the mean along dimension in `region`.
"""
function Base.mean(f::Function, a::AbstractArray, region::Int)
    x = Base.mapreducedim(f, +, a, region)
    n = max(1, Base._length(x)) // Base._length(a)
    x .= x .* n

    return x
end

"""
    mean!(f, r, A)

Apply `f` to each element of A, and compute the mean over the singleton dimensions of `r`, and write the results to `r`.
"""
function Base.mean!(f::Function, r::AbstractArray{T}, a::AbstractArray) where {T<:Number}
    fill!(r, zero(T))
    x = Base.mapreducedim!(f, +, r, a)
    n = max(1, Base._length(x)) // Base._length(a)
    x .= x .* n

    return x
end

"""
    jumping_average(Y::Array{T,2}, jump::Int) where {T<:Number}

For each column in `Y`, take the mean of `jump` elements at a time.

# Example

"""
function jumping_average(Y0::Array{T,2}, jump::Int) where {T<:Number}
    nf, nex = size(Y0)
    (nf%jump != 0) && error("Columns of Y0 are not divisible by jump length")
    m = div(nf, jump)
    out = Array{T,2}(m, nex)

    for j in 1:nex
        for i in 1:m
            # Cartesian Range is used to create non-allocating views of the array
           @inbounds  out[i,j] = rangesum(Y0, CartesianRange(CartesianIndex(((i-1)*jump+1,j)), CartesianIndex(i*jump,j))) /jump
        end
    end

    return out
end

@inline function rangesum(A::Array{T,2}, R) where {T<:Number}
    s = zero(T)
    @simd for I in R
        @inbounds s += A[I]
    end
    return s
end

"""
    jumping_averageT(W::Array{T,2}, jump::Int) where {T<:Number}

Adjoint of `jumping_average` where `W` is the output of `jumping_average(Y, jump)`.
"""
function jumping_averageT(W::Array{T,2}, jump::Int, scale::Bool = false) where {T<:Number}
    m, nex = size(W)
    out = Array{T,2}(m*jump, nex)
    fact = 1//jump

    if scale
        for i in 1:nex
            @simd for j in 1:m
                @inbounds tmp = W[j, i] .* fact
                @inbounds out[(jump*(j-1)+1 : jump*j), i] .= tmp
            end
        end
    else
        for i in 1:nex
            @simd for j in 1:m
                @inbounds tmp = W[j, i]
                @inbounds out[(jump*(j-1)+1 : jump*j), i] .= tmp
            end
        end
    end

    return out
end
