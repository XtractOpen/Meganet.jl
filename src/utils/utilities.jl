
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
    balance(nex::Int, batchsize::Int, n::Int)

Evenly distribute the elements of `nex` in increments of `batchsize` into `n`
chunks, ignoring the remainder.

### Example
    julia> balance(10, 1, 3)
    3-element Array{UnitRange{Int64},1}:
     1:3
     4:6
     7:9

"""
function balance(nex::Int, batchsize::Int, nworkers::Int)
   batches_per_worker = floor(Int, nex/batchsize/nworkers)
   [((p-1)*batchsize*batches_per_worker +1):(p*batchsize*batches_per_worker) for p in 1:nworkers]
end
