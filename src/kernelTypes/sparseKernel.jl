export SparseKernel, getSparseConvKernel2D

"""
kernel where weights parameterize entried of a sparse matrix
"""
type SparseKernel{T}
    nK::Array{Int}
        ival::Array{Int}
        jval::Array{Int}
        colptr::Array{Int}
        rowval::Array{Int}
        nzval::Array
        Qs::AbstractArray
end

"""
2D coupled convolution kernel using sparse matrices
"""
function getSparseConvKernel2D(TYPE::Type, nImg,sK;stride=[1;1])
    A = getCoupledConvMat(TYPE,collect(1.:prod(sK)),nImg,sK,stride=stride)
    A = round.(Int,A)

    nzA = nnz(A);
    nTh = maximum(A);

    Q = sparse(collect(1:nzA),A.nzval,ones(TYPE,nzA),nzA,nTh);
	
	#### this used to be in the constructor:####
	ival,jval = ind2sub(size(A),find(A))
    if size(Q,1)!=length(ival)
        error("sizes must match")
    end
    nK = [size(A,1); size(A,2)]
    return SparseKernel{TYPE}(nK,ival,jval,sort(A.colptr),A.rowval,one(TYPE)*A.nzval,Q)
end

function nTheta(this::SparseKernel)
    return size(this.Qs,2)
end

function nFeatIn(this::SparseKernel)
    return this.nK[2]
end

function nFeatOut(this::SparseKernel)
    return this.nK[1]
end

function initTheta(this::SparseKernel)
    return randn(nTheta(this))
end

function getOp{T}(this::SparseKernel,theta::Array{T})
     this.nzval = this.Qs*vec(theta)
	return SparseMatrixCSC{T}(this.nK[1],this.nK[2],this.colptr,this.rowval,this.nzval)
end

function Jthetamv(this::SparseKernel,dtheta,theta,Y,tmp=nothing)
    return getOp(this,dtheta)*Y
end

function JthetaTmv(this::SparseKernel,Z,theta,Y,tmp=nothing)
    Z = reshape(Z,this.nK[1],:)
    Y = reshape(Y,this.nK[2],:)
    #t = sum(Z[this.ival,:] .* Y[this.jval,:],2)
    
    t = zeros(length(this.ival))
    for j=1:size(Z,2)
        for i=1:length(this.ival)
            @inbounds t[i] += Z[this.ival[i],j] * Y[this.jval[i],j]
        end
    end

    return this.Qs'*t;
end
