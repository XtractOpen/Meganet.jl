
using jInv.Mesh
export getConvMatPeriodic, getAverageMatrix, getCoupledConvMat

function getAverageMatrix(nImg,nc)

    A1 = spdiagm((fill(.5,nImg[1]),fill(.5,nImg[1]-1)),(0,1),nImg[1],nImg[1])
    A1 = A1[1:2:end,:]
    A2 = spdiagm((fill(.5,nImg[2]),fill(.5,nImg[2]-1)),(0,1),nImg[2],nImg[2])
    A2 = A2[1:2:end,:]
    Av    = kron(A2,A1);
    return kron(speye(nc),Av);
end
function getCoupledConvMat(K,nImg,sK;stride=[1;1])
    # build single 2D convolution operator
    nK  = prod(sK[1:2])
    K   = reshape(K,tuple(sK...)) # basic kernel
    K = flipdim(K,2)
    K = flipdim(K,1)
    # K   = permutedims(K,(2,1,3,4))
    # apply stride
    idx = falses(nImg[1],nImg[2])
    idx[1:stride[1]:end,:] = true
    idx[:,1:stride[2]:end] = true

    A = spzeros(0,prod(nImg[1:2])*sK[3])
    for i=1:sK[4]
        Ak = spzeros(prod(nImg[1:2]),0)
        for j=1:sK[3]
            Ak = [Ak getConvMatPeriodic(K[:,:,j,i],[nImg[1:2]; 1])]
        end
        A = [A; Ak[vec(idx).==true,:]]
    end
    return A
end

function getConvMatPeriodic{T<:AbstractFloat}(K::Array{T},n::Array{Int64})

G  = reshape(collect(1:prod(n)),tuple(n...))

jj = zeros(Int64,0); ii = zeros(Int64,0); vv = zeros(0);
I1,I2,I3 = ndgrid(1:n[1],1:n[2],1:n[3]);
for i=1:size(K,1)
    for j=1:size(K,2)
        for k=1:size(K,3)
            ofi = i - div(size(K,1)-1,2) - 1
            ofj = j - div(size(K,2)-1,2) - 1
            ofk = k - div(size(K,3)-1,2) - 1

            t1 = vec(I1.+ofi)
            t1[t1.<=0]   += n[1]
            t1[t1.>n[1]] -= n[1]

            t2 = vec(I2.+ofj)
            t2[t2.<=0]   += n[2]
            t2[t2.>n[2]] -= n[2]

            t3 = vec(I3.+ofk)
			t3[t3.<=0]   += n[3]
            t3[t3.>n[3]] -= n[3]

            jj = [jj; vec(G)];
            ii = [ii; vec(G[sub2ind(n,t1, t2,t3)])];
            vv = [vv; ones(prod(n))*K[i,j,k]];
        end
    end
end
A = sparse(ii, jj ,vv,prod(n),prod(n));
return A
end
