# compare convolutions
using Base.Test
using Meganet
using LinearOperators


nImg = [32,32]
sK = [3,3,16,16]

nex = 50;

Kfft = convFFTKernel(nImg,sK)

theta = initTheta(Kfft)
Y = randn(nFeatIn(Kfft),nex);

import Base.*
*(op :: AbstractLinearOperator, M :: Array{Float64}) = op.prod(M)

@time begin
        Z1 = getOp(Kfft,theta)*Y;
 end

Ksparse = getSparseConvKernel2D(nImg,sK)
@time Z2 = getOp(Ksparse,theta)*Y;

norm(Z1[:]-Z2[:])/norm(Z1)
