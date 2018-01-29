# Benchmark sparseKernel.jl
using Meganet, BenchmarkTools

# Setup example from tests
history = Pkg.dir("Meganet")*"/benchmarks/micro/bm_sparseKernel.jld"
nK = [100; 80]
nT = 5000;
A = sprandn(nK[1],nK[2],.5)
Q = sprandn(nnz(A),nT,.7)
K = getSparseKernel(Float64,A,Q)

# JthetaTmv
th  = initTheta(K)
dth = initTheta(K)
nex = 1
Y  = randn(nFeatIn(K),nex)+nex
Z  = randn(nFeatOut(K),nex)-nex

out = JthetaTmv(K, Z, th, Y)
out_old = Meganet.JthetaTmv_old(K, Z, th, Y)
@code_warntype JthetaTmv(K, Z, th, Y)
trial = @benchmark JthetaTmv(K, Z, th, Y)

# Save benchmarks
if true
    Meganet.updatehistory!(history, trial)
    hist = JLD.load(history, "hist")
    judge(hist)
end
