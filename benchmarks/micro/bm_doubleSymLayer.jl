using Meganet, BenchmarkTools

history = Pkg.dir("Meganet")*"//benchmarks//micro//bm_doubleSymLayer.jld"

TYPE = Float64

npixel = 500
nex = 1000

nchannel = 3
K      = getDenseKernel(TYPE,[4000,2000])
Bin    = randn(TYPE,nFeatOut(K),4)
Bout   = randn(TYPE,nFeatIn(K),3)
nLayer = getTVNormLayer(TYPE,[nex,4])
L      = getDoubleSymLayer(TYPE,K,nLayer,Bin,Bout)

theta = initTheta(L)

Y     = randn(TYPE,nFeatIn(L),nex)

# Warmup
Yout2,Yout2,tmp2 = apply(L,theta,Y,true)

@code_warntype apply(L,theta,Y,true)

trial = @benchmark apply(L,theta,Y,true);

Meganet.updatehistory!(history, trial)
hist = JLD.load(history, "hist")
judge(hist)