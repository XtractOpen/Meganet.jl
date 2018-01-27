using Meganet, BenchmarkTools

history = Pkg.dir("Meganet")*"//benchmarks//micro//bm_batchnorm.jld"

TYPE = Float64

npixel = 500
nex = 1000
nchannel = 3

L = getNormLayer(TYPE,[npixel,nchannel,nex],3)
theta = initTheta(L)
Y     = randn(TYPE,nFeatIn(L),nex)

Yout2,Yout2,tmp2 = apply(L,theta,Y,true)

@code_warntype apply(L,theta,Y,true)

trial = @benchmark apply(L,theta,Y,true);

Meganet.updatehistory!(history, trial)
hist = JLD.load(history, "hist")
judge(hist)