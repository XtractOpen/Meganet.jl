using Meganet, BenchmarkTools

history = Pkg.dir("Meganet")*"//benchmarks//micro//bm_connector.jld"

TYPE = Float64

npixel = 500
nex = 1000

K = randn(TYPE,10,5)
L = getConnector(TYPE,K,outTimes=1)

theta = initTheta(L)

Y     = randn(TYPE,nFeatIn(L),nex)

# Warmup
Yout,Yout,tmp = apply(L,theta,Y,true)

@code_warntype apply(L,theta,Y,true)

trial = @benchmark apply(L,theta,Y,true);

Meganet.updatehistory!(history, trial)
hist = JLD.load(history, "hist")
judge(hist)