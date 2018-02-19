using Meganet, BenchmarkTools

const history = Pkg.dir("Meganet")*"//benchmarks//micro//bm_resNN.jld"

TYPE = Float32
K = getConvGEMMKernel(TYPE,[96,96],[3,3,64,64])
nex = 20
# Bin = randn(TYPE,18,4)
# Bout = randn(TYPE,18,3)
nLayer = getBatchNormLayer(TYPE,[96*96,64],isTrainable=true)
lay     = getSingleLayer(TYPE,K,nLayer)

L = getResNN(TYPE,lay,4)

theta = initTheta(L)

Y     = randn(TYPE,nFeatIn(L),nex)

function benchmarkApply()
    funcName = "apply"
    Zd,Z,tmp  = apply(L,theta,Y,true);
    @code_warntype apply(L,theta,Y,true)

    trial = @benchmark apply(L,theta,Y,true)
    Meganet.updatehistory!(history, trial, funcName)
    hist = JLD.load(history, funcName)
    judge(hist)
end

###
function benchmarkJYTmv()
    funcName = "JYTmv"
    _,_,tmp = apply(L,theta,Y,true)
    Zout = randn(TYPE,nFeatOut(L),nex)

    #Warmup
    Z1 =  JYTmv(L,copy(Zout),(TYPE)[],theta,Y,tmp);
    trial = @benchmark JYTmv(L,copy(Zout),(TYPE)[],theta,Y,tmp)
    Meganet.updatehistory!(history, trial, funcName)
    hist = JLD.load(history, funcName)
    judge(hist)
end

function benchmarkJTmv()
    funcName = "JTmv"
    _,_,tmp = apply(L,theta,Y,true)
    Zout = randn(TYPE,nFeatOut(L),nex)

    #Warmup
    Z1 =  JTmv(L,copy(Zout),(TYPE)[],theta,Y,tmp);
    trial = @benchmark JTmv(L,copy(Zout),(TYPE)[],theta,Y,tmp)
    Meganet.updatehistory!(history, trial, funcName)
    hist = JLD.load(history, funcName)
    judge(hist)
end

function benchmarkJYmv()
    funcName = "JYmv"
    _,_,tmp = apply(L,theta,Y,true)
    Zout = randn(TYPE,nFeatOut(L),nex)

    #Warmup
    Z1 =  JYmv(L,copy(Zout),theta,Y,tmp);
    trial = @benchmark JYmv(L,copy(Zout),theta,Y,tmp)
    Meganet.updatehistory!(history, trial, funcName)
    hist = JLD.load(history, funcName)
    judge(hist)
end

function benchmarkJmv()
    funcName = "Jmv"
    _,_,tmp = apply(L,theta,Y,true)
    Zout = randn(TYPE,nFeatOut(L),nex)
    dtheta = randn(TYPE, size(theta))
    #Warmup
    Z1 =  Jmv(L,copy(dtheta),copy(Zout),theta,Y,tmp);
    trial = @benchmark Jmv(L,copy(dtheta),copy(Zout),theta,Y,tmp)
    Meganet.updatehistory!(history, trial, funcName)
    hist = JLD.load(history, funcName)
    judge(hist)
end

function benchmarkJthetamv()
    funcName = "Jthetamv"
    _,_,tmp = apply(L,theta,Y,true)
    dtheta = randn(TYPE, size(theta))

    #Warmup
    Z1 =  Jthetamv(L,dtheta,theta,Y,tmp);
    trial = @benchmark Jthetamv(L,dtheta,theta,Y,tmp)
    Meganet.updatehistory!(history, trial, funcName)
    hist = JLD.load(history, funcName)
    judge(hist)
end