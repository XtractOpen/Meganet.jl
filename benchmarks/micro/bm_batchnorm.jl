using Meganet, BenchmarkTools

const history = Pkg.dir("Meganet")*"//benchmarks//micro//bm_batchnorm.jld"

const TYPE = Float64

npixel = 500
nex = 1000
nchannel = 3

const L = getNormLayer(TYPE,[npixel,nchannel,nex],3)
const theta = initTheta(L)
const Y     = randn(TYPE,nFeatIn(L),nex)


function benchmarkJYTmv()
    funcName = "JYTmv" #TODO: pass funcName to history instead of calling it "hist"
    _,_,tmp = apply(L,theta,Y,true)
    Zout = randn(TYPE,nFeatOut(L),nex)

    #Warmup
    Z1 =  JYTmv(L,copy(Zout),(TYPE)[],theta,Y,tmp)
    trial = @benchmark JYTmv(L,copy(Zout),(TYPE)[],theta,Y,tmp)
    Meganet.updatehistory!(history, trial, "hist")
    hist = JLD.load(history, "hist")
    judge(hist)
end


function benchmarkJYmv()
    funcName = "JYmv"
    _,_,tmp = apply(L,theta,Y,true)
    dY    = randn(TYPE,nFeatIn(L),nex)
    #Warmup
    dZ, dZ = JYmv(L,dY,theta,Y,tmp)
    # @code_warntype JYmv(L, dY, theta, Y, tmp)
    trial = @benchmark JYmv(L, dY, theta, Y, tmp)
    Meganet.updatehistory!(history, trial, "hist")
    hist = JLD.load(history, "hist")
    judge(hist)
end


function benchmarkApply(L, theta, Y, history)
    funcName = "apply"
    Yout2,Yout2,tmp2 = apply(L,theta,Y,true)

    @code_warntype apply(L,theta,Y,true)

    trial = @benchmark apply(L,theta,Y,true)

    Meganet.updatehistory!(history, trial, "hist")
    hist = JLD.load(history, "hist")
    judge(hist)
end
