using Meganet
using Base.Test

# include("../../src/optimization/sgd.jl")

# network
nf = 18
nc = 3
nclass=10
nex = 40
for TYPE=[Float64,Float32]
    K = getDenseKernel(TYPE,[nf,nf])
    Bout = randn(TYPE,nf,3)
    nLayer = getBatchNormLayer(TYPE,[div(nf,nc),nc],isTrainable=true)
    L     = getSingleLayer(TYPE,K,nLayer,Bout=Bout)
    net = getResNN(TYPE,L,8,1.0)

    # regularizers
    pRegTh = getTikhonovReg(TYPE,alpha=1e-5)
    pRegW = getTikhonovReg(TYPE,alpha=1e-5)
    # loss
    pLoss = getSoftMaxLoss(TYPE)
    # data
    objFun = dnnObjFctn(net,pLoss,pRegTh,pRegW)

    Y   = randn(TYPE,nf,nex)/10
    nw2 = (pLoss.addBias) ? nf+1 : nf
    C   = full(sparse(full(rand(1:nclass,nex)),collect(1:nex),ones(nex),nclass,nex))
    C  = convert.(TYPE,C)
    #initialize
    theta = initTheta(net)
    W     = randn(TYPE,nclass,nw2)/10

    opt = getSGDsolver(TYPE,learningRate=1e-4,maxEpochs=10,miniBatch=2)
    display(opt)

solve(opt,objFun::dnnObjFctn,[vec(theta);vec(W)],Y,C,Y,C)
end
