using Meganet
using Base.Test

# include("../../src/optimization/sgd.jl")

# network
nf = 18
nc = 3
nclass=10
nex = 40
K = getDenseKernel(Float64,[nf,nf])
Bout = randn(Float64,nf,3)
nLayer = getBatchNormLayer(Float64,[div(nf,nc),nc],isTrainable=true)
L     = getSingleLayer(Float64,K,nLayer,Bout=Bout)
net = getResNN(Float64,L,8,1.0)

# regularizers
pRegTh = getTikhonovReg(Float64,alpha=1e-5)
pRegW = getTikhonovReg(Float64,alpha=1e-5)
# loss
pLoss = getSoftMaxLoss(Float64)
# data
objFun = dnnObjFctn(net,pLoss,pRegTh,pRegW)

Y   = randn(nf,nex)/10
nw2 = (pLoss.addBias) ? nf+1 : nf
C   = full(sparse(full(rand(1:nclass,nex)),collect(1:nex),ones(nex),nclass,nex))

#initialize
theta = initTheta(net)
W     = randn(nclass,nw2)/10

opt = getSGDsolver(Float64,learningRate=1e-4,maxEpochs=10,miniBatch=2)
display(opt)

solve(opt,objFun::dnnObjFctn,[vec(theta);vec(W)],Y,C,Y,C)
