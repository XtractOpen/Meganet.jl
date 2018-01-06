using Meganet
using Base.Test
using jInv.Utils

include("../../src/optimization/sgd.jl")

# network
nf = 18
nc = 3
nclass=10
nex = 40
K = DenseKernel([nf,nf])
Bout = randn(nf,3)
nLayer = getBatchNormLayer([div(nf,nc),nc],isTrainable=true)
L     = singleLayer(K,nLayer,Bout=Bout)
net = ResNN(L,8,1.0)

# regularizers
pRegTh = TikhonovReg(alpha=1e-5)
pRegW = TikhonovReg(alpha=1e-5)
# loss
pLoss = SoftmaxLoss()
# data
objFun = dnnObjFctn(net,pLoss,pRegTh,pRegW)

Y   = randn(nf,nex)/10
nw2 = (pLoss.addBias) ? nf+1 : nf
C   = full(sparse(full(rand(1:nclass,nex)),collect(1:nex),ones(nex),nclass,nex))

#initialize
theta = initTheta(net)
W     = randn(nclass,nw2)/10

opt = SGD(learningRate=1e-4,maxEpochs=10,miniBatch=2)
display(opt)

solve(opt,objFun::dnnObjFctn,[vec(theta);vec(W)],Y,C,Y,C)
