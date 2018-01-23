using MAT
using Meganet

vars = matread("ECNN_compareCIFAR10.mat")
Y = vars["Y0"]
C = vars["C"]
theta = vars["theta"]
Z = vars["Z"]
timeM = vars["timeMAT"]

miniBatchSize = 50;
nImg = [32; 32]
cin  = 3
nc   = [16;32;64;64]
nt   = 2*[1;1;1]
h    = [1.;1.;1.]

# opening layer
K1 = getSparseConvKernel2D(nImg,[3,3,cin,nc[1]])
nL = getBatchNormLayer([prod(nImg);nc[1]],isTrainable=true)
blocks = [singleLayer(K1,nL)]

for k=1:length(nt)
    # ResNN layers
    K2 = getSparseConvKernel2D(nImg,[3,3,nc[k],nc[k]])
    nL = getBatchNormLayer([prod(nImg);nc[k]],isTrainable=true)
    L2 = DoubleSymLayer(K2,nL)
    RN  = ResNN(L2,nt[k],h[k])
    if k<length(nt)
        RN.outTimes *=0
    end
    blocks = [blocks;RN]

    # change channels
    Kc = getSparseConvKernel2D(nImg,[1,1,nc[k],nc[k+1]])
    nL = getBatchNormLayer([prod(nImg);nc[k+1]],isTrainable=true)
    blocks = [blocks; singleLayer(Kc,nL)]

    if k<length(nt)
        Kp = getAverageMatrix(nImg,nc[k+1])
        blocks = [blocks; Connector(Kp)]
        nImg ./=2
    end
end
# Connector block
B      = kron(speye(nc[end]),ones(prod(nImg)))/prod(nImg);
blocks = [blocks; Connector(B')];
blocks[end].outTimes=1

net = NN(blocks)

display(net)

@time Zj = apply(net,theta,Y[:,1:2],true)
# println("norm(Zj[1]-Z)/norm(Z)=$(norm(Zj[1]-Z)/norm(Z))")


# regularizers
pRegTh = TikhonovReg(alpha=4e-4)
pRegW = TikhonovReg(alpha=4e-4)
pLoss = SoftmaxLoss()
objFun = dnnObjFctn(net,pLoss,pRegTh,pRegW)
opt = SGD(learningRate=1e-2,maxEpochs=1,miniBatch=10,out=true)

W      = 0.1*vec(randn(10,nFeatOut(net)+1));
W = min.(W,.2);
W = max.(W,-.2);

solve(opt,objFun::dnnObjFctn,[vec(theta);vec(W)],Y,C,Y,C)

Profile.clear()
Profile.clear_malloc_data()
Profile.init(n = 10^7, delay = 0.01)
@profile solve(opt,objFun::dnnObjFctn,[vec(theta);vec(W)],Y,C,Y,C)

open("/tmp/EREsNN_CIFAR10.txt", "w") do s
    Profile.print(IOContext(s, :displaysize => (24, 500)))
end