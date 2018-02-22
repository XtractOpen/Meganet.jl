using MAT, Meganet, BenchmarkTools, Compat, JLD

# Macro Benchmark on CIFAR10
n = 512
miniBatchSize = 64

path2data = "/home/klensink/Documents/cifar-10-batches-mat/"
history = Pkg.dir("Meganet")*"/benchmarks/CIFAR10/cifar10_512_64.jld"

Y_train,C_train,Y_test,C_test = getCIFAR10(n, path2data)

nImg = [32; 32]
cin  = 3
nc   = [16;32;64;64]
nt   = 2*[1;1;1]
h    = [1.;1.;1.]

TYPE = Float32;

getConvKernel = (nImg,sK) -> getConvGEMMKernel(TYPE,nImg,sK);

# opening layer
K1 = getConvKernel(nImg,[3,3,cin,nc[1]]);

nL = getBatchNormLayer(TYPE,[prod(nImg);nc[1]],isTrainable=true);
blocks = [getSingleLayer(TYPE,K1,nL)]

for k=1:length(nt)
    # ResNN layers
    K2 = getConvKernel(nImg,[3,3,nc[k],nc[k]])
    nL = getBatchNormLayer(TYPE,[prod(nImg);nc[k]],isTrainable=true)
    L2 = getDoubleSymLayer(TYPE,K2,nL)
    RN  = getResNN(TYPE,L2,nt[k],h[k])

    if k<length(nt)
        RN.outTimes *=0
    end
    blocks = [blocks;RN]
    # change channels
    Kc = getConvKernel(nImg,[1,1,nc[k],nc[k+1]]);

    nL = getBatchNormLayer(TYPE,[prod(nImg);nc[k+1]],isTrainable=true)
    blocks = [blocks; getSingleLayer(TYPE,Kc,nL)]

    if k<length(nt)
        Kp = getAverageMatrix(TYPE,nImg,nc[k+1])
        blocks = [blocks; getConnector(TYPE,Kp)]
        # nImg ./=2
	nImg = div.(nImg,2)
    end
end

# Connector block
B      = kron(speye(TYPE,nc[end]),ones(TYPE, prod(nImg)))/prod(nImg);
blocks = [blocks; getConnector(TYPE,B')];
blocks[end].outTimes=1

net = getNN(blocks)
theta = initTheta(net);

display(net)

# regularizers
pRegTh = getTikhonovReg(TYPE;alpha=4e-4)
pRegW = getTikhonovReg(TYPE;alpha=4e-4)
pLoss = getSoftMaxLoss(TYPE);
objFun = dnnObjFctn(net,pLoss,pRegTh,pRegW)
opt = getSGDsolver(TYPE,learningRate=1e-2,maxEpochs=1,miniBatch=miniBatchSize,out=true)

W      = 0.01*vec(randn(TYPE,10,nFeatOut(net)+1));
W = min.(W,.2);
W = max.(W,-.2);
W = convert(Array{TYPE},W);

# Save benchmarks
trial = @benchmark solve(opt,objFun::dnnObjFctn,[vec(theta);vec(W)],Y_train,C_train,Y_test,C_test)
if true
    Meganet.updatehistory!(history, trial)
end

hist = JLD.load(history, "hist")
judge(hist)
