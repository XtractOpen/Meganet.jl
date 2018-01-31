using MAT, Meganet

n = 32;
Y_train,C_train,Y_test,C_test = getCIFAR10(n,Pkg.dir("Meganet")*"/data/CIFAR10/");

# using PyPlot
# y = Y_train[:,50]; y = y - minimum(y); y = y./maximum(y);
# y = reshape(y,32,32,3);
# y[:,:,1] = y[:,:,1]';y[:,:,2] = y[:,:,2]';y[:,:,3] = y[:,:,3]';
# figure(); imshow(y)

miniBatchSize = 32;
nImg = [32; 32]
cin  = 3
nc   = [16;32;64;64]
nt   = 2*[1;1;1]
h    = [1.;1.;1.]

TYPE = Float32;

getConvKernel = (nImg,sK) -> getConvGEMMKernel(TYPE,nImg,sK);
# getConvKernel = (nImg,sK) -> getConvFFTKernel(TYPE,nImg,sK);
#getConvKernel = (nImg,sK) -> getSparseConvKernel2D(TYPE,nImg,sK);

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

# @time Zj = apply(net,theta,Y_train[:,1:2],true)
# @time Zj = apply(net,theta,Y_train[:,1:miniBatchSize],true)


# regularizers
pRegTh = getTikhonovReg(TYPE;alpha=4e-4)
pRegW = getTikhonovReg(TYPE;alpha=4e-4)
pLoss = getSoftMaxLoss(TYPE);
objFun = dnnObjFctn(net,pLoss,pRegTh,pRegW)
opt = getSGDsolver(TYPE,learningRate=1e-2,maxEpochs=1,miniBatch=miniBatchSize,out=true)

W      = 0.1*vec(randn(TYPE,10,nFeatOut(net)+1));
W = min.(W,.2);
W = max.(W,-.2);
W = convert(Array{TYPE},W);

solve(opt,objFun::dnnObjFctn,[vec(theta);vec(W)],Y_train,C_train,Y_test,C_test)
@time solve(opt,objFun::dnnObjFctn,[vec(theta);vec(W)],Y_train,C_train,Y_test,C_test)

# Profile.clear()
# Profile.clear_malloc_data()
# Profile.init(n = 10^7, delay = 0.01)
# @profile solve(opt,objFun::dnnObjFctn,[vec(theta);vec(W)],Y,C,Y,C)

# open("/tmp/EREsNN_CIFAR10.txt", "w") do s
    # Profile.print(IOContext(s, :displaysize => (24, 500)))
# end
