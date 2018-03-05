using MAT, Meganet, JLD, BenchmarkTools, ProfileView
ENV["LD_LIBRARY_PATH"] = "/home/klensink/.juliapro/JuliaPro-0.6.2.1/JuliaPro/pkgs-0.6.2.1/v0.6/Meganet/src/mkl/"
BLAS.set_num_threads(1)

# Load data
srand(123)
n = 512;
Y_train,C_train,Y_test,C_test = getCIFAR10(n,Pkg.dir("Meganet")*"/data/CIFAR10/");

miniBatchSize = 64;
nImg = [32; 32]
cin  = 3
nc   = [16;32;64;64]
nt   = 2*[1;1;1]
h    = [1.;1.;1.]
TYPE = Float32
getConvKernel = (nImg,sK) -> getConvMKLKernel(TYPE,nImg,sK);

# first layer
k = getConvKernel(nImg, [3, 3, cin, 16])
nl = getTVNormLayer(TYPE, [prod(nImg); 16], isTrainable = true)
layer = getSingleLayer(TYPE, k, nl)
blocks = [layer]

# Connector
# B      = kron(speye(TYPE,16),ones(TYPE, prod(nImg)))/prod(nImg);
# blocks = [blocks; getConnector(TYPE, B')]
 blocks = [blocks; getConnector(TYPE, Y -> jumping_average(Y, prod(nImg)),
                                      W -> jumping_averageT(W, prod(nImg)),
                                      16*prod(nImg),
                                      16)]
blocks[end].outTimes=1

# Build Network
net = getNN(blocks)
theta = initTheta(net)
display(net)

# Setup classifier and regularization
pRegTh  = getTikhonovReg(TYPE;alpha=4e-4)
pRegW   = getTikhonovReg(TYPE;alpha=4e-4)
pLoss   = getSoftMaxLoss(TYPE)
objFun  = dnnObjFctn(net, pLoss, pRegTh, pRegW)
opt     = getSGDsolver(TYPE,learningRate=1e-2,
                            maxEpochs=1,
                            miniBatch=miniBatchSize,
                            out=true,
                            nesterov=true)

# Init classifier weights
W      = 0.1*vec(randn(TYPE,10,nFeatOut(net)+1));
W = min.(W,.2);
W = max.(W,-.2);
W = convert(Array{TYPE},W);


solve(opt, objFun, [vec(theta); vec(W)], Y_train, C_train, Y_test, C_test)

opt     = getSGDsolver(TYPE,learningRate=1e-2,
                            maxEpochs=10,
                            miniBatch=miniBatchSize,
                            out=true,
                            nesterov=true)

if false
    Profile.clear()
    Profile.init(n = 10000000)
    @profile solve(opt, objFun, [vec(theta); vec(W)], Y_train, C_train, Y_test, C_test)
    ProfileView.view()
else
    @benchmark solve(opt, objFun, [vec(theta); vec(W)], Y_train, C_train, Y_test, C_test)
end
