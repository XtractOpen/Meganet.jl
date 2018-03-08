using Meganet
# Note that MKL calls TV norm LRN (Local Responce Norm)

# Parameters
TYPE = Float32
npixel = 500
nex = 1000
nchannel = 3

# ----------- BATCH NORM -----------
# Setup norm layers and example data
bnorm   = getNormLayer(TYPE,[npixel,nchannel,nex],3)
theta   = initTheta(bnorm)
Y       = randn(TYPE,nFeatIn(bnorm),nex)
Z       = randn(TYPE,nFeatOut(bnorm),nex)

# Forward
Y_b, _, tmp = apply(bnorm, copy(theta), copy(Y), true) # Copy input just to make sure it can't be modified
# Backward
Z_b =  JYTmv(bnorm, copy(Z),(TYPE)[], copy(theta), copy(Y), copy(tmp))

# ----------- TV NORM -----------
# Setup norm layers and example data
TVnorm  = getNormLayer(TYPE,[npixel,nchannel,nex],2)
theta   = initTheta(TVnorm)
Y       = randn(TYPE,nFeatIn(TVnorm),nex)
Z       = randn(TYPE,nFeatOut(TVnorm),nex)

# Forward
Y_tv, _, tmp = apply(TVnorm, copy(theta), copy(Y), true)
# Backward
Z_tv =  JYTmv(TVnorm, copy(Z),(TYPE)[], copy(theta), copy(Y), copy(tmp))

# ----------- AVG POOLING -----------

# Params
nImg = [32; 32]
nc = 16
Y0 = rand(TYPE, 16384, 64)
W  = vcat([i*nc .+ (collect(1.0:64.0))' for i in 1:nc]...)

# Connector
B   = kron(speye(TYPE,16),ones(TYPE, prod(nImg)))/prod(nImg);
con = getConnector(TYPE, B')

# Forward
Y_pool = con.K * Y0
# Backward
Z_pool = con.K' * W
