using Meganet
using Base.Test


# network
nf = 18
nc = 3
nex = 4
for TYPE=[Float64,Float32]
K = getDenseKernel(TYPE,[nf,nf])
Bout = randn(TYPE,nf,3)
nLayer = getBatchNormLayer(TYPE,[div(nf,nc),nc],isTrainable=true)
L     = getSingleLayer(TYPE,K,nLayer,Bout=Bout)
net = getResNN(TYPE,L,4,.1)

# regularizers
pRegTh = getTikhonovReg(TYPE)
pRegW = getTikhonovReg(TYPE)
# loss
pLoss = getSoftMaxLoss(TYPE)
# data
Y   = randn(TYPE,nf,nex)/10
nw2 = (pLoss.addBias) ? nf+1 : nf
C   = full(sparse(full(rand(1:nc,nex)),collect(1:nex),ones(nex),nc,nex))
C = convert.(TYPE,C)
#initialize
theta = initTheta(net)
W     = randn(TYPE,nc,nw2)/10

objFun = dnnObjFctn(net,pLoss,pRegTh,pRegW)

@testset "dThLoss $TYPE" begin
function testdThLoss(x,v=nothing)
        F,his,dF, = getMisfit(objFun,x,W,Y,C,[],true)
        if v!==nothing
            return F,dot(dF,v)
        else
            return F
        end
end
ckDer, = checkDerivative(testdThLoss,theta,out=false)
@test ckDer
end

@testset "dWLoss $TYPE" begin
function testdWLoss(x,v=nothing)
        F,his,dFth,dF = getMisfit(objFun,theta,x,Y,C,[],true)
        if v!==nothing
            return F,dot(dF,v)
        else
            return F
        end
end
chkDer, = checkDerivative(testdWLoss,vec(W),out=false)
@test chkDer
end

@testset "dJ $TYPE" begin
function testdJ(x,v=nothing)
        F,his,dF = evalObjFctn(objFun,x,Y,C,[],true)
        if v!==nothing
            return F,dot(dF,v)
        else
            return F
        end
end
chkDer, = checkDerivative(testdJ,[vec(theta);vec(W)],out=false)
@test chkDer
end
end
