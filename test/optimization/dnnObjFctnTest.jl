using Meganet
using Base.Test
using jInv.Utils


# network
nf = 18
nc = 3
nex = 4
K = DenseKernel([nf,nf])
Bout = randn(nf,3)
nLayer = getBatchNormLayer([div(nf,nc),nc],isTrainable=true)
L     = singleLayer(K,nLayer,Bout=Bout)
net = ResNN(L,4,.1)

# regularizers
pRegTh = TikhonovReg()
pRegW = TikhonovReg()
# loss
pLoss = SoftmaxLoss()
# data
Y   = randn(nf,nex)/10
nw2 = (pLoss.addBias) ? nf+1 : nf
C   = full(sparse(full(rand(1:nc,nex)),collect(1:nex),ones(nex),nc,nex))

#initialize
theta = initTheta(net)
W     = randn(nc,nw2)/10

objFun = dnnObjFctn(net,pLoss,pRegTh,pRegW)

@testset "dThLoss" begin
function testdThLoss(x,v=nothing)
        F,his,dF, = getMisfit(objFun,x,W,Y,C,true)
        if v!==nothing
            return F,dot(dF,v)
        else
            return F
        end
end
ckDer, = checkDerivative(testdThLoss,theta,out=false)
@test ckDer
end

@testset "dWLoss" begin
function testdWLoss(x,v=nothing)
        F,his,dFth,dF = getMisfit(objFun,theta,x,Y,C,true)
        if v!==nothing
            return F,dot(dF,v)
        else
            return F
        end
end
chkDer, = checkDerivative(testdWLoss,vec(W),out=false)
@test chkDer
end

@testset "dJ" begin
function testdJ(x,v=nothing)
        F,his,dF = evalObjFctn(objFun,x,Y,C,true)
        if v!==nothing
            return F,dot(dF,v)
        else
            return F
        end
end
chkDer, = checkDerivative(testdJ,[vec(theta);vec(W)],out=false)
@test chkDer
end
