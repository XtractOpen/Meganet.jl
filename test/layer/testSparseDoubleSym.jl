using Base.Test
using Meganet

numel = length

nImg   = [4 6]
nc1    = 4
nc2    = 2
nex    = 8
K      = getSparseConvKernel2D(nImg,[3,3,nc1,nc2])
Bin    = randn(nFeatOut(K),4)
Bout   = randn(nFeatIn(K),3)
nLayer = getTVNormLayer([prod(nImg),nc2])
L      = DoubleSymLayer(K,nLayer,Bin,Bout)
#L      = DoubleSymLayer(K,nLayer)

#@testset "doubleSymLayer (conv/TV)" begin
# testAbstractMeganetElement(L)
#end

theta     = initTheta(L)
Y         = randn(nFeatIn(L),nex)
dtheta    = randn(nTheta(L))
Z         = randn(nFeatOut(L),nex)

tmp = apply(L,theta,copy(Y),true)

J1 = zeros(numel(Z),numel(theta)); E1 = eye(numel(theta))
J2 = zeros(numel(theta),numel(Z)); E2 = eye(numel(Z))
for i=1:numel(theta)
    ei = zeros(numel(theta)); ei[i] = 1;
    J1[:,i] = vec(Jthetamv(L,ei,theta,copy(Y),tmp[3])[2])
end
for i=1:numel(Z)
    ei = zeros(numel(Z)); ei[i] = 1;
    ei = reshape(ei,size(Z))
    J2[:,i] = vec(JthetaTmv(L,ei,[],theta,copy(Y),tmp[3]))
end


Z1 =  Jthetamv(L,copy(dtheta),theta,copy(Y),tmp[3])[2]
Z2 =  JthetaTmv(L,copy(Z),[],theta,copy(Y),tmp[3])

t1 = dot(Z1,Z)
t2 = dot(Z2,dtheta)
#if out
    println("t1=$t1 \t t2=$t2 \t err = $(t1-t2)")
#end
