using MAT
using Meganet

nImg = vec([32 32])
sK  =  vec([3 3 32 32])
nex = 100;

K1 = getConvGEMMKernel(Float64,nImg,sK)
K2 = getConvFFTKernel(Float64,nImg,sK)

theta = randn(nTheta(K1));

Y = zeros(tuple([nImg;sK[3];nex]...))
Y[2:end-1,2:end-1,:] = randn(tuple([nImg-2;sK[3];nex]...));

t1 = Amv(K1,theta,Y)
@time t1 = Amv(K1,theta,Y)

t2 = Amv(K2,theta,Y)
@time t2 = Amv(K2,theta,Y)

println(norm(t1[:]-t2[:])/norm(t1[:]))