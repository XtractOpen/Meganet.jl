using Meganet


nImage = [64,64];
sK = [3,3,32,64];
batch = 100
K = randn(Float32, tuple(sK...));
Y = randn(Float32, nImage[1]*nImage[2]*sK[3], batch)
Z = randn(Float32, nImage[1]*nImage[2]*sK[4], batch)

Kernel = getConvGEMMKernel(Float32, nImage,sK);

@time ADY = JthetaTmv(Kernel, Z, 0,Y);


YY = reshape(Y, nImage[1],nImage[2],sK[3], batch)
ZZ = reshape(Z, nImage[1],nImage[2],sK[4], batch)
@time ADY2 = convDerivMKL(sK, ZZ,YY);

