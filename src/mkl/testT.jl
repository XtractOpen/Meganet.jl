using Meganet


nImage = [64,64];
sK = [3,3,32,16];
K = randn(Float32, tuple(sK...));
Y = randn(Float32, nImage[1]*nImage[2]*sK[4], 100);

Kernel = getConvGEMMKernel(Float32, nImage,sK);

@time ATY = ATmv(Kernel, K,Y);


YY = reshape(Y, nImage[1],nImage[2],sK[4], 100 );
@time ATY2 = convTMKL(K,YY);

