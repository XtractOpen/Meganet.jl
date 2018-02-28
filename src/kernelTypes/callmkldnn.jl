
# export MKLDNNROOT=/home/shekht/dnn/install
# export LD_LIBRARY_PATH=/home/shekht/dnn/install/lib

# To compile the code:
# g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib convolution.cpp convolutionT.cpp convolutionDeriv.cpp  -lmkldnn -lmklml_intel -liomp5 -O -fPIC -shared -o mkldnn.so
using Meganet

const mkldnnpath = "/home/moumita/Desktop/test/mkldnn.so"

function convMKL( K::Array{Float32,4}, Y::Array{Float32,4} )
   # Convoltion time vector
   # K should be nk x nk x n2 x n1
   # Y should be nImage2 x nImage1 x n2 x batch
   # Output is nimage2, nimage1, n1, batch
   
   
   nk, nk,  n2, n1 = size(K)
   nimage2, nimage1, n2_2, batch = size(Y)
   
   if nk != 3 ; error("nk != 3") ; end
   if n2 != n2_2 ; error("n2 != n2_2") ; end
   
   AY = Array{Float32}(nimage2, nimage1, n1, batch)
   
   ccall( (:Convolution, mkldnnpath),
          Void, ( Int32,Int32,Int32,Int32,Int32,Int32, Ref{Float32},Ref{Float32},Ref{Float32} ),
                  batch, nk, nimage1, nimage2, n1, n2,  Y, K,  AY)
                   
   return AY
end  # function convMKL

#----------------------------------------------------------------------------------

function convTMKL( K::Array{Float32,4}, Y::Array{Float32,4} )
   # Convoltion transpose time vector
   # K should be nk, nk,  n2, n1
   # Y should be nimage2, nimage1, n1, batch
   # Output is nimage2, nimage1, n2, batch
   
   
   nk, nk,  n2, n1 = size(K)
   nimage2, nimage1, n1_2, batch = size(Y)
   
   if nk != 3 ; error("nk != 3") ; end
   if n1 != n1_2 ; error("n1 != n1_2") ; end
   
   
   ATY = Array{Float32}(nimage2, nimage1, n2, batch)
   
   ccall( (:ConvolutionT, mkldnnpath),
          Void, ( Int32,Int32,Int32,Int32,Int32,Int32, Ref{Float32},Ref{Float32},Ref{Float32} ),
                  batch, nk, nimage1, nimage2, n1, n2, ATY, K,  Y)
                   
   return ATY
end  # function convTMKL

#----------------------------------------------------------------------------------

function convDerivMKL( sK::Array{Int}, Z::Array{Float32,4}, Y::Array{Float32,4} )
   # Z should be nimage2, nimage1, n1, batch
   # Y should be nimage2, nimage1, n2, batch
   # Output is  nk, nk,  n2, n1
   
   nk, nk,  n2, n1 = sK
   
   nimage2, nimage1, n2_2, batch = size(Y)
   nimage2, nimage1, n1_2, batch = size(Z)
   
   if nk != 3 ; error("nk != 3") ; end
   if n1 != n1_2 || n2 != n2_2 ; error("n1 != n1_2 || n2 != n2_2") ; end
   
   ADY = Array{Float32}(nk, nk,  n2, n1)
   
   ccall( (:ConvolutionDeriv, mkldnnpath),
          Void, ( Int32,Int32,Int32,Int32,Int32,Int32, Ref{Float32},Ref{Float32},Ref{Float32} ),
                  batch, nk, nimage1, nimage2, n1, n2, Y, ADY,  Z)
                   
   return ADY
end  # function convDerivMKL


nImage = [64,64];
sK = [3,3,32,16];
K = randn(Float32, tuple(sK...));
Y = randn(Float32, nImage[1]*nImage[2]*sK[4], 100);

Kernel = getConvGEMMKernel(Float32, nImage,sK);

@time ATY = ATmv(Kernel, K,Y);


YY = reshape(Y, nImage[1],nImage[2],sK[4], 100 );
@time ATY2 = convTMKL(K,YY);