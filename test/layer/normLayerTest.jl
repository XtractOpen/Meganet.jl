using Base.Test
using Meganet

npixel = 20
nex = 8
nchannel = 3
for TYPE=[Float64,Float32]
    @testset "normLayer $TYPE" begin
    for k=1:3
     L     = getNormLayer(TYPE,[npixel,nchannel,nex],k)
     testAbstractMeganetElement(L)
    end
    end

    @testset "getTVNormLayer (no scaling) $TYPE" begin
    Lb = getTVNormLayer(TYPE,[npixel,nchannel,nex],isTrainable=false)
     testAbstractMeganetElement(Lb)
    end

    @testset "getBatchNormLayer (no scaling) $TYPE" begin
    Lb = getBatchNormLayer(TYPE,[npixel,nchannel,nex],isTrainable=false)
     testAbstractMeganetElement(Lb)
    end

    @testset "getTVNormLayer (with scaling) $TYPE" begin
    Lb = getTVNormLayer(TYPE,[npixel,nchannel,nex],isTrainable=true)
     testAbstractMeganetElement(Lb,out=false)
    end

    @testset "getBatchNormLayer (with scaling) $TYPE" begin
    Lb = getBatchNormLayer(TYPE,[npixel,nchannel,nex],isTrainable=true)
     testAbstractMeganetElement(Lb)
    end
end
