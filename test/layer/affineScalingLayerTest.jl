using Base.Test
using Meganet

npixel   = 20
nex      = 12
nchannel = 3

for TYPE=[Float64,Float32]
    L        = getAffineScalingLayer(TYPE,[npixel,nchannel,nex])

    @testset "affineScalingLayer $TYPE" begin
        testAbstractMeganetElement(L)
    end
end
