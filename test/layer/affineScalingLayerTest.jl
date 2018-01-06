using Base.Test
using Meganet

npixel   = 20
nex      = 12
nchannel = 3
L        = AffineScalingLayer([npixel,nchannel,nex])

@testset "affineScalingLayer" begin
    testAbstractMeganetElement(L)
end
