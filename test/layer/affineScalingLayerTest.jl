using Base.Test
using Meganet

npixel   = 20
nex      = 12
nchannel = 3
L        = getAffineScalingLayer(Float64,[npixel,nchannel,nex])

@testset "affineScalingLayer" begin
    testAbstractMeganetElement(L)
end
