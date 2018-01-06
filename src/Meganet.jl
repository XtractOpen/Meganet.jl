module Meganet

using LinearOperators
using Base.Test

include("AbstractMeganetElement.jl")

include("activations/tanhActivation.jl")

include("integrators/NN.jl")
include("integrators/connector.jl")
include("integrators/ResNN.jl")
include("kernelTypes/denseKernel.jl")
include("kernelTypes/sparseKernel.jl")
include("kernelTypes/convFFTKernel.jl")

include("layers/affineScalingLayer.jl")
include("layers/normLayer.jl")
include("layers/doubleSymLayer.jl")
include("layers/singleLayer.jl")

include("loss/softMaxLoss.jl")
include("regularization/TikhonovReg.jl")

include("optimization/dnnBatchObjFctn.jl")
include("optimization/sgd.jl")

include("utils/getConvMatPeriodic.jl")
include("utils/testAbstractMeganetElement.jl")
include("utils/testLossFunction.jl")
include("utils/utilities.jl")
include("utils/checkDerivative.jl")
end
