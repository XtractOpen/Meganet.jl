module Meganet

using LinearOperators
using Base.Test

include("AbstractMeganetElement.jl")

include("activations/tanhActivation.jl")

include("Integrators/NN.jl")
include("Integrators/connector.jl")
include("Integrators/ResNN.jl")
include("KernelTypes/denseKernel.jl")
include("KernelTypes/sparseKernel.jl")
include("KernelTypes/convFFTKernel.jl")

include("Layers/affineScalingLayer.jl")
include("Layers/normLayer.jl")
include("Layers/doubleSymLayer.jl")
include("Layers/singleLayer.jl")

include("loss/softMaxLoss.jl")
include("regularization/TikhonovReg.jl")

include("optimization/dnnBatchObjFctn.jl")
include("optimization/sgd.jl")

include("Utils/getConvMatPeriodic.jl")
include("Utils/testAbstractMeganetElement.jl")
include("Utils/testLossFunction.jl")
include("Utils/utilities.jl")
include("Utils/checkDerivative.jl")
end
