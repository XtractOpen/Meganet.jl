module Meganet

using LinearOperators, MAT, Base.Test, DistributedArrays

import JLD, BenchmarkTools

include("AbstractMeganetElement.jl")

include("activations/reluActivation.jl")

include("integrators/NN.jl")
include("integrators/connector.jl")
include("integrators/ResNN.jl")
include("kernelTypes/abstractConvKernel.jl");
include("kernelTypes/denseKernel.jl")
include("kernelTypes/sparseKernel.jl")
include("kernelTypes/convFFTKernel.jl");
include("kernelTypes/convGEMMKernel.jl");
include("kernelTypes/convCircKernel.jl");
include("kernelTypes/convMKLKernel.jl");



include("layers/affineScalingLayer.jl")
include("layers/normLayer.jl")
include("integrators/batchNormNN.jl")
include("layers/doubleSymLayer.jl")
include("layers/singleLayer.jl")

include("loss/softMaxLoss.jl")
include("regularization/TikhonovReg.jl")

include("optimization/dnnBatchObjFctn.jl")
include("optimization/sgd.jl")

include("utils/getConvMatPeriodic.jl")
include("utils/testAbstractMeganetElement.jl")
include("utils/testLossFunction.jl")
include("utils/utilities.jl");
include("utils/checkDerivative.jl");
include("utils/normalizeData.jl");

include("utils/getCIFAR10.jl");
include("utils/Benchmark.jl");


end
