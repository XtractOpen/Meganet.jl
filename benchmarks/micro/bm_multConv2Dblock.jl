using Meganet, JLD, BenchmarkTools, Juno

history = Pkg.dir("Meganet")*"//benchmarks//micro//bm_multConv2Dblock.jld"
file = Pkg.dir("Meganet")*"/benchmarks/micro/vars_multConv2Dblock.jld"

x = load(file, "x")
K = load(file, "K")
y = load(file, "y")
t = load(file, "t")
shiftX = load(file, "shiftX")
shiftT = load(file, "shiftT")
imIdx = load(file, "imIdx")
doDerivative = load(file, "doDerivative")

trial = @benchmark Meganet.multConv2Dblock(x, K, y, t, shiftX, shiftT, imIdx, doDerivative = doDerivative)
@enter Meganet.multConv2Dblock(x, K, y, t, shiftX, shiftT, imIdx, doDerivative = doDerivative)
display(trial)
hist = load(history, "multConv2Dblock")

if false
    Meganet.updatehistory!(history, trial, "multConv2Dblock")
end

y = load(file, "y")
Meganet.multConv2Dblock(x, K, y, t, shiftX, shiftT, imIdx, doDerivative = doDerivative)
y = load(file, "y")
Profile.clear_malloc_data()
Meganet.multConv2Dblock(x, K, y, t, shiftX, shiftT, imIdx, doDerivative = doDerivative);
