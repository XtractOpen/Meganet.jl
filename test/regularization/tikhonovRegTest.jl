using Meganet
using Base.Test


S = getTikhonovReg(Float64)

x = randn(10)
Sc,dS,d2S = regularizer(S,x)

@test norm(x-dS)/norm(x) < 1e-10
