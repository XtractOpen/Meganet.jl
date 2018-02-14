struct Benchmark{T<:Union{BenchmarkTools.Trial, BenchmarkTools.BenchmarkGroup}}

    env::Base.EnvHash
    time::DateTime
    commit::String
    trial::T
end

function Benchmark(trial::BenchmarkTools.Trial)
    Benchmark(ENV, now(), readstring(`git log --pretty=format:'%h' -n 1`), trial)
end

"""
    Use: updatehistory!(history::String, trial::BenchmarkTools.Trial, funcName::String; pkg::Module = Meganet)

Appends `hist` in the JLD file `history` with the latest trial and metadata contained
in a `Benchmark` instance.
"""
function updatehistory!(history::String, trial::BenchmarkTools.Trial, funcName::String; pkg::Module = Meganet)

    cd(Pkg.dir("$pkg"))
    if isfile(history)
        println("Appending trial history: "*history)

        hist = JLD.load(history)
        if haskey(hist, funcName)
            histFunc = hist[funcName]
        else
            histFunc = Vector{Meganet.Benchmark}()
        end

        push!(histFunc, Meganet.Benchmark(trial))

        JLD.jldopen(history, "w") do file
            write(file, funcName, histFunc)
        end
    else
        println("Creating trial history: "*history)
        hist = Vector{Meganet.Benchmark}()
        push!(hist, Meganet.Benchmark(trial))

        JLD.jldopen(history, "w") do file
            write(file, funcName, hist)
        end
    end
end

function Base.show(io::IO, b::Meganet.Benchmark)
    println()
    println(b.time)
    println(b.trial)
end

function Base.show(io::IO, bv::Array{Meganet.Benchmark,1})
    show(io, bv[1])
    println("""
                ...
                    """)
    show(io, bv[end - 1])
    show(io, bv[end])
end

"""
    Use: judge(hist::Vector{Meganet.Benchmark}, i::Int, j::Int;
                                    estimator::Function = BenchmarkTools.median)

Judge between benchmark `i` and `j` in `hist` using `estimator`.
"""
function BenchmarkTools.judge(hist::Vector{Meganet.Benchmark}, i::Int, j::Int;
                                    estimator::Function = BenchmarkTools.median)
    length(hist) <= 1 && error("Only one benchmark in history")
    a, b = hist[i].trial, hist[j].trial
    j = BenchmarkTools.judge(estimator(a), estimator(b))

    return j
end

"""
    Use: judge(hist::Vector{Meganet.Benchmark}
                                    estimator::Function = BenchmarkTools.median)

Judge between the two most recent benchmarks in `hist` using `estimator`.
"""
function BenchmarkTools.judge(hist::Vector{Meganet.Benchmark};
                                    estimator::Function = BenchmarkTools.median)
    n = length(hist)
    n <= 1 && error("Only one benchmark in history")
    j = BenchmarkTools.judge(hist, n, n - 1)

    return j
end
