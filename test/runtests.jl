@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

using NLPModels
using State
using Stopping
using CUTEst

include("test-unitaire-stopping-meta.jl")
print_with_color(:green, "StoppingMeta tests passed \n")
include("test-unitaire-ls-stopping.jl")
print_with_color(:green, "LineSearch stopping tests passed \n")
include("test-unitaire-nlp-stopping.jl")
print_with_color(:green, "Unconsmin test passed \n")
