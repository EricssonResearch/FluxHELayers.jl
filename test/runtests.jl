################################################################################
#
# Quick and dirty testcases.
#
################################################################################

using Test
using SEAL

include("./utils.jl")

@testset "LinearTransformations" begin
    include("LinearTransformations.jl")
end

SEALParams = setup_ckks(2^14, [40, 40, 40, 40, 40, 40, 40, 40, 40, 40])

@testset "Polynomials" begin
    include("Polynomials.jl")
end
