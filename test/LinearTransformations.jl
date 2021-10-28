################################################################################
#
# Testcases for LinearTransformations.jl are added here.
#
################################################################################

using FluxHELayers.LinearTransformations

@testset "naive_matrix_multiply" begin
    SEALParams = setup_ckks(2^13, [40, 40, 40, 40, 40])
    initial_scale = 2.0^40

    vector = encrypt_array([1.0, 5.0, 9.0, 13.0], initial_scale, SEALParams)

    matrix = [1.0 2.0 3.0 4.0;
              5.0 6.0 7.0 8.0;
              9.0 10.0 11.0 12.0;
              13.0 14.0 15.0 16.0]

    output = naive_matrix_multiply(matrix,
                                   vector,
                                   SEALParams.evaluator,
                                   SEALParams.encoder,
                                   SEALParams.relinearization_keys)

    output = decrypt_array(output, SEALParams)

    @test output[1:4] ≈ [90, 202, 314, 426] atol=0.1
end

@testset "square_matrix_multiply" begin
    SEALParams = setup_ckks(2^13, [40, 40, 40, 40, 40])
    initial_scale = 2.0^40

    vector = encrypt_vector(repeat([1.0, 5.0, 9.0, 13.0], 1024),
                            initial_scale,
                            SEALParams)

    matrix = [1.0 2.0 3.0 4.0;
              5.0 6.0 7.0 8.0;
              9.0 10.0 11.0 12.0;
              13.0 14.0 15.0 16.0]

    output = square_matrix_multiply(matrix,
                                    vector,
                                    SEALParams.evaluator,
                                    SEALParams.encoder,
                                    SEALParams.galois_keys,
                                    SEALParams.relinearization_keys)

    output = decrypt_vector(output, SEALParams)

    @test output[1:4] ≈ [90, 202, 314, 426] atol=0.1
end

@testset "hybrid_matrix_multiply" begin
    SEALParams = setup_ckks(2^13, [40, 40, 40, 40, 40])
    initial_scale = 2.0^40

    vector = encrypt_vector(repeat([1.0, 5.0, 9.0, 13.0], 1024),
                            initial_scale,
                            SEALParams)

    matrix = [1.0 2.0 3.0 4.0;
              5.0 6.0 7.0 8.0]

    output = hybrid_matrix_multiply(matrix,
                                    vector,
                                    SEALParams.evaluator,
                                    SEALParams.encoder,
                                    SEALParams.galois_keys,
                                    SEALParams.relinearization_keys)

    output = decrypt_vector(output, SEALParams)

    @test output[1:2] ≈ [90, 202] atol=0.1
end
