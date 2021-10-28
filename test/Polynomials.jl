################################################################################
#
# Testcases for Polynomials.jl are added here.
#
################################################################################

using Polynomials
using FluxHELayers.Polynomials: Poly2, Poly3, Poly4, Poly5, Poly6, Poly7, Poly8, Poly9, Poly10

@testset "Poly2" begin
    initial_scale = 2.0^40

    poly = Polynomial([10, 9, 8])
    poly_test = Poly2(tuple(poly.coeffs...))

    cases = [1.0, 1.25, 1.5, 1.75, 2.0]
    expected = map(x -> poly(x), cases)

    vector = encrypt_array(cases, initial_scale, SEALParams)
    output = poly_test(vector)
    output = decrypt_array(output, SEALParams)

    @test output[1:5] ≈ expected atol=0.1
end

@testset "Poly3" begin
    initial_scale = 2.0^40

    poly = Polynomial([10, 9, 8, 7])
    poly_test = Poly3(tuple(poly.coeffs...))

    cases = [1.0, 1.25, 1.5, 1.75, 2.0]
    expected = map(x -> poly(x), cases)

    vector = encrypt_array(cases, initial_scale, SEALParams)
    output = poly_test(vector)
    output = decrypt_array(output, SEALParams)

    @test output[1:5] ≈ expected atol=0.1
end

@testset "Poly4" begin
    initial_scale = 2.0^40

    poly = Polynomial([10, 9, 8, 7, 6])
    poly_test = Poly4(tuple(poly.coeffs...))

    cases = [1.0, 1.25, 1.5, 1.75, 2.0]
    expected = map(x -> poly(x), cases)

    vector = encrypt_array(cases, initial_scale, SEALParams)
    output = poly_test(vector)
    output = decrypt_array(output, SEALParams)

    @test output[1:5] ≈ expected atol=0.1
end

@testset "Poly5" begin
    initial_scale = 2.0^40

    poly = Polynomial([10, 9, 8, 7, 6, 5])
    poly_test = Poly5(tuple(poly.coeffs...))

    cases = [1.0, 1.25, 1.5, 1.75, 2.0]
    expected = map(x -> poly(x), cases)

    vector = encrypt_array(cases, initial_scale, SEALParams)
    output = poly_test(vector)
    output = decrypt_array(output, SEALParams)

    @test output[1:5] ≈ expected atol=0.1
end

@testset "Poly6" begin
    initial_scale = 2.0^40

    poly = Polynomial([10, 9, 8, 7, 6, 5, 4])
    poly_test = Poly6(tuple(poly.coeffs...))

    cases = [1.0, 1.25, 1.5, 1.75, 2.0]
    expected = map(x -> poly(x), cases)

    vector = encrypt_array(cases, initial_scale, SEALParams)
    output = poly_test(vector)
    output = decrypt_array(output, SEALParams)

    @test output[1:5] ≈ expected atol=0.3
end

@testset "Poly7" begin
    initial_scale = 2.0^40

    poly = Polynomial([10, 9, 8, 7, 6, 5, 4, 3])
    poly_test = Poly7(tuple(poly.coeffs...))

    cases = [1.0, 1.25, 1.5, 1.75, 2.0]
    expected = map(x -> poly(x), cases)

    vector = encrypt_array(cases, initial_scale, SEALParams)
    output = poly_test(vector)
    output = decrypt_array(output, SEALParams)

    @test output[1:5] ≈ expected atol=0.1
end

@testset "Poly8" begin
    initial_scale = 2.0^40

    poly = Polynomial([10, 9, 8, 7, 6, 5, 4, 3, 2])
    poly_test = Poly8(tuple(poly.coeffs...))

    cases = [1.0, 1.25, 1.5, 1.75, 2.0]
    expected = map(x -> poly(x), cases)

    vector = encrypt_array(cases, initial_scale, SEALParams)
    output = poly_test(vector)
    output = decrypt_array(output, SEALParams)

    @test output[1:5] ≈ expected atol=0.1
end

@testset "Poly9" begin
    initial_scale = 2.0^40

    poly = Polynomial([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    poly_test = Poly9(tuple(poly.coeffs...))

    cases = [1.0, 1.25, 1.5, 1.75, 2.0]
    expected = map(x -> poly(x), cases)

    vector = encrypt_array(cases, initial_scale, SEALParams)
    output = poly_test(vector)
    output = decrypt_array(output, SEALParams)

    @test output[1:5] ≈ expected atol=0.1
end

@testset "Poly10" begin
    initial_scale = 2.0^40

    poly = Polynomial([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1])
    poly_test = Poly10(tuple(poly.coeffs...))

    cases = [1.0, 1.25, 1.5, 1.75, 2.0]
    expected = map(x -> poly(x), cases)

    vector = encrypt_array(cases, initial_scale, SEALParams)
    output = poly_test(vector)
    output = decrypt_array(output, SEALParams)

    @test output[1:5] ≈ expected atol=0.1
end
