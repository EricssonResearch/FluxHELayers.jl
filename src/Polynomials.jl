################################################################################
#
# The following file implements polynomials (in a hard coded, hacky way). The
# plaintext polynomials included here are hard-coded in order for Flux.jl GPU
# training to work properly without having to offload the operations to the
# CPU.
#
################################################################################

module Polynomials

    using Memoize
    using SEAL

    export PolynomialActivation

    """
        PolynomialActivation(degree::Integer, coeffs::Vector)

    Evaluates the polynomial of the provided degree and coefficients. The
    vector of coefficients should start from the 0:th degree up to the provided
    degree.
    """
    function PolynomialActivation(degree::Integer, coeffs::Vector)
        # TODO: do this in a better way.
        if degree == 2
            return Poly2(tuple(coeffs...))
        elseif degree == 3
            return Poly3(tuple(coeffs...))
        elseif degree == 4
            return Poly4(tuple(coeffs...))
        elseif degree == 5
            return Poly5(tuple(coeffs...))
        elseif degree == 6
            return Poly6(tuple(coeffs...))
        elseif degree == 7
            return Poly7(tuple(coeffs...))
        elseif degree == 8
            return Poly8(tuple(coeffs...))
        elseif degree == 9
            return Poly9(tuple(coeffs...))
        elseif degree == 10
            return Poly10(tuple(coeffs...))
        else
            throw(ArgumentError("""
            Sorry, I hard-coded this for GPU training to work.
            Pull requests with a fix are welcome :)
            """))
        end
    end

    struct Poly2
        coeffs::Tuple{Float32,
                      Float32,
                      Float32}
    end

    function (p::Poly2)(x::AbstractVecOrMat)
        p.coeffs[1] .+
        p.coeffs[2] .* x .+
        p.coeffs[3] .* x .^2
    end

    function (p::Poly2)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function.")
        else
            SEALParams = Main.SEALParams
        end

        @debug "Performing forward-pass of Poly2 layer"

        poly_log_eval(x, p.coeffs)
    end

    struct Poly3
        coeffs::Tuple{Float32,
                      Float32,
                      Float32,
                      Float32}
    end

    function (p::Poly3)(x::AbstractVecOrMat)
        p.coeffs[1] .+
        p.coeffs[2] .* x .+
        p.coeffs[3] .* x .^2 .+
        p.coeffs[4] .* x .^3
    end

    function (p::Poly3)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function.")
        else
            SEALParams = Main.SEALParams
        end

        @debug "Performing forward-pass of Poly3 layer"

        poly_log_eval(x, p.coeffs)
    end

    struct Poly4
        coeffs::Tuple{Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32}
    end

    function (p::Poly4)(x::AbstractVecOrMat)
        p.coeffs[1] .+
        p.coeffs[2] .* x .+
        p.coeffs[3] .* x .^2 .+
        p.coeffs[4] .* x .^3 .+
        p.coeffs[5] .* x .^4
    end

    function (p::Poly4)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function.")
        else
            SEALParams = Main.SEALParams
        end

        @debug "Performing forward-pass of Poly4 layer"

        poly_log_eval(x, p.coeffs)
    end

    struct Poly5
        coeffs::Tuple{Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32}
    end

    function (p::Poly5)(x::AbstractVecOrMat)
        p.coeffs[1] .+
        p.coeffs[2] .* x .+
        p.coeffs[3] .* x .^2 .+
        p.coeffs[4] .* x .^3 .+
        p.coeffs[5] .* x .^4 .+
        p.coeffs[6] .* x .^5
    end

    function (p::Poly5)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function.")
        else
            SEALParams = Main.SEALParams
        end

        @debug "Performing forward-pass of Poly5 layer"

        poly_log_eval(x, p.coeffs)
    end

    struct Poly6
        coeffs::Tuple{Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32}
    end

    function (p::Poly6)(x::AbstractVecOrMat)
        p.coeffs[1] .+
        p.coeffs[2] .* x .+
        p.coeffs[3] .* x .^2 .+
        p.coeffs[4] .* x .^3 .+
        p.coeffs[5] .* x .^4 .+
        p.coeffs[6] .* x .^5 .+
        p.coeffs[7] .* x .^6
    end

    function (p::Poly6)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function.")
        else
            SEALParams = Main.SEALParams
        end

        @debug "Performing forward-pass of Poly6 layer"

        poly_log_eval(x, p.coeffs)
    end

    struct Poly7
        coeffs::Tuple{Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32}
    end

    function (p::Poly7)(x::AbstractVecOrMat)
        p.coeffs[1] .+
        p.coeffs[2] .* x .+
        p.coeffs[3] .* x .^2 .+
        p.coeffs[4] .* x .^3 .+
        p.coeffs[5] .* x .^4 .+
        p.coeffs[6] .* x .^5 .+
        p.coeffs[7] .* x .^6 .+
        p.coeffs[8] .* x .^7
    end

    function (p::Poly7)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function.")
        else
            SEALParams = Main.SEALParams
        end

        @debug "Performing forward-pass of Poly7 layer"

        poly_log_eval(x, p.coeffs)
    end

    struct Poly8
        coeffs::Tuple{Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32}
    end

    function (p::Poly8)(x::AbstractVecOrMat)
        p.coeffs[1] .+
        p.coeffs[2] .* x .+
        p.coeffs[3] .* x .^2 .+
        p.coeffs[4] .* x .^3 .+
        p.coeffs[5] .* x .^4 .+
        p.coeffs[6] .* x .^5 .+
        p.coeffs[7] .* x .^6 .+
        p.coeffs[8] .* x .^7 .+
        p.coeffs[9] .* x .^8
    end

    function (p::Poly8)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function.")
        else
            SEALParams = Main.SEALParams
        end

        @debug "Performing forward-pass of Poly8 layer"

        poly_log_eval(x, p.coeffs)
    end

    struct Poly9
        coeffs::Tuple{Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32}
    end

    function (p::Poly9)(x::AbstractVecOrMat)
        p.coeffs[1] .+
        p.coeffs[2] .* x .+
        p.coeffs[3] .* x .^2 .+
        p.coeffs[4] .* x .^3 .+
        p.coeffs[5] .* x .^4 .+
        p.coeffs[6] .* x .^5 .+
        p.coeffs[7] .* x .^6 .+
        p.coeffs[8] .* x .^7 .+
        p.coeffs[9] .* x .^8 .+
        p.coeffs[10] .* x .^9
    end

    function (p::Poly9)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function.")
        else
            SEALParams = Main.SEALParams
        end

        @debug "Performing forward-pass of Poly9 layer"

        poly_log_eval(x, p.coeffs)
    end

    struct Poly10
        coeffs::Tuple{Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32,
                      Float32}
    end

    function (p::Poly10)(x::AbstractVecOrMat)
        p.coeffs[1] .+
        p.coeffs[2] .* x .+
        p.coeffs[3] .* x .^2 .+
        p.coeffs[4] .* x .^3 .+
        p.coeffs[5] .* x .^4 .+
        p.coeffs[6] .* x .^5 .+
        p.coeffs[7] .* x .^6 .+
        p.coeffs[8] .* x .^7 .+
        p.coeffs[9] .* x .^8 .+
        p.coeffs[10] .* x .^9 .+
        p.coeffs[11] .* x .^10
    end

    function (p::Poly10)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function.")
        else
            SEALParams = Main.SEALParams
        end

        @debug "Performing forward-pass of Poly10 layer"

        poly_log_eval(x, p.coeffs)
    end

    """
        poly_log_eval(x::AbstractVecOrMat, coeffs::Vector)

    Evaluates and returns the polynomial of 'x' defined by the passed in
    coefficients 'coeffs'. The evaluation is performed in a logarithmic
    multiplicative depth fashion (i.e. low number of consecutive multiplications).
    """
    function poly_log_eval(x::AbstractVecOrMat{T}, coeffs) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function")
        else
            SEALParams = Main.SEALParams
        end

        # The polynomial should not _only_ be a constant.
        @assert length(coeffs) > 1

        res = similar(x)

        constant, rest... = coeffs

        res_initalized = map(x -> x = false, Array{Bool}(undef, size(x)))

        # Go through the non-constant coefficients in reverse order:
        for (deg, coeff) in Iterators.reverse(enumerate(rest))
            x_i = power_log_eval(x, deg)

            # Calculate c * x_i
            for i in 1:length(x_i)
                p = Plaintext()
                encode!(p, Float64(coeff), scale(x_i[i]), SEALParams.encoder)
                mod_switch_to_inplace!(p, parms_id(x_i[i]), SEALParams.evaluator)

                c = Ciphertext()

                multiply_plain!(c, x_i[i], p, SEALParams.evaluator)
                relinearize_inplace!(c, SEALParams.relinearization_keys, SEALParams.evaluator)
                rescale_to_next_inplace!(c, SEALParams.evaluator)

                if res_initalized[i]
                    # Make sure that x1 and x2 has the same modulus.
                    c_m = chain_index(get_context_data(SEALParams.context, parms_id(c)))
                    res_m = chain_index(get_context_data(SEALParams.context, parms_id(res[i])))

                    if c_m < res_m
                        mod_switch_to_inplace!(res[i], parms_id(c), SEALParams.evaluator)
                    else
                        mod_switch_to_inplace!(c, parms_id(res[i]), SEALParams.evaluator)
                    end

                    # NOTE: useless scaling, but without it there's a bug in
                    # SEAL that returns NaN...
                    scale!(c, scale(c))

                    # NOTE: Hard-coded rescaling to make addition work :/
                    scale!(c, scale(res[i]))

                    add!(res[i], res[i], c, SEALParams.evaluator)
                else
                    res[i] = c
                    res_initalized[i] = true
                end
            end
        end

        # Add the constant coefficient:
        for i in 1:length(x)
            p = Plaintext()
            encode!(p, Float64(constant), scale(res[i]), SEALParams.encoder)
            mod_switch_to_inplace!(p, parms_id(res[i]), SEALParams.evaluator)

            add_plain!(res[i], res[i], p, SEALParams.evaluator)
        end

        return res
    end

    """
        power_log_eval(x::AbstractVecOrMat, degree::Integer)

    Evaluates and returns 'x' raised to the power of 'degree' in a logarithmic
    multiplicative depth fashion (i.e. low number of consecutive multiplications).
    """
    @memoize function power_log_eval(x::AbstractVecOrMat{T}, degree::Integer) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function")
        else
            SEALParams = Main.SEALParams
        end

        res = similar(x)

        # Base case (poor man's version of a deep clone since it's not available).
        # We use multiply for this (and thereby consume one multiplication) in
        # order to simplify scale handling for other functions.
        if degree == 1
            for i in 1:length(x)
                p = Plaintext()
                encode!(p, Float64(1), scale(x[i]), SEALParams.encoder)

                x_c = Ciphertext()

                multiply_plain!(x_c, x[i], p, SEALParams.evaluator)
                relinearize_inplace!(x_c, SEALParams.relinearization_keys, SEALParams.evaluator)
                rescale_to_next_inplace!(x_c, SEALParams.evaluator)

                res[i] = x_c
            end

            return res
        end

        # Base case:
        if degree == 2
            for i in 1:length(x)
                x_c = Ciphertext()

                square!(x_c, x[i], SEALParams.evaluator)
                relinearize_inplace!(x_c, SEALParams.relinearization_keys, SEALParams.evaluator)
                rescale_to_next_inplace!(x_c, SEALParams.evaluator)

                res[i] = x_c
            end

            return res
        end

        # Recursive case:
        split = Int(2^(ceil(log2(degree) - 1)))

        x1 = power_log_eval(x, split)
        x2 = power_log_eval(x, degree - split)

        for i in 1:length(x)
            x_c = Ciphertext()

            # Make sure that x1 and x2 has the same modulus.
            x1_m = chain_index(get_context_data(SEALParams.context, parms_id(x1[i])))
            x2_m = chain_index(get_context_data(SEALParams.context, parms_id(x2[i])))

            if x1_m < x2_m
                mod_switch_to_inplace!(x2[i], parms_id(x1[i]), SEALParams.evaluator)
            else
                mod_switch_to_inplace!(x1[i], parms_id(x2[i]), SEALParams.evaluator)
            end

            multiply!(x_c, x1[i], x2[i], SEALParams.evaluator)
            relinearize_inplace!(x_c, SEALParams.relinearization_keys, SEALParams.evaluator)
            rescale_to_next_inplace!(x_c, SEALParams.evaluator)

            res[i] = x_c
        end

        return res
    end
end
