################################################################################
#
# Project root module.
#
################################################################################

module FluxHELayers

    include("LinearTransformations.jl")
    include("Polynomials.jl")

    using Flux
    using SEAL

    using .LinearTransformations: naive_matrix_multiply
    using .Polynomials

    export Square

    """
        (a::Dense)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}

    Performs a fully connected layer forward pass over a vector of individually
    encrypted ciphertexts.

    In what can be described as an ugly hack, this function uses a global variable
    called "SEALParams" which is an object that contains:
        - the SEAL CKKS evaluator, called "evaluator"
        - the SEAL CKKS encoder, called "encoder"
        - the SEAL CKKS relinearization keys, called "relinearization_keys"
    """
    function (a::Dense)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function.")
        else
            SEALParams = Main.SEALParams
        end

        @debug "Performing forward-pass of Dense layer"

        W, b = a.W, a.b

        # Compute the layer weight multiplication.
        result = naive_matrix_multiply(W,
                                       x,
                                       SEALParams.evaluator,
                                       SEALParams.encoder,
                                       SEALParams.relinearization_keys)

        # For each intermediate result, add the bias term.
        for i in 1:length(b)
            # Encode the bias term with the same initial scale as the
            # intermediate result ciphertext and mod switch to the same level.
            bias_plaintext = Plaintext()
            encode!(bias_plaintext, Float64(b[i]), scale(result[i]), SEALParams.encoder)
            mod_switch_to_inplace!(bias_plaintext, parms_id(result[i]), SEALParams.evaluator)

            # Add the bias plaintext to the intermediate result.
            add_plain_inplace!(result[i], bias_plaintext, SEALParams.evaluator)
        end

        result
    end

    """
        (BN::BatchNorm)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}

    Performs a Batch Norm forward pass over a vector of individually encrypted
    ciphertexts.

    In what can be described as an ugly hack, this function uses a global variable
    called "SEALParams" which is an object that contains:
        - the SEAL CKKS evaluator, called "evaluator"
        - the SEAL CKKS encoder, called "encoder"
        - the SEAL CKKS relinearization keys, called "relinearization_keys"
    """
    function (bn::BatchNorm)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function.")
        else
            SEALParams = Main.SEALParams
        end

        @debug "Performing forward-pass of BatchNorm layer"

        β, γ, μ, σ², ϵ = bn.β, bn.γ, bn.μ, bn.σ², bn.ϵ

        output = Array{Ciphertext}(undef, length(x))

        for i in 1:length(x)
            # Negate, encode and mod switch μ to calculate the numerator.
            μ_neg_plaintext = Plaintext()
            numerator_ciphertext = Ciphertext()
            encode!(μ_neg_plaintext, Float64(-μ[i]), scale(x[i]), SEALParams.encoder)
            mod_switch_to_inplace!(μ_neg_plaintext, parms_id(x[i]), SEALParams.evaluator)

            # Calculate the numerator.
            add_plain!(numerator_ciphertext, x[i], μ_neg_plaintext, SEALParams.evaluator)

            # Calculate and encode the denominator.
            denominator_plaintext = Plaintext()
            encode!(denominator_plaintext, Float64(1 / sqrt(σ²[i] + ϵ)), scale(x[i]), SEALParams.encoder)
            mod_switch_to_inplace!(denominator_plaintext, parms_id(x[i]), SEALParams.evaluator)

            # Calculate x.
            xi_ciphertext = Ciphertext()
            multiply_plain!(xi_ciphertext, numerator_ciphertext, denominator_plaintext, SEALParams.evaluator)
            relinearize_inplace!(xi_ciphertext, SEALParams.relinearization_keys, SEALParams.evaluator)
            rescale_to_next_inplace!(xi_ciphertext, SEALParams.evaluator)

            # Calculate the linear transformation w.r.t β.
            β_plaintext = Plaintext()
            encode!(β_plaintext, Float64(β[i]), scale(xi_ciphertext), SEALParams.encoder)
            mod_switch_to_inplace!(β_plaintext, parms_id(xi_ciphertext), SEALParams.evaluator)

            result_i_ciphertext = Ciphertext()
            add_plain!(result_i_ciphertext, xi_ciphertext, β_plaintext, SEALParams.evaluator)

            # Calculate the linear transforamtion w.r.t γ.
            γ_plaintext = Plaintext()
            encode!(γ_plaintext, Float64(γ[i]), scale(result_i_ciphertext), SEALParams.encoder)
            mod_switch_to_inplace!(γ_plaintext, parms_id(result_i_ciphertext), SEALParams.evaluator)

            multiply_plain_inplace!(result_i_ciphertext, γ_plaintext, SEALParams.evaluator)
            relinearize_inplace!(result_i_ciphertext, SEALParams.relinearization_keys, SEALParams.evaluator)
            rescale_to_next_inplace!(result_i_ciphertext, SEALParams.evaluator)

            output[i] = result_i_ciphertext
        end

        output
    end

    """
        Square(x::AbstractVecOrMat)

    Defines a square activation function that can be used for the forwards and
    backwards pass over normal values.
    """
    function Square(x::AbstractVecOrMat)
        x .^ 2
    end

    """
        Square(x::AbstractVecOrMat{T}) where {T<:Ciphertext}

    Performs a square activation function forward pass over a vector of
    individually encrypted ciphertexts.

    In what can be described as an ugly hack, this function uses a global variable
    called "SEALParams" which is an object that contains:
        - the SEAL CKKS evaluator, called "evaluator"
        - the SEAL CKKS encoder, called "encoder"
        - the SEAL CKKS relinearization keys, called "relinearization_keys"
    """
    function Square(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
        if !isdefined(Main, :SEALParams)
            throw("SEALParams needs to be defined to run this function.")
        else
            SEALParams = Main.SEALParams
        end

        @debug "Performing forward-pass of Square layer"

        for i in 1:length(x)
            square_inplace!(x[i], SEALParams.evaluator)
            relinearize_inplace!(x[i], SEALParams.relinearization_keys, SEALParams.evaluator)
            rescale_to_next_inplace!(x[i], SEALParams.evaluator)
        end

        x
    end

    """
        ReLU(x::AbstractVecOrMat)

    Defines a ReLU activation function that can be used for the forwards and
    backwards pass over plaintext values.
    """
    function ReLU(x::AbstractVecOrMat)
        relu.(x)
    end

    """
        ReLU(x::AbstractVecOrMat{T}) where {T<:Ciphertext}

    Throw a friendly reminder error, since we can't use the ReLU function
    on homomorphically encrypted data.
    """
    function ReLU(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
        throw(ArgumentError("ReLU on encryted data not available."))
    end
end
