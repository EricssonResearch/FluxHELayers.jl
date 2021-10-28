################################################################################
#
# The following file implements three different methods of performing linear
# transformations on CKKS encrypted vectors.
#
# The different methods that are implemented are:
#   - The naive method.
#   - The square matrix diagonal method by Halevi and Shoup.
#   - The rectangular matrix hybird method by Juvekar, Vaikuntanathan and
#   Chandrakasan.
#
# The documentation for each function should be read before use.
#
################################################################################

module LinearTransformations

    using SEAL

    export naive_matrix_multiply, square_matrix_multiply, hybrid_matrix_multiply

    """
        naive_matrix_multiply(matrix::AbstractMatrix,
                              vector::AbstractArray,
                              evaluator::Evaluator,
                              encoder::CKKSEncoder,
                              relinearization_keys::RelinKeys)

    Compute the matrix multiplication between the provided plaintext matrix and
    array of ciphertext values.

    In the naive approach each entry of the ciphertext vector is encrypted
    independently and element-wise multiplication between the matrix and vector
    entries are performed as usual. When adding the element-wise
    multiplications of a matrix row the addition is performed in a balanced
    (log₂(n)) fashion in order to account for the CKKS scaling differences that
    occur when performing multiplication.
    """
    function naive_matrix_multiply(matrix::AbstractMatrix,
                                   vector::AbstractArray,
                                   evaluator::Evaluator,
                                   encoder::CKKSEncoder,
                                   relinearization_keys::RelinKeys)
        rows = size(matrix, 1)
        cols = size(matrix, 2)

        @debug "Performing naive matrix multiplication" rows cols

        temp = Array{Ciphertext}(undef, rows)

        for row in 1:rows
            list = []

            @debug "Processing row $(row) / $(rows) of matrix"

            # Force the garbage collector to garbage collect the old list.
            GC.gc()

            for col in 1:cols
                m = matrix[row, col]
                c = vector[col]

                # Encode the matrix value with the same initial scale as the
                # ciphertext values scale and mod switch to the same level.
                m_plaintext = Plaintext()
                encode!(m_plaintext, Float64(m), scale(c), encoder)
                mod_switch_to_inplace!(m_plaintext, parms_id(c), evaluator)

                # Since the plaintext and ciphertext have the same scale and
                # are on the same level, we are now able to multiply them.
                r = Ciphertext()
                multiply_plain!(r, c, m_plaintext, evaluator)
                relinearize_inplace!(r, relinearization_keys, evaluator)
                rescale_to_next_inplace!(r, evaluator)

                push!(list, r)
            end

            temp[row] = sum_ciphertexts(list, evaluator)
        end

        temp
    end

    """
        square_matrix_multiply(matrix::AbstractMatrix,
                               vector::Ciphertext,
                               evaluator::Evaluator,
                               encoder::Encoder,
                               galois_keys::GaloisKeys,
                               relinearization_keys::RelinKeys)

    Compute the matrix multiplication between the provided square plaintext
    matrix and ciphertext vector.

    The diagonal matrix multiplication approach uses the method presented by
    Halevi and Shoup in "Algorithms in HElib" and does only work on square
    matrices where both the row and column count are a divisor of the slot
    count (power of 2) in order to handle rotations properly. As usual, the
    addition operations of the method are performed in a balanced (log₂(n))
    fashion.
    """
    function square_matrix_multiply(matrix::AbstractMatrix,
                                    vector::Ciphertext,
                                    evaluator::Evaluator,
                                    encoder::CKKSEncoder,
                                    galois_keys::GaloisKeys,
                                    relinearization_keys::RelinKeys)
        rows = size(matrix, 1)
        cols = size(matrix, 2)

        if rows != cols || ((rows != 0) && ((rows & (rows - 1)) != 0))
            throw(ArgumentError("Expected power 2 sized, square matrix, got: ($rows, $cols)."))
        end

        ciphertexts = diag_matrix_multiply(matrix,
                                           vector,
                                           evaluator,
                                           encoder,
                                           galois_keys,
                                           relinearization_keys)

        sum_ciphertexts(ciphertexts, evaluator)
    end

    """
        hybrid_matrix_multiply(matrix::AbstractMatrix,
                               vector::Ciphertext,
                               evaluator::Evaluator,
                               encoder::Encoder,
                               galois_keys::GaloisKeys,
                               relinearization_keys::RelinKeys)

    Compute the matrix multiplication between the provided rectangular
    plaintext matrix and ciphertext vector.

    The hybrid matrix multiplication approach uses the method presented by
    Juvekar, Vaikuntanathan and Chandrakasan in "GAZELLE: A Low Latency
    Framework for Secure Neural Network Inference" and does only work on
    rectangular matrices where the number of rows are fewer than the number of
    columns and both the row and column count are a divisor of the slot count
    (power of 2) in order to handle rotations properly. As usual, the addition
    operations of the method are performed in a balanced (log₂(n)) fashion.
    """
    function hybrid_matrix_multiply(matrix::AbstractMatrix,
                                    vector::Ciphertext,
                                    evaluator::Evaluator,
                                    encoder::CKKSEncoder,
                                    galois_keys::GaloisKeys,
                                    relinearization_keys::RelinKeys)
        rows = size(matrix, 1)
        cols = size(matrix, 2)

        if cols < rows || (rows != 0) && ((rows & (rows - 1)) != 0)
            throw(ArgumentError("Expected power 2 sized matrix with #rows <= #cols, got: ($rows, $cols)."))
        end

        ciphertexts = diag_matrix_multiply(matrix,
                                           vector,
                                           evaluator,
                                           encoder,
                                           galois_keys,
                                           relinearization_keys)

        temp = sum_ciphertexts(ciphertexts, evaluator)

        # The number of rotations is determined by the rectangular factor.
        rotations = cols / rows

        ciphertexts = []

        # Extract the vector rotations:
        for idx in 1:rotations
            r = Ciphertext()
            rotate_vector!(r, temp, ((idx - 1) * rows), galois_keys, evaluator)

            push!(ciphertexts, r)
        end

        sum_ciphertexts(ciphertexts, evaluator)
    end

    """
        diag_matrix_multiply(matrix::AbstractMatrix,
                             vector::Ciphertext,
                             evaluator::Evaluator,
                             encoder::CKKSEncoder,
                             galois_keys::GaloisKeys,
                             relinearization_keys::RelinKeys)

    Computes the element-wise multiplication between the (wrapped) diagonals of
    a matrix and a vector.
    """
    function diag_matrix_multiply(matrix::AbstractMatrix,
                                  vector::Ciphertext,
                                  evaluator::Evaluator,
                                  encoder::CKKSEncoder,
                                  galois_keys::GaloisKeys,
                                  relinearization_keys::RelinKeys)
        rows = size(matrix, 1)
        cols = size(matrix, 2)

        diagonals = []

        # Extract the diagonals.
        for offset in 0:(rows - 1)
            push!(diagonals, wrapped_diag(matrix, offset))
        end

        ciphertexts = []

        for (idx, diagonal) in enumerate(diagonals)
            d = Plaintext()
            encode!(d, diagonal, scale(vector), encoder)
            mod_switch_to_inplace!(d, parms_id(vector), evaluator)

            r = Ciphertext()
            rotate_vector!(r, vector, idx - 1, galois_keys, evaluator)

            temp = Ciphertext()
            multiply_plain!(temp, r, d, evaluator)
            relinearize_inplace!(temp, relinearization_keys, evaluator)
            rescale_to_next_inplace!(temp, evaluator)

            push!(ciphertexts, temp)
        end

        ciphertexts
    end

    """
        sum_ciphertexts(list:Array)

    Takes an even list of ciphertexts and adds them together.
    """
    function sum_ciphertexts(list::Array, evaluator::Evaluator)
        len = length(list)

        if len == 1
            return list[1]
        end

        result = Ciphertext()

        if len == 2
            add!(result, list[1], list[2], evaluator)
        else
            split = len ÷ 2

            sum1 = sum_ciphertexts(list[1:split], evaluator)
            sum2 = sum_ciphertexts(list[(split + 1):len], evaluator)

            add!(result, sum1, sum2, evaluator)
        end

        result
    end

    """
        wrapped_diag(matrix::AbstractMatrix, offset::Integer)

    Extracts the (wrapped) diagonal of the provided matrix.

    This function is intended to be used by the diagonal and hybrid matrix
    multiplication algorithms and is designed with that in mind (i.e. comes
    without warranty for any other case).
    """
    function wrapped_diag(matrix::AbstractMatrix, offset::Integer=0)
        rows = size(matrix, 1)
        cols = size(matrix, 2)

        if offset < 0 || rows <= offset
            throw(ArgumentError("Offset $(offset) out of range."))
        end

        row_idx = 0
        col_idx = offset

        indices = []

        for _ in 1:cols
            push!(indices, (((row_idx % rows) + 1), (col_idx % cols) + 1))

            row_idx += 1
            col_idx += 1
        end

        [matrix[row, col] for (row, col) in indices]
    end
end
