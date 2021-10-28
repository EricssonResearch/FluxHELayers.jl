################################################################################
#
# Convenience function used during testing.
#
################################################################################

function setup_ckks(poly_modulus_degree, primes)
    # Set the encryption scheme to CKKS.
    params = EncryptionParameters(SchemeType.ckks)

    # Set the degree of the cyclotomic polynomial to be a power of 2.
    set_poly_modulus_degree!(params, poly_modulus_degree)

    # Set the ciphertext coefficient modulus (a large integer) as the product of
    # large distinct prime numbers.
    set_coeff_modulus!(params, coeff_modulus_create(poly_modulus_degree, primes))

    # Now that all the parameters are set, we are ready to construct a SEALContext
    # object. This is a heavy class that checks the validity and properties of the
    # parameters we just set.
    context = SEALContext(params)

    # Generate the public and private keys by using the KeyGenerator, which
    # automatically generates the public and private keys for us.
    keygen = KeyGenerator(context)

    pub_key = PublicKey()
    create_public_key!(pub_key, keygen)

    pri_key = secret_key(keygen)

    relin_keys = RelinKeys()
    create_relin_keys!(relin_keys, keygen)

    galois_keys = GaloisKeys()
    create_galois_keys!(galois_keys, keygen)

    # Create an encryptor, evaluator and decryptor.
    encryptor = Encryptor(context, pub_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, pri_key)

    # Create a CKKSEncoder for encoding values and vectors.
    encoder = CKKSEncoder(context)

    (
        context = context,
        public_key = pub_key,
        private_key = pri_key,
        relinearization_keys = relin_keys,
        galois_keys = galois_keys,
        encryptor = encryptor,
        evaluator = evaluator,
        decryptor = decryptor,
        encoder = encoder
    )
end

function encrypt_array(arr, initial_scale, SEALParams)
    output = Array{Ciphertext}(undef, length(arr))

    for (idx, elem) in enumerate(arr)
        p = Plaintext()
        encode!(p, elem, initial_scale, SEALParams.encoder)

        c = Ciphertext()
        encrypt!(c, p, SEALParams.encryptor)

        output[idx] = c
    end

    output
end

function decrypt_array(arr, SEALParams)
    output = []

    for elem in arr
        p = Plaintext()
        decrypt!(p, elem, SEALParams.decryptor)

        m = similar(zeros(slot_count(SEALParams.encoder)))
        decode!(m, p, SEALParams.encoder)

        push!(output, m[1])
    end

    output
end

function encrypt_vector(arr, initial_scale, SEALParams)
    p = Plaintext()
    encode!(p, arr, initial_scale, SEALParams.encoder)

    c = Ciphertext()
    encrypt!(c, p, SEALParams.encryptor)

    c
end

function decrypt_vector(arr, SEALParams)
    p = Plaintext()
    decrypt!(p, arr, SEALParams.decryptor)

    m = similar(zeros(slot_count(SEALParams.encoder)))
    decode!(m, p, SEALParams.encoder)

    m
end
