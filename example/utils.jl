################################################################################
#
# CKKS scheme setup.
#
################################################################################

using SEAL

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

    # Check context parameters:
    print_parameters(context)
    println("Parameter validation: ", parameter_error_message(context))

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

function print_parameters(context::SEALContext)
    context_data = key_context_data(context)
    encryption_parms = parms(context_data)
    scheme_type = scheme(encryption_parms)
    if scheme_type == SchemeType.ckks
      scheme_name = "CKKS"
    elseif scheme_type == SchemeType.bfv
      scheme_name = "BFV"
    else
      error("unsupported scheme")
    end

    println("/")
    println("| Encryption parameters:")
    println("|   scheme: ", scheme_name)
    println("|   poly_modulus_degree: ", poly_modulus_degree(encryption_parms))

    print("|   coeff_modulus size: ", total_coeff_modulus_bit_count(context_data), " (")
    bit_counts = [bit_count(modulus) for modulus in coeff_modulus(encryption_parms)]
    print(join(bit_counts, " + "))
    println(") bits")

    if scheme_type == SchemeType.ckks
      println("|   plain_modulus: ", value(plain_modulus(encryption_parms)))
    end

    println("\\")
end
