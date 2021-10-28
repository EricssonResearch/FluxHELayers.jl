################################################################################
#
# A quick an dirty example of performing homomorphic inference of an MLP on the
# MNIST dataset.
#
################################################################################

using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using BSON: @save, @load
using CUDA
using MLDatasets

include("../src/FluxHELayers.jl")
include("./utils.jl")

@kwdef mutable struct Args
    η::Float64 = 3e-4       # learning rate
    batchsize::Int = 256    # batch size
    epochs::Int = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
end

function getdata(args, device = cpu)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32, dir = "$(@__DIR__)/mnist-dataset")
    xtest, ytest = MLDatasets.MNIST.testdata(Float32, dir = "$(@__DIR__)/mnist-dataset")

    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return train_loader, test_loader
end

function loss_and_accuracy(data_loader, model, device)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = device(x), device(y)
        ŷ = model(x)
        ls += logitcrossentropy(ŷ, y, agg=sum)
        acc += sum(onecold(ŷ) .== onecold(y))
        num +=  size(x)[end]
    end
    return ls / num, acc / num
end

function train(; kws...)
    args = Args(; kws...)

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    train_loader, test_loader = getdata(args, device)

    model = Chain(
        Dense(prod([28, 28, 1]), 128),
        FluxHELayers.Square,
        Dense(128, 128),
        FluxHELayers.Square,
        Dense(128, 10)
    )
    model = model |> device

    ps = Flux.params(model)

    opt = ADAM(args.η)

    for epoch in 1:args.epochs
        for (x, y) in train_loader
            x, y = device(x), device(y)
            gs = gradient(() -> logitcrossentropy(model(x), y), ps)
            Flux.Optimise.update!(opt, ps, gs)
        end

        train_loss, train_acc = loss_and_accuracy(train_loader, model, device)
        test_loss, test_acc = loss_and_accuracy(test_loader, model, device)
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
    end

    @save "$(@__DIR__)/model.bson" model
end

function test_plaintext(; kws...)
    args = Args(; kws...)

    @load "$(@__DIR__)/model.bson" model

    x_test, y_test = MLDatasets.MNIST.testdata(Float32, dir = "$(@__DIR__)/mnist-dataset")
    x_test = Flux.flatten(x_test)

    output = model(x_test)

    labels = Flux.onecold(output, 0:9)

    accuracy = sum(labels .== y_test) / length(y_test)

    println("Accuracy: $(accuracy)")
end

function test_ciphertext(; kws...)
    args = Args(; kws...)

    @load "$(@__DIR__)/model.bson" model

    x_test, y_test = MLDatasets.MNIST.testdata(Float64, dir = "$(@__DIR__)/mnist-dataset")
    x_test = Flux.flatten(x_test)

    cyclotomic_degree = 2^14
    primes = [40, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 40]
    initial_scale = 2.0^40
    vector_size = cyclotomic_degree ÷ 2

    # Initialize and set up the CKKS scheme.
    global SEALParams = setup_ckks(cyclotomic_degree, primes)

    # Matrix for storing the final plaintext output:
    output = Array{Float64, 2}(undef, 10, 10000)

    @time begin
        # Use ciphertext packing and perform the inference in batches:
        for range in Iterators.partition(1:size(x_test, 2), cyclotomic_degree ÷ 2)

            # Create the ciphertext vector:
            input = Array{Ciphertext}(undef, size(x_test, 1))

            for idx in 1:784
                @debug "Encrypting pixel $(idx)"

                pixels = x_test[idx, range]

                # Encode to plaintext:
                p = Plaintext()
                encode!(p, pixels, initial_scale, SEALParams.encoder)

                # Encrypt to ciphertext:
                c = Ciphertext()
                encrypt!(c, p, SEALParams.encryptor)

                input[idx] = c
            end

            @debug "Running FluxHELayers"

            # Run the model and get the encrypted output before the softmax layer:
            output_ciphertexts = model(input)

            for (idx, c) in enumerate(output_ciphertexts)
                @debug "Decrypting pixel $(idx)"

                # Decrypt to plaintext:
                p = Plaintext()
                decrypt!(p, c, SEALParams.decryptor)

                # Decode to message:
                m = similar(zeros(slot_count(SEALParams.encoder)))
                decode!(m, p, SEALParams.encoder)

                output[idx, range] = m[1:length(range)]
            end

            # Force garbage collector to run.
            input = nothing
            output_ciphertexts = nothing
            GC.gc()
        end
    end

    # Map the softmax function over each column.
    output = mapslices(softmax, output, dims=1)

    # Create labels from output:
    labels = Flux.onecold(output, 0:9)

    # Check the accuracy of the model:
    accuracy = sum(labels .== y_test) / length(y_test)

    println("Accuracy: $(accuracy)")
end
