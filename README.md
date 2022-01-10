# FluxHELayers.jl

**FluxHELayers.jl is not actively maintained.** See fork [svenanderzen/FluxHELayers.jl](https://github.com/svenanderzen/FluxHELayers.jl) for the most recent updates.


FluxHELayers.jl is a small library for performing neural network inference over
homomorphically encrypted data using [Microsoft
SEAL](https://github.com/microsoft/SEAL), the
[SEAL.jl](https://github.com/JuliaCrypto/SEAL.jl) Julia wrapper and the
[Flux.jl](https://github.com/FluxML/Flux.jl) library.

The layers which are currently implemented are:

- [Flux.Dense](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense)
- [Flux.BatchNorm](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.BatchNorm)

which should be enough to perform inference of simple MLP models. In order to
perform inference of other layers you will need to extend the library a bit,
but it should be fairly easy to do by just implementing a overloaded version of
the layer you want to use within `src/FluxHELayers.jl`.

As an example, implementing a convolutional layer (i.e. the `Flux.Conv` layer)
would look something along these lines:

```
function (conv::Conv)(x::AbstractVecOrMat{T}) where {T<:Ciphertext}
  // Your code for performing inference over homomorphically encrypted data.
end
```

### Matrix multiplication:

The library currently implements three types of matrix multiplication between a
plaintext matrix and ciphertext vector. The types implemented are:

- **Naive matrix multiplication:** which performs matrix multiplication between
  a plaintext matrix and a ciphertext vector by encrypting each entry in the
  vector as it's own ciphertext. While this is inefficient for single samples
  (e.g. one image of the MNIST dataset) it can be very efficient if multiple
  samples are batched together (i.e. having each entry in the vector be a CKKS
  ciphertext vector and operate on the data in a SIMD fashion).
- **Diagonal / Square matrix multiplication:** which performs matrix
  multiplication according to the method presented by Halevi and Shoup in
  "Algorithms in HElib" [1]. It only works on square matrices where both the
  row and column count are a divisor of the slot count (power of 2) in order to
  handle rotations properly but is able to utilize SIMD operations for single
  samples (e.g. one image of this MNIST dataset). This can substantially
  speed up the inference time and reduce the latency when operating over single
  data samples.
- **Hybrid matrix multiplication:** which performs matrix
  multiplication according to the method presented by Juvekar, Vaikuntanathan
  and Chandrakasan in "GAZELLE: A Low Latency Framework for Secure Neural
  Network Inference" [2] and only works on rectangular matrices where the
  number of rows are fewer than the number of columns and both the row and
  column count are a divisor of the slot count (power of 2) in order to handle
  rotations properly. While this has the same benefits that the diagonal /
  square method has, it allows for multiplication of rectangular matrices
  (which can be crucial in neural network applications).

Although the above methods are implemented (with test coverage), it's currently
only the naive matrix multiplication method that's used for inference since
that works for any size of matrices. If you want to use the other methods
(which you really should in case you're performing inference of single samples
and want low latency and good resource utilization), you'll need to tweak the
library a bit. This is surprisngly simple to do so don't be afraid to clone
this repository and to fiddle around with it :)

### Polynomial activation functions:

Neural networks are **useless** unless you are able to perform _non-linear
transformations_ . Normally, we use non-linear activation functions such as the
Rectified Linear Unit (ReLU) for this, but when we evaluate neural
networks over homomorphically encrypted data this is not (efficiently)
possible. For this reason, we often use _polynomial approximations_ of various
activation functions in order to be able to evaluate neural networks over
homomorphically encrypted data.

FluxHELayers.jl implements some functions for performing evaluations of
plaintext polynomials over ciphertext data. As always, this is done using plain
Julia method overloading of the methods with the signature `PolyX` where `X` is
the degree of the polynomial. We also include a `Square` (i.e. `x^2`)
activation function which can be used as an activation function.

## Installation

This package is not published anywhere, so in order to install it use the built
in package manager in Julia and install the library directly from the
repository.

```
pkg> add https://git.ailab.rnd.ki.sw.ericsson.se/esvnann/fluxhelayers.jl.git
```

You can also clone this library to you local machine and install it directly
from that location.

## Usage

In order to try out an example, start a Julia session and activate the
environment `./example` (by typing `] activate ./example` in the prompt) and
then run the following commands:

```julia
# Include the example file (see the source file for the full example):
include("example/mlp.jl")

# Train a 3 layer MLP model over the MNIST dataset using x^2 as the activation function.
train()

# Run inference over unencrypted / plaintext data.
test_plaintext()
Accuracy: 0.9697

# Run inference over encrypted / ciphertext data.
test_ciphertext()
2026.818505 seconds (30.30 M allocations: 2.061 GiB, 4.35% gc time, 0.01% compilation time)
Accuracy: 0.9697
```

In case you want some debugging output while you wait for the ciphertext
inference to work you can turn on debugging output by setting
`ENV["JULIA_DEBUG"] = Main` before calling the `test_ciphertext()` function.

## Tests
Some basic unit tests for the functionality in this library are implemented
using the `Test` Julia library and are placed in the `./test` folder. In order
to run the tests simply open a Julia session and run:

```
julia> ]

(@v1.6) pkg> activate .

(FluxHELayers) pkg> test
```

Note: press `]` in order to get to the built-in package manager (i.e. get the `pkg` prompt).

## Contributing

This library was done by me ([Sven
Anderzén](https://www.linkedin.com/in/svenanderzen)) for my master thesis on
end-to-end encrypted neural network inference. Since my thesis is over and I'm
now working full-time elsewhere I'm not able to provide any support for this library nor
accept any pull requests. In other words, feel free to fork the project and
take it from here :)

In case you're in need of support, want to know more about homomorphic
encryption or want to collaborate on a project in the future feel free to
connect with me on LinkedIn :)

## FAQ:

### This library looks a bit unfinished?
Yeah, sort of :) I've only implemented the things I needed in order to complete
my thesis, so there are a plethora of improvements that could be done to make
it more polished and feature complete. Feel free to pick it up and take it from
here!

### Why didn't you just use library X that supports PyTorch / Tensorflow?
Good question! While those options might be more "production ready" than this
trivial library with less than 500 lines of code, it's actually the few lines
of code that makes this library quite powerful (if I may say so myself). This
is not due to any stoke of genius from my end but solely due to the simplicity,
expressiveness and extensibility of the Julia language, Flux.jl machine
learning library and the SEAL.jl wrapper of the Microsoft SEAL library.

This makes this library great for research or proof-of-concept purposes where
you quickly want to implement, iterate and evaluate on your idea with as few
lines of code as possible. Furthermore, if you want to do interesting work
involving homomorphic encryption and Just-In-Time compilation, the Julia
language supports that perfectly using e.g. the
[Cassette.jl](https://github.com/JuliaLabs/Cassette.jl) library.

While this library does require some knowledge about the Microsoft SEAL API, it
more than makes up for that in being simple to work with. There's no C++ /
Python interoperability to deal with, no computational graph to take into
consideration and no TensorFlow or PyTorch API you need to understand. The only
thing you need to know is a tiny bit of Julia, what a function and method
overloading is and you can basically read, understand and extend this library
all within a days worth of work :)

But anyway, that's just my opinion and you're free to form your own :)

## References:

1. [Algorithms in HElib](https://www.shoup.net/papers/helib.pdf)
2. [GAZELLE: A Low Latency Framework for Secure Neural Network Inference](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-juvekar.pdf)
