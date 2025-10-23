using CUDA 
using Random 
using KernelAbstractions
using Adapt
CUDA.seed!(1)
Random.seed!(1) 
include("ka_utils.jl")

struct BasicHMM{T, E}
    transition_matrix::T # K x K    pr(x'|x)
    emission_matrix::E   # K x T    log(p(y|x))
end
Adapt.@adapt_structure BasicHMM

buffers(::BasicHMM) = nothing

@kernel function forward_kernel!(
        @Const(current), # K x T 
        updated,         # K x T
        @Const(model), 
        buffers
        )

    _t = @index(Global) # from 1:(T-1)
    t = _t + 1          # from 2:T

    current_slice = @view current[:, t - 1]
    updated_slice = @view updated[:, t]

    update!(current_slice, updated_slice, model, t, buffers) 
    normalize!(updated_slice) 
end

function normalize!(weights)
    norm = sum(weights) 
    weights .= weights ./ norm
end

function update!(current_slice::AbstractVector{T}, updated_slice, model, t, buffers) where {T}
    error("implement this!")
end

function main(T, K, backend, F = Float64)
    transitions = KernelAbstractions.ones(backend, F, K, K)
    emissions = KernelAbstractions.ones(backend, F, K, T)
    model = BasicHMM(transitions, emissions)

    current = KernelAbstractions.ones(backend, F, K, T)
    updated = KernelAbstractions.ones(backend, F, K, T)

    n_tasks = T - 1 
    
    b = buffers(model)

    error("prep kernel, launch it an time it")
end


function test() 
    K = 20
    T = 10000
    for i in 1:3 
        @show i
        main(T, K, CPU())
        main(T, K, CUDABackend())
    end
end

test()