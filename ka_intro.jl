using CUDA # loading CUDA will enable KernelAbstractions' CUDABackend(). Metal.jl for Mac GPU, etc.
using Random 
using KernelAbstractions
using Polyester
CUDA.seed!(1)
Random.seed!(1) 
include("ka_utils.jl")

# When we call "kernel(..)" below, implicitly, 
# the "sin_kernel!(..)" function gets called once for each index in the array
@kernel function sin_kernel!(array)
    i = @index(Global) # index for this call
    array[i] = sin(array[i])
end

function sin_ka(array, backend)
    work_array = copy_to_device(array, backend)
    n_tasks = length(array)
    kernel = sin_kernel!(backend) 
    @time begin 
        kernel(work_array, ndrange = n_tasks) # Note "ndrange" argument: it tells KA how many times to launch sin_kernel!
        KernelAbstractions.synchronize(backend) # avoid usual benchmarking trap
    end
    return work_array
end

function test_case(s = 10^8)

    for i in 1:3

        @show i
        x = randn(s)

        println("method 0: CPU naive")
        y = copy(x)
        @time begin
            for i = 1:s
                y[i] = sin(y[i])
            end
        end

        println("method 1: CUDA array programming technique")
        y = CuArray(x)
        @time begin
            sin.(y)
            CUDA.synchronize() 
        end

        println("method 2: KernelAbstraction (CPU)")
        y = copy(x)
        sin_ka(y, CPU())

        println("method 3: KernelAbstraction (GPU)")
        y = copy(x)
        sin_ka(y, CUDABackend())

        println("method 4: CPU, @threads")
        y = copy(x)
        @time begin
            Threads.@threads for i = 1:s # even with :static this is equally slow :( 
                y[i] = sin(y[i])
            end
        end

        println("method 5: CPU, @batch (from Polyester.jl)")
        y = copy(x)
        @time begin
            @batch for i = 1:s
                y[i] = sin(y[i])
            end
        end




    end
end

test_case()

