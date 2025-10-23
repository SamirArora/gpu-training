using CUDA 
using Random 
using KernelAbstractions
using Polyester
CUDA.seed!(1)
Random.seed!(1) 
include("ka_utils.jl")

@kernel function sin_kernel!(array)
    i = @index(Global)
    array[i] = sin(array[i])
end

function sin!(array, backend)
    work_array = copy_to_device(array, backend)
    n_tasks = length(array)
    kernel = sin_kernel!(backend) #, cpu_args(multi_threaded, n_tasks, backend)...)
    @time begin 
        kernel(work_array, ndrange = n_tasks) 
        KernelAbstractions.synchronize(backend) 
    end
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
        sin!(y, CPU(), false)

        println("method 3: KernelAbstraction (GPU)")
        y = copy(x)
        sin!(y, CUDABackend())

        println("method 4: CPU, @threads")
        y = copy(x)
        @time begin
            Threads.@threads for i = 1:s # with :static equally slow
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

