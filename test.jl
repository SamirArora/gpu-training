using CUDA 

s = 10^8
cpu_array = randn(s)


for i in 1:3
    @time mapreduce(exp, +, cpu_array) 
end

gpu_array = CuArray(cpu_array)
@show typeof(gpu_array)
@show typeof(mapreduce(exp, +, gpu_array)) # NB: an actual Float64 is returned to CPU, so OK to skip CUDA.synchronize() --- see need_sync.jl for a different situation
for i in 1:3
    @time mapreduce(exp, +, gpu_array)
end


