using CUDA

x = CUDA.rand(10^8)

for i in 1:5 

    # BAD: No sync
    t1 = @elapsed sin.(x)

    # Correct way to benchmark
    t2 = @elapsed begin
        sin.(x)
        CUDA.synchronize() 
    end

    println("Without sync: $t1 seconds")
    println("   With sync: $t2 seconds")
end