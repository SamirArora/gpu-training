using CUDA 
using LinearAlgebra 

s = 10^3
a = CUDA.randn(Float64, s, s)
a = a'*a 
a = (a + a')/2
for _ in 1:3
    
    println("GPU chol")
    @time begin 
        cholesky(a)
        CUDA.synchronize() 
    end

    println("CPU chol")
    @time cholesky(Matrix(a));
end

nothing