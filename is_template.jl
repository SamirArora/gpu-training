using CUDA 
using Random
CUDA.seed!(1)
Random.seed!(1)

# Assume for this tutorial that the proposal is a univariate std normal

function importance_sampling(log_target, std_normal_draws) 
    # TODO
    # Hint: use exp_normalize! defined below
    return (; log_norm, particles, weights = log_weights)
end

function integrate(is_output, test_fct = Base.identity) 
    # TODO
end

function exp_normalize!(log_weights)
    m = maximum(log_weights)
    log_weights .= exp.(log_weights .- m) 
    return m + log(normalize!(log_weights))
end 

function normalize!(weights) 
    s = sum(weights)
    weights .= weights ./ s 
    return s
end



## Test 

my_log_target(x) = -(x - 2)^2 / 2 


# check GPU and CPU give exactly the same approximation 
std_normals = randn(2) 
@show cpu_approx = importance_sampling(my_log_target, std_normals)
@show gpu_approx = importance_sampling(my_log_target, CuArray(std_normals))
@assert CUDA.@allowscalar cpu_approx.weights == gpu_approx.weights && CUDA.@allowscalar cpu_approx.log_norm == gpu_approx.log_norm

gpu(n_particles) = importance_sampling(my_log_target, CUDA.randn(Float64, n_particles)) 
cpu(n_particles) = importance_sampling(my_log_target, randn(Float64, n_particles)) 


# check we recover the correct mean of approx 2
n_particles = 10^8
@show integrate(gpu(n_particles))


# timings 
for i in 1:5 
    @time integrate(gpu(n_particles)) 
end

for i in 1:5 
    @time integrate(cpu(n_particles)) 
end
