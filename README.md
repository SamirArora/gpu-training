Get access to Sockeye

To setup your Sockeye environment, follow instructions at https://ubc-stat-ml.github.io/nf-nest-doc/02_setup.html

Login to Sockeye

Open an interactive gpu with `salloc --account=st-alexbou-1-gpu --partition=interactive_gpu --time=3:00:00 -N 1 -n 2 --mem=16G --gpus=1` (create an alias for that)

You should have 2 terminals: one in a head node (for Pkg), and one with a GPU

You can get info on the GPU via `nvidia-smi`, which gives us for example the runtime version:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.172.08             Driver Version: 570.172.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100-SXM2-16GB           On  |   00000000:1A:00.0 Off |                    0 |
| N/A   37C    P0             38W /  300W |      84MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2036      G   /usr/libexec/Xorg                        66MiB |
|    0   N/A  N/A            2133      G   /usr/bin/gnome-shell                     16MiB |
+-----------------------------------------------------------------------------------------+
```

On the login node:

```
julia --compiled-modules=no
using Pkg 
Pkg.activate(".")
Pkg.add("CUDA")
using CUDA 
CUDA.set_runtime_version!(v"12.8")
Pkg.precompile() # this will force Julia to download runtime artifact, the rest will fail because we are not in a GPU node, that's OK. 
                 # If you know a cleaner way let me know! 
```

Now move over to the GPU node. 

```
using Pkg 
Pkg.activate(".")
using CUDA 
```