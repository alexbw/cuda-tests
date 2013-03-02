cuda-tests
==========

This is a playground for learning CUDA code. At this point, in `multi_gpu_raster_test.py`, along with the CUDA file `raster_test.cu`, I have implemented a forward-kinematic poser, skinning engine, rasterizer, and likelihood function that will take advantage of as many GPUs as there are in the system. It will be ported to hsmm-particlefilters, where it will live and be happy into perpetuity. 

Actually, I think that this project has served its function, and should be deleted once hsmm-particlefilters is folded back in. 
