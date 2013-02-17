import pycuda.driver as cuda
cuda.init() #init pycuda driver
current_dev = cuda.Device(0) #device we are working on
ctx = current_dev.make_context() #make a working context
ctx.push() #let context make the lead
ctx.pop() #deactivate again
ctx.detach() #delete it
