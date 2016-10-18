## Mocha: convert torch model to/from caffe model easily.
Caffe: load with Python.  
Torch: load with Lua.

Q. How to connect Python & Lua code?  
A. We don't need directly interact Python code with Lua code, which is complex.
We can save the weights as `.npy` file to disk, and load it with `npy4th` package.
