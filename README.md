## Mocha: convert torch model to/from caffe model easily.
Caffe: load with Python.  
Torch: load with Lua.

Q. How to connect Python & Lua code?  
A. We don't need directly call Python from Lua or the other wise, which is complex. We can save the weights as `.npy` file to disk. And load the weights using `npy4th`.
