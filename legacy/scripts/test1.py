import cupy as cp

x = cp.random.rand(1000, 1000)
y = cp.dot(x, x)

print("GPU working:", y.shape)