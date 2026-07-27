import cupy as cp

# Fake input (1 image, 3 channels, 224x224)
x = cp.random.rand(1, 3, 224, 224)

# Simple convolution kernel
kernel = cp.random.rand(8, 3, 3, 3)  # 8 filters

def conv2d(x, k):
    batch, in_c, h, w = x.shape
    out_c, _, kh, kw = k.shape
    
    out = cp.zeros((batch, out_c, h-kh+1, w-kw+1))
    
    for b in range(batch):
        for oc in range(out_c):
            for ic in range(in_c):
                for i in range(h-kh+1):
                    for j in range(w-kw+1):
                        out[b, oc, i, j] += cp.sum(
                            x[b, ic, i:i+kh, j:j+kw] * k[oc, ic]
                        )
    return out

y = conv2d(x, kernel)

print("Output shape:", y.shape)