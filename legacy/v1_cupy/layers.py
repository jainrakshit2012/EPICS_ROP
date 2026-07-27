import cupy as cp

def conv2d(x, w, b, stride=1):
    batch, in_c, h, w_in = x.shape
    out_c, _, kh, kw = w.shape

    out_h = (h - kh)//stride + 1
    out_w = (w_in - kw)//stride + 1

    col = cp.lib.stride_tricks.as_strided(
        x,
        shape=(batch, in_c, kh, kw, out_h, out_w),
        strides=x.strides[:2] + x.strides[2:4] + x.strides[2:4]
    )

    col = col.reshape(batch, in_c*kh*kw, out_h*out_w)
    w_col = w.reshape(out_c, -1)

    out = w_col @ col
    out = out + b.reshape(-1,1)

    return out.reshape(batch, out_c, out_h, out_w)


def relu(x):
    return cp.maximum(0, x)


def maxpool(x, size=2):
    import cupy as cp

    batch, c, h, w = x.shape

    # 🔥 Trim to even size
    h_trim = h - (h % size)
    w_trim = w - (w % size)

    x = x[:, :, :h_trim, :w_trim]

    out_h = h_trim // size
    out_w = w_trim // size

    x = x.reshape(batch, c, out_h, size, out_w, size)
    return x.max(axis=(3,5))


def linear(x, w, b):
    return x @ w + b