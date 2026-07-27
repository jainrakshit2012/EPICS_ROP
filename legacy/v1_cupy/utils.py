import cupy as cp

def softmax(x):
    e = cp.exp(x - cp.max(x, axis=1, keepdims=True))
    return e / cp.sum(e, axis=1, keepdims=True)

def cross_entropy(pred, y):
    return -cp.mean(cp.log(pred[cp.arange(len(y)), y]))

def accuracy(preds, y):
    import cupy as cp
    predicted = cp.argmax(preds, axis=1)
    return cp.mean(predicted == y)