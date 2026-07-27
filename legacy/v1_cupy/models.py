import cupy as cp
from layers import conv2d, relu, maxpool, linear

class CNN:
    def __init__(self):
        # Conv layers
        self.W1 = cp.random.randn(8, 3, 3, 3) * 0.01
        self.b1 = cp.zeros(8)

        self.W2 = cp.random.randn(16, 8, 3, 3) * 0.01
        self.b2 = cp.zeros(16)

        self.W3 = cp.random.randn(32, 16, 3, 3) * 0.01
        self.b3 = cp.zeros(32)

        # 🔥 Dynamically compute FC input size
        self.fc_input = self._get_fc_size()

        self.W4 = cp.random.randn(self.fc_input, 2) * 0.01
        self.b4 = cp.zeros(2)

    def _get_fc_size(self):
        # Dummy input to calculate size
        x = cp.random.rand(1, 3, 2048, 2048)

        x = conv2d(x, self.W1, self.b1, stride=2)
        x = relu(x)

        x = conv2d(x, self.W2, self.b2, stride=2)
        x = relu(x)

        x = maxpool(x)

        x = conv2d(x, self.W3, self.b3)
        x = relu(x)

        x = maxpool(x)

        # Flatten and get size
        return x.reshape(1, -1).shape[1]

    def forward(self, x):
        # 2048 → 1024
        x = conv2d(x, self.W1, self.b1, stride=2)
        x = relu(x)

        # 1024 → 512
        x = conv2d(x, self.W2, self.b2, stride=2)
        x = relu(x)

        # 512 → 256
        x = maxpool(x)

        # 256 → ~254
        x = conv2d(x, self.W3, self.b3)
        x = relu(x)

        # ~254 → ~127
        x = maxpool(x)

        x = x.reshape(x.shape[0], -1)

        out = linear(x, self.W4, self.b4)

        return out, x   # x = flattened features