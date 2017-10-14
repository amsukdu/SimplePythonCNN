import numpy as np


def softmax_loss(x, y):
    x = x.T
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[range(N), y])) / N
    dx = probs
    dx[range(N), y] -= 1
    dx /= N
    return loss, dx


def logistic_loss(x, y):
    N = x.shape[0]
    loss = np.sum(np.square(y - x) / 2) / N
    dx = -(y - x)
    return loss, dx.T


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def rmsprop(neurons, lr, l2_reg=0, decay_rate=0.9, eps=1e-8):
    for n in neurons:
        l2 = l2_reg * n.weights
        dx = (n.last_input.dot(n.delta)).T + l2
        d_bias = np.sum(n.delta)

        n.cache = decay_rate * n.cache + (1 - decay_rate) * (dx ** 2)
        n.weights += - lr * dx / (np.sqrt(n.cache) + eps)
        n.b -= lr * d_bias

def adam_update(neurons, lr, t, l2_reg=0, beta1=np.float32(0.9), beta2=np.float32(0.999), eps=1e-8):
    for n in neurons:
        l2 = l2_reg * n.weights
        dx = (n.last_input.dot(n.delta)).T + l2
        d_bias = np.sum(n.delta)

        n.m = beta1 * n.m + (1 - beta1) * dx
        n.v = beta2 * n.v + (1 - beta2) * (dx**2)

        m = n.m / np.float32(1-beta1**t)
        v = n.v / np.float32(1-beta2**t)

        n.weights -= lr * m / (np.sqrt(v) + eps)
        n.b -= lr * d_bias


def nag_update(neurons, lr, l2_reg=0, mu=np.float32(0.9)):
    for n in neurons:
        l2 = l2_reg * n.weights
        dx = (n.last_input.dot(n.delta)).T + l2
        d_bias = np.sum(n.delta)

        n.v_prev = n.v
        n.v = mu * n.v - lr * dx

        n.weights += -mu * n.v_prev + (1 + mu) * n.v
        n.b -= lr * d_bias


def momentum_update(neurons, lr, l2_reg=0, mu=np.float32(0.9)):
    for n in neurons:
        l2 = l2_reg * n.weights
        dx = (n.last_input.dot(n.delta)).T + l2
        d_bias = np.sum(n.delta)

        n.v = mu * n.v - lr * dx
        n.weights += n.v

        n.v_bias = mu * n.v_bias - lr * d_bias
        n.b += n.v_bias


def vanila_update(neurons, lr, l2_reg=0):
    for n in neurons:
        l2 = l2_reg * n.weights
        dx = (n.last_input.dot(n.delta)).T + l2
        d_bias = np.sum(n.delta)

        n.weights -= lr * dx + l2
        n.b -= lr * d_bias


def sigmoid(input):
    return 1/(1+np.exp(-input))


def relu(input):
    return np.maximum(0, input)


def sigmoid_d(input):
    return input * (1 - input)


def relu_d(input):
    return input > 0
