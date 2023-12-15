import torch


def adjust_learning_rate(method, base_lr, iters, max_iters, power):
    if method == 'poly':
        lr = base_lr * ((1 - float(iters) / max_iters) ** (power))
    else:
        raise NotImplementedError
    return lr


def adjust_learning_rate_use_stair(method, base_lr, iters, max_iters, power):
    if 5000 <= iters < 10000:
        base_lr = base_lr * 8/10
    elif 10000 <= iters < 15000:
        base_lr = base_lr * 6/10
    elif 15000 <= iters < 20000:
        base_lr = base_lr * 4/10
    elif 20000 <= iters < 25000:
        base_lr = base_lr * 2/10
    elif 25000 <= iters:
        base_lr = base_lr * 1/10

    if method=='poly':
        lr = base_lr * ((1 - float(iters) / max_iters) ** (power))
    else:
        raise NotImplementedError
    return lr


def adjust_learning_rate_use_stair2(method, base_lr, iters, max_iters, power):
    if 2000 <= iters < 3000:
        base_lr = base_lr * 9/10
    elif 3000 <= iters < 4000:
        base_lr = base_lr * 8/10
    elif 4000 <= iters < 5000:
        base_lr = base_lr * 7/10
    elif 5000 <= iters < 6000:
        base_lr = base_lr * 6/10
    elif 6000 <= iters < 7000:
        base_lr = base_lr * 5/10
    elif 7000 <= iters < 8000:
        base_lr = base_lr * 4/10
    elif 8000 <= iters < 9000:
        base_lr = base_lr * 3/10
    elif 9000 <= iters < 10000:
        base_lr = base_lr * 2/10
    elif iters >= 10000:
        base_lr = base_lr * 1/10

    if method=='poly':
        lr = base_lr * ((1 - float(iters) / max_iters) ** (power))
    else:
        raise NotImplementedError
    return lr