import numpy as np

def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    if type(data[0]) is list:
        n_rows = len(data[0])
    else:
        n_rows = data[0].shape[0]

    batches = n_rows / size + 1
    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        end = min(end, n_rows)
        if start == end:
            break
        if len(data) == 1:
            ret = data[0][start:end]
            yield ret
        else:
            yield tuple([d[start:end] for d in data]) 

def iter_indices(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size + 1
    for b in range(batches):
        yield b

def shuffle(*data):
    idxs = np.random.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return data[0][idxs]
    else:
        return [d[idxs] for d in data]

def floatX(X):
    import theano
    return np.asarray(X, dtype=theano.config.floatX)

def intX(X):
    return np.asarray(X, dtype='int32')

def sharedX(X, dtype=None):
    import theano
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(np.asarray(X, dtype=dtype))

def downcast_float(X):
    return np.asarray(X, dtype=np.float32)

def case_insensitive_import(module, name):
    mapping = dict((k.lower(), k) for k in dir(module))
    return getattr(module, mapping[name])
