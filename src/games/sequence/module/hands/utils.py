def take(iterator, size):
    for x in iterator:
        yield x
        size -= 1
        if not size:
            break
