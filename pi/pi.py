from multiprocessing import Queue, Pool
import sys
import mpmath
from mpmath import mpf

pi_queue = Queue()

def L(q: int) -> int:
    return (545140134 * q) + 13591409

def X(q: int) -> int:
    return (-262537412640768000) ** q

class MCalculate:
    def __init__(self):
        self.last_num = 1
        self.last_den = 1
        self.last_q = 0
    
    def __call__(self, q):
        num = self.last_num
        den = self.last_den
        if q > self.last_q:
            for i in range(self.last_q, q):
                num *= (12*i + 6) ** 3 - (192*i + 96)
                den *= (i + 1) ** 3
            self.last_num = num
            self.last_den = den
        elif q < self.last_q:
            for i in range(self.last_q - 1, q + 1, -1):
                num /= (12*i + 6) ** 3 - (192*i + 96)
                den /= (i + 1) ** 3
        # don't have to write anything for q == self.last_q because return will be same
        return num // den # avoid returning floats

def range_chunks(start, end, n): # TODO: calculate extra numbers
    # range [start, end)
    # n: number of chunks
    def custom_range(start, end, extra=None):
        for i in range(start, end):
            yield i
        if extra is not None:
            yield extra

    interval = (end - start) // n
    extras = iter(range(interval * n + 1, end + 1))
    # print(extras)
    for i in range(n):
        start_num = start + (interval * i)
        extra = None
        try:
            extra = next(extras)
        except StopIteration:
            pass

        yield iter(custom_range(start_num, start_num + interval, extra))

def pi_calculator(r):
    # r is is range to calculate
    M = MCalculate()
    pi_sum = 0
    for i in r:
        pi_sum += M(i) * mpf(L(i)) / X(i)
    # pi_queue.put(pi_sum)
    return pi_sum

def main():
    iters = 10
    mpmath.mp.dps = iters * 8
    constant = mpf(426880) * mpf(10005).sqrt()
    # with Pool(processes=2) as pool:
        # print(constant / sum(pool.map(pi_calculator, [*range_chunks(0, 10, 2)])))
    print(constant / pi_calculator([*range_chunks(0, iters, 1)][0]))

if __name__ == '__main__':
    main()
