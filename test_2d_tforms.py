from numba_mini_ffts import numba_fft2_32, numba_rfft2_32

from pyCamSet.utils.general_utils import benchmark
import numpy as np

test = np.random.random((32,32))

l = lambda : numba_fft2_32(test)
my_fun = l()

l_real = lambda : numba_rfft2_32(test)
_ = l_real()

l_ref = lambda :np.fft.fft2(test)
correct = l_ref()

l_ref_real = lambda :np.fft.rfft2(test)
_ = l_ref_real()

test_n = 10000
print("numba fft")
benchmark(l, mode='us', repeats=test_n)
print("numba real fft")
benchmark(l_real, mode='us', repeats=test_n)
print("numpy fft")
benchmark(l_ref, mode='us', repeats=test_n)
print("numpy real fft")
benchmark(l_ref_real, mode='us', repeats=test_n)

assert np.all(np.isclose(correct, my_fun)), "failed to assert the answer was correct"
print("passed correctness check")
