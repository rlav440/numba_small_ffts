from pyCamSet.utils.general_utils import benchmark
import numpy as np
import numba 
from numba import njit
from copy import copy


#make the complex matrix 
def make_e(N):
    n = np.arange(N)
    k = n.reshape((N, 1))
    return np.exp(-2j * np.pi * k * n / N)
e32 = make_e(32)
e04 = make_e(4)
e02 = make_e(2)
## precompute the twiddle factors for pre-emptive laziness.
## maybe even map these out of complex values.

def make_FMATS(N):
    fbase = np.exp(-2j*np.pi*np.arange(N)/ N)
    fimpl = np.concatenate(
        [fbase[:N//2], (-fbase[:N//2] + fbase[N//2:])]
    )
    fDIF = np.exp(-2j*np.pi*np.arange(N//2)/ N)
    return fbase, fimpl, fDIF

F32, F32M, F32D = make_FMATS(32)
F16, F16M, F16D = make_FMATS(16)
F08, F08M, F08D = make_FMATS(8)
F04, F04M, F04D = make_FMATS(4)
F02, F02M, F02D = make_FMATS(2)


def generate_permutes2(n):
    return recursive_permuter2(np.arange(0,n), n)
def recursive_permuter2(pts, chunksize):
    "assuming you have an fft of the given size, attempts to work out how it is permuted"
    if chunksize==1:
        return pts
    #do the shuffle
    chunkstack = pts.reshape((-1, chunksize))
    for chunk in chunkstack:
        chunk_a = recursive_permuter2(chunk[0::2], chunksize//2)
        chunk_b = recursive_permuter2(chunk[1::2], chunksize//2)
        chunk[:] = np.concatenate((chunk_a, chunk_b))
    return pts 

def generate_permutes(n):
    return recursive_permuter(np.arange(0,n), n)
def recursive_permuter(pts, chunksize):
    "assuming you have an fft of the given size, attempts to work out how it is permuted"
    if chunksize==1:
        return

    #do the shuffle
    chunkstack = pts.reshape((-1, chunksize))
    for chunk in chunkstack:
        chunk_a = chunk[0::2]
        chunk_b = chunk[1::2]
        chunk[:] = np.concatenate((chunk_a, chunk_b))
    recursive_permuter(pts, chunksize//2)
    return pts 

perm_32 = generate_permutes(32)
perm_16 = generate_permutes(16)
perm_8 = generate_permutes(8)
perm_4 = generate_permutes(4)

perm2_32 = generate_permutes2(32)
perm2_8 = generate_permutes2(8)
perm2_4 = generate_permutes2(4)
print('previous')
print(perm_4)
print(perm_8)
print(perm_32)
print('double recursive')
print(perm2_4)
print(perm2_8)
print(perm2_32)

@njit(cache=True, forceinline=True)
def DIF_butterfly(d, a_loc, b_loc, T):
    temp = d[b_loc]
    d[b_loc] = (d[a_loc] - d[b_loc]) * T
    d[a_loc] += temp

# #@njit(cache=True
def unwound_fft_4_impl(d):

    d_local = d + 0j    
    for i in range(2):
        for j in range(1):
            d_local[0 + i] += F02M[0] * d_local[2 + i]
            d_local[2 + i]  = d_local[0 + i] + F02M[1] * d_local[2 + i]

    for i in range(1):
        d_local[0] += F04M[0] * d_local[1]
        d_local[1]  = d_local[0] + F04M[2] * d_local[1]

        d_local[2] += F04M[1] * d_local[3]
        d_local[3]  = d_local[2] + F04M[3] * d_local[3]

    return d_local[perm_4]


@njit(cache=True, forceinline=True)
def do_level_impl(d_local, fft_size, level, TWIDDLE):
    #level is also akin to the groupsize of the data fft'd
    num_groups = fft_size//level
    within_group_step = 2 * fft_size//level
    ifactor = fft_size//num_groups//2
    for i in range(num_groups): # the number of groups to run
        for j in range(level//2): # the number of twists within a group.
            i0 = 0 + i + j * within_group_step
            i1 = i0 + num_groups
            DIF_butterfly(d_local, i0, i1, TWIDDLE[int(i*ifactor)]) #gives the twiddle factor.

#@njit(cache=True)
def unwound_fft_4_baked(d):
    d_local = d + 0j    
    do_level_impl(d_local, 4, 2, F04D)
    do_level_impl(d_local, 4, 4, F04D)
    return d_local[perm_4]

#@njit(cache=True)
def unwound_fft_8_baked(d):
    d_local = d + 0j    
    print("8 val fft")
    do_level_impl(d_local, 8, 2, F08D)
    do_level_impl(d_local, 8, 4, F08D)
    do_level_impl(d_local, 8, 8, F08D)
    return d_local[perm_8]

@njit(cache=True)
def unwound_fft_32_baked(d):
    d_local = d + 0j    
    do_level_impl(d_local, 32, 2, F32D)
    do_level_impl(d_local, 32, 4, F32D)
    do_level_impl(d_local, 32, 8, F32D)
    do_level_impl(d_local, 32, 16, F32D)
    do_level_impl(d_local, 32, 32, F32D)
    return d_local[perm_32]

@njit(cache=True)
def unwound_fft_32_zippy(d):
    d_local = d + 0j    

    #DFT 32
    num_groups = 16
    for i in range(num_groups): # the number of groups to run
        i0 = 0 + i 
        i1 = i0 + num_groups
        DIF_butterfly(d_local, i0, i1, F32D[i]) #gives the twiddle factor.

    #DFT 16
    num_groups = 8
    within_group_step = 16
    ifactor = 2
    for i in range(num_groups): # the number of groups to run
        for j in range(2): # the number of twists within a group.
            i0 = 0 + i + j * within_group_step
            i1 = i0 + num_groups
            DIF_butterfly(d_local, i0, i1, F32D[int(i*ifactor)]) #gives the twiddle factor.

    #DFT 8
    num_groups = 4
    within_group_step = 8
    ifactor = 4
    for i in range(num_groups): # the number of groups to run
        for j in range(4): # the number of twists within a group.
            i0 = 0 + i + j * within_group_step
            i1 = i0 + num_groups
            DIF_butterfly(d_local, i0, i1, F32D[int(i*ifactor)]) #gives the twiddle factor.

    #DFT 4
    num_groups = 2
    within_group_step = 4
    ifactor = 8
    for i in range(num_groups): # the number of groups to run
        for j in range(8): # the number of twists within a group.
            i0 = 0 + i + j * within_group_step
            i1 = i0 + num_groups
            DIF_butterfly(d_local, i0, i1, F32D[int(i*ifactor)]) #gives the twiddle factor.

    #DFT 2
    num_groups = 1
    within_group_step = 2
    ifactor = 16
    for j in range(16): # the number of twists within a group.
        i0 = 0 + j * within_group_step
        i1 = i0 + num_groups
        DIF_butterfly(d_local, i0, i1, F32D[0]) #gives the twiddle factor.
    return d_local[perm_32]

def matrix_fft_32(data:np.array):
    return e32 @ data 

#@njit
def matrix_fft_02(data:np.array):
    return e02 @ data 
#@njit
def matrix_fft_04(data:np.array):
    return e04 @ data



test_size = 32
work_space = np.empty(test_size, dtype=complex)
test = np.arange(0,test_size)
test = np.random.random(test_size)
correct = np.fft.fft(test)
# matrix = matrix_fft_02(test)
# comp = unwound_fft_2(test, work_space).real


# my_fun = unwound_fft_4_baked(test)
# unwound_fft_4_impl(test)


fdict = {
    4:unwound_fft_4_baked,
    8:unwound_fft_8_baked,
    32:unwound_fft_32_baked,
}

my_fun = fdict[test_size](test) 
test_fn = fdict[test_size]
l = lambda : test_fn(test)
l()

l_comp = lambda :np.fft.fft(test)
l_zippy = lambda :unwound_fft_32_zippy(test)
l_comp()
l_zippy()

test_n = 10000
print("unwound fft")
benchmark(l, mode='us', repeats=test_n)
print("unwound fft - manually calc'd")
benchmark(l_zippy, mode='us', repeats=test_n)
print("numpy fft")
benchmark(l_comp, mode='us', repeats=test_n)



print("np ref", correct.real)
print("my fun", my_fun.real)
assert np.all(np.isclose(correct, my_fun)), "failed to assert the answer was correct"
