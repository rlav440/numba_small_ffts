import numpy as np
import numba 
from numba import njit
from copy import copy

def make_FMATS(N):
    return np.exp(-2j*np.pi*np.arange(N//2)/ N)

F32D = make_FMATS(32)
F16D = make_FMATS(16)
F08D = make_FMATS(8)
F04D = make_FMATS(4)
F02D = make_FMATS(2)

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

@njit(cache=True, forceinline=True)
def DIF_butterfly(d, a_loc, b_loc, T):
    temp = d[b_loc]
    d[b_loc] = (d[a_loc] - d[b_loc]) * T
    d[a_loc] += temp

@njit(cache=True, forceinline=True)
def do_level(d_local, fft_size, level, TWIDDLE):
    #level is also akin to the groupsize of the data fft'd
    num_groups = fft_size//level
    within_group_step = 2 * fft_size//level
    ifactor = fft_size//num_groups//2
    for i in range(num_groups): # the number of groups to run
        for j in range(level//2): # the number of twists within a group.
            i0 = 0 + i + j * within_group_step
            i1 = i0 + num_groups
            DIF_butterfly(d_local, i0, i1, TWIDDLE[int(i*ifactor)]) #gives the twiddle factor.

@njit(cache=True, forceinline=True)
def DIF_butterfly_vector(d, a_loc, b_loc, T):
    #does the transformation in column major order along a row
    v = d.shape[1]
    for i in range(v):
        temp = d[b_loc, i]
        d[b_loc, i] = (d[a_loc, i] - d[b_loc, i]) * T
        d[a_loc, i] += temp

@njit(cache=True, forceinline=True)
def do_level_v(d_local, fft_size, level, TWIDDLE):
    #level is also akin to the groupsize of the data fft'd
    num_groups = fft_size//level
    within_group_step = 2 * fft_size//level
    ifactor = fft_size//num_groups//2
    for i in range(num_groups): # the number of groups to run
        for j in range(level//2): # the number of twists within a group.
            i0 = 0 + i + j * within_group_step
            i1 = i0 + num_groups
            DIF_butterfly_vector(d_local, i0, i1, TWIDDLE[int(i*ifactor)]) #gives the twiddle factor.

@njit(cache=True, forceinline=True)
def do_final_v(d_local, fft_size, level): #no need for a twiddle here.
    #level is also akin to the groupsize of the data fft'd
    for j in range(level//2): # the number of twists within a group.
        i0 = j * 2
        i1 = i0 + 1
        v = d_local.shape[1]
        for i in range(v):
            d_local[i0, i] += d_local[i1, i]
        
@njit(cache=True)
def numba_fft_4(d):
    d_local = d + 0j    
    do_level(d_local, 4, 2, F04D)
    do_level(d_local, 4, 4, F04D)
    return d_local[perm_4]

@njit(cache=True)
def numba_fft_8(d):
    d_local = d + 0j    
    do_level(d_local, 8, 2, F08D)
    do_level(d_local, 8, 4, F08D)
    do_level(d_local, 8, 8, F08D)
    return d_local[perm_8]

@njit(cache=True)
def numba_fft_32(d):
    d_local = d + 0j    
    do_level(d_local, 32, 2, F32D)
    do_level(d_local, 32, 4, F32D)
    do_level(d_local, 32, 8, F32D)
    do_level(d_local, 32, 16, F32D)
    do_level(d_local, 32, 32, F32D)
    return d_local[perm_32]

@njit(cache=True)
def numba_fft2_32(d):
    d_local = d.T + 0j    
    do_level_v(d_local, 32, 2, F32D)
    do_level_v(d_local, 32, 4, F32D)
    do_level_v(d_local, 32, 8, F32D)
    do_level_v(d_local, 32, 16, F32D)
    do_level_v(d_local, 32, 32, F32D)
    d_local = d_local[perm_32]
    d_local = d_local.T
    do_level_v(d_local, 32, 2, F32D)
    do_level_v(d_local, 32, 4, F32D)
    do_level_v(d_local, 32, 8, F32D)
    do_level_v(d_local, 32, 16, F32D)
    do_level_v(d_local, 32, 32, F32D)
    return d_local[perm_32]


@njit(cache=True)
def numba_rfft2_32(d):
    d_local = d.T + 0j    
    do_level_v(d_local, 32, 2, F32D)
    do_level_v(d_local, 32, 4, F32D)
    do_level_v(d_local, 32, 8, F32D)
    do_level_v(d_local, 32, 16, F32D)
    do_final_v(d_local, 32, 32) #minor symmetrx exploitation
    d_local = d_local[perm_32]
    d_local = d_local.T[:, :16]
    do_level_v(d_local, 32, 2, F32D)
    do_level_v(d_local, 32, 4, F32D)
    do_level_v(d_local, 32, 8, F32D)
    do_level_v(d_local, 32, 16, F32D)
    do_level_v(d_local, 32, 32, F32D)
    return d_local[perm_32]
