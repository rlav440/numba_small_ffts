from pyCamSet.utils.general_utils import benchmark
import numpy as np
import numba 
from numba import njit
from copy import copy


#make the complex matrix 
N = 32
n = np.arange(N)
k = n.reshape((N, 1))
e = np.exp(-2j * np.pi * k * n / N)

## precompute the twiddle factors for pre-emptive laziness.
## maybe even map these out of complex values.
F32 = np.exp(-2j*np.pi*np.arange(N)/ N)
N = 16
F16 = np.exp(-2j*np.pi*np.arange(N)/ N)
N = 8
F08 = np.exp(-2j*np.pi*np.arange(N)/ N)
N = 4
F04 = np.exp(-2j*np.pi*np.arange(N)/ N)
N = 2
F02 = np.exp(-2j*np.pi*np.arange(N)/ N)

@njit(cache=True, forceinline=True)
def unwound_fft_vector_forwards(d, work_space):
    # x00, x01, x02,x03,x04,x05,x06,x07,x08,x09,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31 = data

    #can we reduce the memory usage?
    #store te data for F02 as a multiplication. can multiply, cache, just requires storing the data

    d_local = d
    level=2
    for i in range(32//level):
        work_space[:, 0] = d_local[:, 0 + i]
        work_space[:, 1] = d_local[:, 16 + i]
        d_local[:, 0 + i]  = work_space[0] + F02[0] * work_space[1]
        d_local[:, 16 + i]  = work_space[0] + F02[1] * work_space[1]

    level = 4
    for i in range(32//level):
        work_space[:, 0:level] = d_local[:, i::(32//level)]
        #even part
        d_local[:, 0 + i,]  = work_space[0] + F04[0] * work_space[1]
        d_local[:, 16 + i]  = work_space[0 + level//2] + F04[1] * work_space[1 + level//2]
        #odd part #USES THE SECOND PART OF THE PRECALCED FACTORS
        d_local[:,8 + i]  = work_space[0] + F04[0 + level//2] * work_space[1]
        d_local[:,24 + i]  = work_space[0 + level//2] + F04[1 + level//2] * work_space[1 + level//2]

    level = 8 
    for i in range(32//level):
        work_space[:, 0:level] = d_local[:, i::(32//level)]
        f = 32//level
        for j in range(level//2):
            #even part
            d_local[:, (2*j)*f + i]  = work_space[2*j] + F08[j] * work_space[2*j + 1]
            #odd part
            d_local[:, (2*j + 1)*f  + i]  = work_space[2*j] + F08[j + level//2] * work_space[2*j + 1]

    level=16
    for i in range(2): #32//level
        work_space[:, 0:level] = d_local[:, i::(32//level)]
        f = 32//level
        for j in range(level//2):
            #even part
            d_local[:, (2*j)*f + i]  = work_space[2*j] + F16[j] * work_space[2*j + 1]
            #odd part
            d_local[:, (2*j + 1)*f  + i]  = work_space[2*j] + F16[j + level//2] * work_space[2*j + 1]

    level=32 # no more iterations, only the final sum.
    work_space = d_local
    f = 1 #32//level
    for j in range(16):
        #even part
        d_local[:, (2*j)*f]  = work_space[2*j] + F32[j] * work_space[2*j + 1]
        #odd part
        d_local[:, (2*j + 1)*f]  = work_space[2*j] + F32[j + level//2] * work_space[2*j + 1]

@njit(cache=True, forceinline=True, looplift=True)
def unwound_fft_vector(d, work_space):
    # x00, x01, x02,x03,x04,x05,x06,x07,x08,x09,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31 = data

    #can we reduce the memory usage?
    #store te data for F02 as a multiplication. can multiply, cache, just requires storing the data

    d_local = d + 0j
    level=2
    for i in range(32//level): #-9us
        # work_space[0] = d_local[16 + i, :]
        # work_space[1] = d_local[16 + i, :]
        for u in range(32):
            d_local[0 + i, u] += F02[0] * d_local[16 + i, u]
            d_local[16 + i,u]  = d_local[0 + i,u] + F02[1] * d_local[16 + i, u]

    level = 4
    for i in range(32//level): # - 11 us
        work_space[0:level] = d_local[i::(32//level)]
        #even part
        for u in range(32):
            d_local[0 + i,u]  = work_space[0, u] + F04[0] * work_space[1, u]
            d_local[16 + i,u]  = work_space[0 + level//2, u] + F04[1] * work_space[1 + level//2, u]
            #odd part #USES THE SECOND PART OF THE PRECALCED FACTORS
            d_local[8 + i,u]  = work_space[0, u] + F04[0 + level//2] * work_space[1, u]
            d_local[24 + i,u]  = work_space[0 + level//2, u] + F04[1 + level//2] * work_space[1 + level//2, u]

    level = 8 
    for i in range(32//level): # - 10us
        work_space[0:level] = d_local[i::(32//level),:]
        f = 32//level
        for j in range(level//2):
            for u in range(32):
                #even part
                d_local[(2*j)*f + i, u]  = work_space[2*j, u] + F08[j] * work_space[2*j + 1, u]
                #odd part
                d_local[(2*j + 1)*f  + i, u]  = work_space[2*j, u] + F08[j + level//2] * work_space[2*j + 1, u]

    level=16
    for i in range(2): #32//level
        work_space[0:level] = d_local[i::(32//level),:]
        f = 32//level
        for j in range(level//2):
            for u in range(32):
                #even part
                d_local[(2*j)*f + i, u]  = work_space[2*j, u] + F16[j] * work_space[2*j + 1, u]
                #odd part
                d_local[(2*j + 1)*f + i, u] = work_space[2*j, u] + F16[j + level//2] * work_space[2*j + 1, u]

    level=32 # no more iterations, only the final sum.
    work_space = d_local
    f = 1 #32//level
    for j in range(16):
        for u in range(32):
            #even part
            d_local[(2*j)*f, u] = work_space[2*j, u] + F32[j] * work_space[2*j + 1, u]
            #odd part
            d_local[(2*j + 1)*f, u] = work_space[2*j, u] + F32[j + level//2] * work_space[2*j + 1, u]

@njit(cache=True, forceinline=True)
def unwound_fft(d, work_space):
    # x00, x01, x02,x03,x04,x05,x06,x07,x08,x09,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31 = data

    #can we reduce the memory usage?
    #store te data for F02 as a multiplication. can multiply, cache, just requires storing the data

    d_local = d + 0j    
    level=2
    for i in range(32//level):
        work_space[0] = d_local[0 + i]
        work_space[1] = d_local[16 + i]
        d_local[0 + i]  = work_space[0] + F02[0] * work_space[1]
        d_local[16 + i]  = work_space[0] + F02[1] * work_space[1]

    level = 4
    for i in range(32//level):
        work_space[0:level] = d_local[i::(32//level)]
        #even part
        d_local[0 + i]  = work_space[0] + F04[0] * work_space[1]
        d_local[16 + i]  = work_space[0 + level//2] + F04[1] * work_space[1 + level//2]
        #odd part #USES THE SECOND PART OF THE PRECALCED FACTORS
        d_local[8 + i]  = work_space[0] + F04[0 + level//2] * work_space[1]
        d_local[24 + i]  = work_space[0 + level//2] + F04[1 + level//2] * work_space[1 + level//2]

    level = 8 
    for i in range(32//level):
        work_space[0:level] = d_local[i::(32//level)]
        f = 32//level
        for j in range(level//2):
            #even part
            d_local[(2*j)*f + i]  = work_space[2*j] + F08[j] * work_space[2*j + 1]
            #odd part
            d_local[(2*j + 1)*f  + i]  = work_space[2*j] + F08[j + level//2] * work_space[2*j + 1]

    level=16
    for i in range(2): #32//level
        work_space[0:level] = d_local[i::(32//level)]
        f = 32//level
        for j in range(level//2):
            #even part
            d_local[(2*j)*f + i]  = work_space[2*j] + F16[j] * work_space[2*j + 1]
            #odd part
            d_local[(2*j + 1)*f  + i]  = work_space[2*j] + F16[j + level//2] * work_space[2*j + 1]

    level=32 # no more iterations, only the final sum.
    work_space[0:32] = d_local
    f = 1 #32//level
    for j in range(16):
        #even part
        d_local[(2*j)*f]  = work_space[2*j] + F32[j] * work_space[2*j + 1]
        #odd part
        d_local[(2*j + 1)*f]  = work_space[2*j] + F32[j + level//2] * work_space[2*j + 1]
    return d_local


# @njit(cache=True)
def matrix_fft_32(data:np.array):
    return e @ data 

@njit(cache=True)
def matrix_fft2_32(data):
    data = data + 0j
    for i in range(32):
        data[i, :] = matrix_fft_32(data[i, :])
    data = data.T.copy()
    for i in range(32):
        data[i, :] = matrix_fft_32(data[i, :])
    return data

# @njit(cache=True)
def unwound_fft2_32(data, workspace):
    data = data + 0j
    # for i in range(32):
    #     data[i, :] = unwound_fft(data[i, :], workspace[0])
    data = data.T
    unwound_fft_vector(data, workspace)
    # for i in range(32):
    #     data[i, :] = unwound_fft(data[i, :], workspace)
    data = data.T
    unwound_fft_vector(data, workspace)
    return data



r = np.random.random((32, 32))
workspace = np.empty((32), dtype=complex)
workspace2 = np.empty((32, 32), dtype=complex)
# l = lambda: unwound_fft(r[0], workspace)
# lt = lambda :unwound_fft_transpose(r, workspace, 0)
# lt()
l = lambda: unwound_fft2_32(r, workspace2)
l()
ln = lambda :np.fft.fft2(r)
ln()

benchmark(l, mode='us', repeats=10000)
# benchmark(lt, mode='us', repeats=1000)
# benchmark(ln, mode='us', repeats=1000)
