#include <stdio.h>

// V1: Single channel, no batching, fairly naive shared memory implementation

// error checking macro
#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

const int DSIZE = 1 << 10;
const int KERNEL_SIZE = 3;
const int HALO = KERNEL_SIZE / 2;
const int BLOCK_SIZE = 16;

__global__ void conv2d(int *in, int *out, int *kernel)
{
    int gidx = threadIdx.x + blockIdx.x * blockDim.x + HALO;
    int gidy = threadIdx.y + blockIdx.y * blockDim.y + HALO;
    if (gidy + HALO > DSIZE || gidx + HALO > DSIZE) return;

    int lidx = threadIdx.x + HALO;
    int lidy = threadIdx.y + HALO;

    // Make local copy of input tile (larger than output tile)
    __shared__ int lkernel[KERNEL_SIZE][KERNEL_SIZE];
    __shared__ int lin[BLOCK_SIZE + 2 * HALO][BLOCK_SIZE + 2 * HALO];

    // Fill centre of tile
    lin[lidx][lidy] = in[gidx + DSIZE * gidy];

    // Fill left edge of tile
    if (threadIdx.x < HALO)
    {
        lin[lidx - HALO][lidy] = in[gidx - HALO + DSIZE * gidy];

        // Fill top left corner
        if (threadIdx.y < HALO)
        {
            lin[lidx - HALO][lidy - HALO] = in[gidx - HALO + DSIZE * (gidy - HALO)];
        }

        // Fill bottom left corner
        if (threadIdx.y >= blockDim.y - HALO)
        {
            lin[lidx - HALO][lidy + HALO] = in[gidx - HALO + DSIZE * (gidy + HALO)];
        }
    }

    // Fill right edge of tile
    if (threadIdx.x >= blockDim.x - HALO)
    {
        lin[lidx + HALO][lidy] = in[gidx + HALO + DSIZE * gidy];

        // Fill top right corner
        if (threadIdx.y < HALO)
        {
            lin[lidx + HALO][lidy - HALO] = in[gidx + HALO + DSIZE * (gidy - HALO)];
        }

        // Fill bottom right corner
        if (threadIdx.y >= blockDim.y - HALO)
        {
            lin[lidx + HALO][lidy + HALO] = in[gidx + HALO + DSIZE * (gidy + HALO)];
        }
    }

    // Fill top of tile
    if (threadIdx.y < HALO)
    {
        lin[lidx][lidy - HALO] = in[gidx + DSIZE * (gidy - HALO)];
    }

    // Fill bottom of tile
    if (threadIdx.y >= blockDim.y - HALO)
    {
        lin[lidx][lidy + HALO] = in[gidx + DSIZE * (gidy + HALO)];
    }

    // Make local copy of kernel
    if (threadIdx.x < KERNEL_SIZE && threadIdx.y < KERNEL_SIZE)
    {
        lkernel[threadIdx.x][threadIdx.y] = kernel[threadIdx.x + KERNEL_SIZE * threadIdx.y];
    }

    __syncthreads();

    // Do the calculation
    int value = 0;
    for (int i = 0; i < KERNEL_SIZE; ++i)
    {
        for (int j = 0; j < KERNEL_SIZE; ++j)
        {
            value += lkernel[i][j] * lin[lidx + i - HALO][lidy + j - HALO];
        }
    }

    // Store the result
    out[gidx + DSIZE * gidy] = value;
}

int main()
{
    int *h_in = new int[DSIZE * DSIZE];
    int *h_out = new int[DSIZE * DSIZE];
    int *h_kernel = new int[KERNEL_SIZE * KERNEL_SIZE];

    for (int i = 0; i < DSIZE * DSIZE; ++i)
    {
        h_in[i] = 1;
        h_out[i] = 0;
    }
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i)
    {
        h_kernel[i] = 1;
    }

    int *d_in;
    int *d_out;
    int *d_kernel;

    cudaMalloc(&d_in, DSIZE * DSIZE * sizeof(int));
    cudaMalloc(&d_out, DSIZE * DSIZE * sizeof(int));
    cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(int));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_in, h_in, DSIZE * DSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, DSIZE * DSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);
    conv2d<<<grid, block>>>(d_in, d_out, d_kernel);
    cudaCheckErrors("Kernel launch failure");

    cudaMemcpy(h_out, d_out, DSIZE * DSIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    // Test result
    for (int i = HALO; i < DSIZE - HALO; ++i)
    {
        for (int j = HALO; j < DSIZE - HALO; ++j)
        {
            if (h_out[i + DSIZE * j] != KERNEL_SIZE * KERNEL_SIZE)
            {
                printf("mismatch at index %i, %i, was: %i, should be: %i\n",
                       i, j, h_out[i + DSIZE * j], KERNEL_SIZE * KERNEL_SIZE);
                return -1;
            }
        }
    }

    return 0;
}