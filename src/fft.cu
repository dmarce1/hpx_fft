#include <fft/fft.hpp>

#define BLOCK_SIZE 32

using real = double;

__device__ static void radix2(double* X, double* Y, int N, int M) {
	const int& tid = threadIdx.x;
	const int N2 = N >> 1;
	const int MN2 = M * N2;
	if (N > 2) {
		radix2(X, Y, N2, M);
		radix2(X + MN2, Y + MN2, N2, M);
	}
	const real twopioN = real(-2.0 * M_PI) / N;
	for (int k2 = 0; k2 < N2; k2++) {
		const real phi = twopioN * k2;
		const real cos1 = cos(phi);
		const real sin1 = sin(phi);
		for (int m = tid; m < M; m += BLOCK_SIZE) {
			const int& i0 = M * k2 + m;
			const int i1 = i0 + MN2;
			real& er0 = X[i0];
			real& er1 = X[i1];
			real& ei0 = Y[i0];
			real& ei1 = Y[i1];
			real tr1 = er1;
			er1 = er0 - cos1 * tr1 + ei1 * sin1;
			ei1 = ei0 - sin1 * tr1 - ei1 * cos1;
			er0 += er0 - er1;
			ei0 += ei0 - ei1;
		}
	}
}

__global__ void fft_1d_kernel(double* X, double* Y, int N, int M) {
	radix2(X, Y, N, M);
}

void cuda_fft_1d(double* Xh, double* Yh, int N, int M) {
	real* Xd;
	real* Yd;
	const size_t size = N * M * sizeof(real);
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));
	CUDA_CHECK(cudaMallocAsync(&Xd, size, stream));
	CUDA_CHECK(cudaMallocAsync(&Yd, size, stream));
	CUDA_CHECK(cudaMemcpyAsync(Xd, Xh, size, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(Yd, Yh, size, cudaMemcpyHostToDevice, stream));
	fft_1d_kernel<<<1, BLOCK_SIZE, 0, stream>>>(Xd, Yd, N, M);
	CUDA_CHECK(cudaMemcpyAsync(Xh, Xd, size, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(Yh, Yd, size, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaFreeAsync(Xd, stream));
	CUDA_CHECK(cudaFreeAsync(Yd, stream));
	while (cudaStreamQuery(stream) != cudaSuccess) {
		yield();
	}
	CUDA_CHECK(cudaStreamDestroy(stream));
}

