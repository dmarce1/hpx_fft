#include <fft/fft.hpp>

#include <cmath>

void transpose_xy(double* x, int N) {
	std::vector<double> tmp(N);
	const int copy_sz = N * sizeof(double);
	for (int i = 0; i < N; i++) {
		for (int j = i + 1; j < N; j++) {
			double* xn = x + N * (j + N * i);
			double* xm = x + N * (i + N * j);
			std::memcpy(tmp.data(), xn, copy_sz);
			std::memcpy(xn, xm, copy_sz);
			std::memcpy(xm, tmp.data(), copy_sz);
		}
	}
}

void fft_3d_local(double* X, double* Y, int N) {
	const int N3 = N * N * N;
	// xyz
	scramble(X, N3);
	scramble(Y, N3);
	// zyx
	for( int i = 0; i < N; i++) {
		const int iii = N * N * i;
		fft_batch_1d(X + iii, Y + iii, N, N);
		transpose_re(X + iii, N, 1);
		transpose_re(Y + iii, N, 1);
		fft_batch_1d(X + iii, Y + iii, N, N);
	}
	// zxy
	transpose_hi(X, N);
	transpose_hi(Y, N);
	// xzy
	for( int i = 0; i < N; i++) {
		const int iii = N * N * i;
		fft_batch_1d(X + iii, Y + iii, N, N);
		transpose_re(X + iii, N, 1);
		transpose_re(Y + iii, N, 1);
		// xyz
	}
}
