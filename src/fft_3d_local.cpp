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
	const int log2N = std::ilogb(N);
	static std::vector<std::vector<double>> Wr;
	static std::vector<std::vector<double>> Wi;
	static std::vector<const double*> Wrptr;
	static std::vector<const double*> Wiptr;
	const int begin = Wr.size();
	Wr.resize(log2N + 1);
	Wi.resize(log2N + 1);
	Wrptr.resize(log2N + 1);
	Wiptr.resize(log2N + 1);
	for (int n = begin; n <= log2N; n++) {
		const int M = 1 << n;
		Wr[n].resize(M);
		Wi[n].resize(M);
		for (int m = 0; m < M; m++) {
			const auto W = std::polar(1.0, -2.0 * M_PI * m / M);
			Wr[n][m] = W.real();
			Wi[n][m] = W.imag();
		}
		Wrptr[n] = Wr[n].data();
		Wiptr[n] = Wi[n].data();
	}

	scramble_hi(X, N, N * N);
	scramble_hi(Y, N, N * N);
	for (int xi = 0; xi < N; xi++) {
		const int n = xi * N * N;
		scramble_hi(X + n, N, N);
		scramble_hi(Y + n, N, N);
		transpose_re(X + n, N, 1);
		transpose_re(Y + n, N, 1);
		scramble_hi(X + n, N, N);
		scramble_hi(Y + n, N, N);
		transpose_re(X + n, N, 1);
		transpose_re(Y + n, N, 1);
	}
    for (int yi = 0; yi < N; yi++) {
		fft_vector_2d(X + yi * N * N, Y + yi * N * N, Wrptr.data(), Wiptr.data(), N, N);
	}
	fft_batch_1d(X, Y, N, N * N);
}
