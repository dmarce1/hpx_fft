#include <fft/fft.hpp>
#include <fft/timer.hpp>

#include <cmath>

void transpose_yx(double* x, int N) {
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

void _3d_twiddles(double* X, double* Y, int N1, int N2) {
	const int N = N1 * N2;
	for (int x1 = 0; x1 < N1; x1++) {
		for (int y1 = 0; y1 < N1; y1++) {
			for (int z1 = 0; z1 < N1; z1++) {
				for (int x2 = 0; x2 < N2; x2++) {
					for (int y2 = 0; y2 < N2; y2++) {
						for (int z2 = 0; z2 < N2; z2++) {
							const int i = x1 * N2 + x2;
							const int j = y1 * N2 + y2;
							const int k = z1 * N2 + z2;
							const int iii = k + N * (j + N * i);
							const double phi = -2.0 * M_PI * (x1 * x2 + y1 * y2 + z1 * z2) / N;
							const auto w = std::polar(1.0, phi);
							double tmp = X[iii];
							X[iii] = X[iii] * w.real() - Y[iii] * w.imag();
							Y[iii] = tmp * w.imag() + Y[iii] * w.real();
						}
					}
				}
			}
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
	timer tm1, tm2, tm3;
	const auto scramble3d = [N, &X, &Y]() {
		scramble_hi(X, N, N * N);
		scramble_hi(Y, N, N * N);
		for (int xi = 0; xi < N; xi++) {
			const int n = xi * N * N;
			scramble_hi(X + n, N, N);
			scramble_hi(Y + n, N, N);
			transpose_zy(X + n, N);
			transpose_zy(Y + n, N);
			scramble_hi(X + n, N, N);
			scramble_hi(Y + n, N, N);
			transpose_zy(X + n, N);
			transpose_zy(Y + n, N);
		}
	};
	const int N1 = 1 << (std::ilogb(N) >> 1);
	const int N2 = N / N1;
	const int log2N1 = std::ilogb(N1);
	const int log2N2 = std::ilogb(N2);
	scramble3d();
	for (int xi = 0; xi < N; xi++) {
		for (int yi = 0; yi < N; yi += N2) {
			for (int zi = 0; zi < N; zi += N2) {
				const int n = zi + (yi + xi * N) * N;
				fft_vector_2d(X + n, Y + n, Wrptr.data(), Wiptr.data(), N2, N);
			}
		}
	}
	for (int xi = 0; xi < N; xi += N2) {
		const int n = xi * N * N;
		fft_1d_dit(X + n, Y + n, Wr[log2N2].data(), Wi[log2N2].data(), N2, N * N);
	}
	_3d_twiddles(X, Y, N1, N2);
	for (int xi = 0; xi < N; xi++) {
		for (int yi = 0; yi < N; yi += N1) {
			for (int zi = 0; zi < N; zi += N2) {
				const int n = zi + (yi + xi * N) * N;
				fft_vector_2d(X + n, Y + n, Wrptr.data(), Wiptr.data(), N1, N);
			}
		}
	}
	for (int xi = 0; xi < N; xi += N1) {
		for (int yi = 0; yi < N; yi += N2) {
			const int n = (yi + xi * N) * N;
			fft_1d_dit(X + n, Y + n, Wr[log2N1].data(), Wi[log2N1].data(), N1, N * N);
		}
	}

}
