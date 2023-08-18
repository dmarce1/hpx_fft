#include <fft/fft.hpp>

#include <cmath>

void vector_radix2_2d(double* Xr, double* Xi, int xb, int yb, int N, int M) {
	const int N2 = N >> 1;
	if (N > 2) {
		vector_radix2_2d(Xr, Xi, xb, yb, N2, M);
		vector_radix2_2d(Xr, Xi, xb + N2, yb, N2, M);
		vector_radix2_2d(Xr, Xi, xb, yb + N2, N2, M);
		vector_radix2_2d(Xr, Xi, xb + N2, yb + N2, N2, M);
	}
	const int xe = xb + N2;
	const int ye = yb + N2;
	const double phi0 = -2.0 * M_PI / N;
	for (int kx = xb; kx < xe; kx++) {
		const auto w0x = std::polar(1.0, phi0 * kx);
		for (int ky = yb; ky < ye; ky++) {
			const auto w0y = std::polar(1.0, phi0 * ky);
			const auto wxy = w0x * w0y;
			for (int m = 0; m < M; m++) {
				const int i00 = (M * kx + ky) * M + m;
				const int i01 = i00 + M * N2;
				const int i10 = i00 + M * N2 * M;
				const int i11 = i00 + M * N2 * (M + 1);
				double tr00, tr01, tr10, tr11;
				double ti00, ti01, ti10, ti11;
				auto& er00 = Xr[i00];
				auto& er01 = Xr[i01];
				auto& er10 = Xr[i10];
				auto& er11 = Xr[i11];
				auto& ei00 = Xi[i00];
				auto& ei01 = Xi[i01];
				auto& ei10 = Xi[i10];
				auto& ei11 = Xi[i11];
				tr00 = er00;
				ti00 = ei00;
				tr10 = er10 * w0x.real() - ei10 * w0x.imag();
				ti10 = er10 * w0x.imag() + ei10 * w0x.real();
				tr01 = er01 * w0y.real() - ei01 * w0y.imag();
				ti01 = er01 * w0y.imag() + ei01 * w0y.real();
				tr11 = er11 * wxy.real() - ei11 * wxy.imag();
				ti11 = er11 * wxy.imag() + ei11 * wxy.real();
				er00 = tr00 + tr01 + tr10 + tr11;
				ei00 = ti00 + ti01 + ti10 + ti11;
				er10 = tr00 + tr01 - tr10 - tr11;
				ei10 = ti00 + ti01 - ti10 - ti11;
				er01 = tr00 - tr01 + tr10 - tr11;
				ei01 = ti00 - ti01 + ti10 - ti11;
				er11 = tr00 - tr01 - tr10 + tr11;
				ei11 = ti00 - ti01 - ti10 + ti11;
			}
		}
	}
}

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
	std::vector<std::vector<double>> Wr(log2N + 1);
	std::vector<std::vector<double>> Wi(log2N + 1);
	std::vector<const double*> Wrptr(log2N + 1);
	std::vector<const double*> Wiptr(log2N + 1);
	for (int n = 0; n <= log2N; n++) {
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
