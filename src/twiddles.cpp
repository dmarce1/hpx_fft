#include <cmath>
#include <complex>


void get_twiddles( double* Wr, double* Wi, size_t N) {
	for( int n = 0; n < N; n++) {
		const auto W = std::polar(1.0, -2.0 * M_PI * n / N);
		Wr[n] = W.real();
		Wi[n] = W.imag();
	}
}
