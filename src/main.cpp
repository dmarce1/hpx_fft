#include <fft/fft.hpp>
#include <fft/timer.hpp>

#include <mp-units/systems/cgs/cgs.h>

double fftw_3d(std::complex<double>* x, int N) {
	static std::unordered_map<int, fftw_plan> plans;
	static std::unordered_map<int, fftw_complex*> in;
	static std::unordered_map<int, fftw_complex*> out;
	if (plans.find(N) == plans.end()) {
		in[N] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * N);
		out[N] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N * N);
		//	fftw_plan_with_nthreads(hpx::threads::hardware_concurrency());
		plans[N] = fftw_plan_dft_3d(N, N, N, in[N], out[N], FFTW_FORWARD, FFTW_MEASURE);
	}
	auto* i = in[N];
	auto* o = out[N];
	for (int n = 0; n < N * N * N; n++) {
		i[n][0] = x[n].real();
		i[n][1] = x[n].imag();
	}
	timer tm;
	tm.start();
	fftw_execute(plans[N]);
	tm.stop();
	for (int n = 0; n < N * N * N; n++) {
		x[n].real(o[n][0]);
		x[n].imag(o[n][1]);
	}
	return tm.read();
}

double fftw_1d(std::complex<double>* x, int N) {
	static std::unordered_map<int, fftw_plan> plans;
	static std::unordered_map<int, fftw_complex*> in;
	static std::unordered_map<int, fftw_complex*> out;
	if (plans.find(N) == plans.end()) {
		in[N] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
		out[N] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
		//	fftw_plan_with_nthreads(hpx::threads::hardware_concurrency());
		plans[N] = fftw_plan_dft_1d(N, in[N], out[N], FFTW_FORWARD, FFTW_MEASURE);
	}
	auto* i = in[N];
	auto* o = out[N];
	for (int n = 0; n < N; n++) {
		i[n][0] = x[n].real();
		i[n][1] = x[n].imag();
	}
	timer tm;
	tm.start();
	fftw_execute(plans[N]);
	tm.stop();
	for (int n = 0; n < N; n++) {
		x[n].real(o[n][0]);
		x[n].imag(o[n][1]);
	}
	return tm.read();
}

double rand1() {
	return (rand() + 0.5) / RAND_MAX;
}

inline std::size_t round_up(std::size_t i, std::size_t r) {
	return r * ((i - 1) / r + 1);
}

void* operator new(std::size_t n) {
	void* memptr;
	if (posix_memalign(&memptr, 32, round_up(n, 32)) != 0) {
		printf("operator new failed!\n");
		abort();
	}
	return memptr;
}

void operator delete(void * p) {
	free(p);
}

void *operator new[](std::size_t n) {
	void* memptr;
	if (posix_memalign(&memptr, 32, round_up(n, 32)) != 0) {
		printf("operator new[] failed!\n");
		abort();
	}
	return memptr;
}

void operator delete[](void *p) {
	free(p);
}

void test_fft(int N) {
	const size_t N3 = N * N * N;
	double a = 0.0;
	double err = 0.0;
	double b = 0.0;
	fft3d fft(N, std::vector < hpx::id_type > (64, hpx::find_here()));
	for (int n = 0; n < 1; n++) {
		std::vector<std::complex<double>> V(N3);
		std::vector<std::complex<double>> U(N3);
		std::vector<double> X(N3);
		std::vector<double> Y(N3);
		for (int i = 0; i < N3; i++) {
			U[i] = V[i] = std::complex<double>(i == 0, 0);
			X[i] = U[i].real();
			Y[i] = U[i].imag();
		}
		//	fft.write(std::move(U), { 0, 0, 0 }, { N, N, N });
		timer tm;
		tm.start();
		fft_3d_local(X.data(), Y.data(), N);
		fft.scramble(XDIM);
		tm.stop();
		const auto tm0 = fftw_3d(V.data(), N);
		err = 0.0;
//		U = fft.read( { 0, 0, 0 }, { N, N, N });
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < N; k++) {
					const int nnn = k + N * (j + N * i);
					const std::complex<double> Z = std::complex<double>(X[nnn], Y[nnn]);
					err += std::abs(Z - V[nnn]);
					printf("%4i %4i %4i %15e %15e %15e %15e %15e %15e\n", i, j, k, Z.real(), Z.imag(), V[nnn].real(),
							V[nnn].imag(), Z.real() - V[nnn].real(), Z.imag() - V[nnn].imag());
				}
			}
		}
		err /= N * N * N;
		if (n) {
			a += tm0;
			b += tm.read();
		}
	}
	printf("%i %e %e %e\n", N, err, a, b);

}
void test_fft1d(int N) {
	double a = 0.0;
	double err = 0.0;
	double b = 0.0;
	for (int n = 0; n < 1; n++) {
		std::vector<std::complex<double>> U(N);
		std::vector<double> X(N * SIMD_SIZE);
		std::vector<double> Y(N * SIMD_SIZE);
		for (int j = 0; j < N; j++) {
			U[j] = std::complex<double>(rand1(), rand1());
			for (int i = SIMD_SIZE * j; i < SIMD_SIZE * j + SIMD_SIZE; i++) {
				X[i] = U[j].real();
				Y[i] = U[j].imag();
			}
		}
		timer tm;
		std::vector<double> Wr(N), Wi(N);
		get_twiddles(Wr.data(), Wi.data(), N);
		tm.start();
		fftw_1d(U.data(), N);
		tm.stop();
		const auto tm0 = tm.read();
		tm.reset();
		tm.start();
		fft_1d_dif(X.data(), Y.data(), Wr.data(), Wi.data(), N, SIMD_SIZE);
		scramble_hi(X.data(), N, SIMD_SIZE);
		scramble_hi(Y.data(), N, SIMD_SIZE);
		tm.stop();
		err = 0.0;
		for (int i = 0; i < N; i++) {
			const int nnn = i;
			const std::complex<double> Z = std::complex<double>(X[nnn * SIMD_SIZE], Y[nnn * SIMD_SIZE]);
			err += std::abs(Z - U[nnn]);
			printf("%4i  %15e %15e %15e %15e %15e %15e\n", i, Z.real(), Z.imag(), U[nnn].real(), U[nnn].imag(),
					Z.real() - U[nnn].real(), Z.imag() - U[nnn].imag());

		}
		err /= N;
		if (n) {
			a += tm0;
			b += tm.read();
		}
	}
	printf("%i %e %e %e\n", N, err, a, b);

}

void yield() {
	hpx::this_thread::yield();
}

int hpx_main(int argc, char *argv[]) {
	//fftw_init_threads();
	for (int N = 16; N <= 16; N *= 2) {
		test_fft1d(N);
	}
	return hpx::finalize();
}

int main(int argc, char *argv[]) {
	hpx::init(argc, argv);
}

HPX_REGISTER_COMPONENT(hpx::components::managed_component<fft_server>, fft_server);

