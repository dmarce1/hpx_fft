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
	for (int n = 0; n < 4; n++) {
		std::vector<double> X(N3);
		std::vector<double> Y(N3);
		std::vector<std::complex<double>> V(N3);
		for (int i = 0; i < N3; i++) {
			V[i] = std::complex<double>(rand1(), rand1());
			X[i] = V[i].real();
			Y[i] = V[i].imag();
		}
		timer tm;
		tm.start();
		fft_3d_local(X.data(), Y.data(), N);
		tm.stop();
		const auto tm0 = fftw_3d(V.data(), N);
		err = 0.0;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < N; k++) {
					const int nnn = k + N * (j + N * i);
					const std::complex<double> Z(X[nnn], Y[nnn]);
					err += std::abs(Z - V[nnn]);
					//		printf("%4i %4i %4i %15e %15e %15e %15e %15e %15e\n", i, j, k, Z.real(), Z.imag(), V[nnn].real(), V[nnn].imag(), Z.real() - V[nnn].real(), Z.imag() - V[nnn].imag());
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

void yield() {
	hpx::this_thread::yield();
}

int hpx_main(int argc, char *argv[]) {
	//fftw_init_threads();
	return hpx::finalize();
}

int main(int argc, char *argv[]) {
	for (int N = 4; N <= 512; N *= 2) {
		test_fft(N);
	}
	//auto i = mp_units::cgs::unit_symbols::cm;
	//hpx::init(argc, argv);
}

HPX_REGISTER_COMPONENT(hpx::components::managed_component<fft_server>, fft_server);

