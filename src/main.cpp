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
		fftw_plan_with_nthreads(hpx::threads::hardware_concurrency());
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

inline int round_up(int i, int r) {
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
	std::vector<std::complex<double>> U(N3);
	std::vector<std::complex<double>> V(N3);
	const int sqrtN = 1 << (std::ilogb(N) >> 1);
	int nthreads = 1 << (NDIM * (std::ilogb(hpx::threads::hardware_concurrency() - 1) / NDIM + 1));
	while (nthreads > sqrtN) {
		nthreads >>= NDIM;
	}
	fft3d fft(N, std::vector < hpx::id_type > (nthreads, hpx::find_here()));
	for (int i = 0; i < N3; i++) {
		U[i] = V[i] = std::complex<double>(rand1(), rand1());
	}
	fft.write(std::move(U), { 0, 0, 0 }, { N, N, N });

	fft.transpose_zyx();
	fft.scramble();
	fft.apply_fft();
	fft.apply_twiddles();
	fft.transpose_x();
	fft.apply_fft();
	fft.transpose_x();
	fft.transpose_zyx();

	fft.transpose_yxz();
	fft.scramble();
	fft.apply_fft();
	fft.apply_twiddles();
	fft.transpose_x();
	fft.apply_fft();
	fft.transpose_x();
	fft.transpose_yxz();

	fft.scramble();
	fft.apply_fft();
	fft.apply_twiddles();
	fft.transpose_x();
	fft.apply_fft();
	fft.transpose_x();

	U = fft.read( { 0, 0, 0 }, { N, N, N });
	fftw_3d(V.data(), N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				const int nnn = k + N * (j + N * i);
				if (U[nnn].real() - V[nnn].real() || U[nnn].imag() - V[nnn].imag()) {
					printf("%4i %4i %4i %15e %15e %15e %15e %15e %15e\n", i, j, k, U[nnn].real(), U[nnn].imag(), V[nnn].real(), V[nnn].imag(), U[nnn].real() - V[nnn].real(), U[nnn].imag() - V[nnn].imag());
				}
			}
		}
	}
}

int hpx_main(int argc, char *argv[]) {
	fftw_init_threads();
	test_fft(16);
	return hpx::finalize();
}

int main(int argc, char *argv[]) {
	auto i = mp_units::cgs::unit_symbols::cm;
	hpx::init(argc, argv);
}

HPX_REGISTER_COMPONENT(hpx::components::managed_component<fft_server>, fft_server);

