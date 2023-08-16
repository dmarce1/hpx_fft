#include <fft/fft.hpp>
#include <fft/permuted.hpp>
#include <fft/timer.hpp>

double fftw_3d(std::complex<real>* x, int N) {
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

double fftw_2d(std::complex<real>* x, int N) {
	static std::unordered_map<int, fftw_plan> plans;
	static std::unordered_map<int, fftw_complex*> in;
	static std::unordered_map<int, fftw_complex*> out;
	if (plans.find(N) == plans.end()) {
		in[N] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
		out[N] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
		fftw_plan_with_nthreads(hpx::threads::hardware_concurrency());
		plans[N] = fftw_plan_dft_2d(N, N, in[N], out[N], FFTW_FORWARD, FFTW_MEASURE);
	}
	auto* i = in[N];
	auto* o = out[N];
	for (int n = 0; n < N * N; n++) {
		i[n][0] = x[n].real();
		i[n][1] = x[n].imag();
	}
	timer tm;
	tm.start();
	fftw_execute(plans[N]);
	tm.stop();
	for (int n = 0; n < N * N; n++) {
		x[n].real(o[n][0]);
		x[n].imag(o[n][1]);
	}
	return tm.read();
}

double rand1() {
	return (rand() + 0.5) / RAND_MAX;
}

void test_fft(integer M) {
	const integer N = 2 * M * M;
	integer nthreads = std::min((integer) (1 << (std::ilogb(hpx::threads::hardware_concurrency() - 1) + 1)), M / SIMD_SIZE);
	fft fft(N, std::vector < hpx::id_type > (nthreads, hpx::find_here()));
	for (int n = 0; n < 2; n++) {
		std::vector<real> U(N), V(N);
		for (int i = 0; i < N; i++) {
			U[i] = rand1();
			V[i] = U[i];
		}
		U[0] = V[0] = 1.0;
		fft.write(std::move(U), 0, N);
		timer tm;
		tm.start();
		fft.fft_2d();
		tm.stop();
		U = fft.read(0, N);
		const double tm0 = fftw_2d((std::complex<double>*) V.data(), M);
		//double tm0 = 0.0;
		double err = 0.0;
		for (int i = 0; i < N; i++) {
			err += std::abs(U[i] - V[i]);
		}
		err /= N;
		if (n == 1) {
			printf("%2i %2i %14e %14e %14e\n", M, nthreads, err, tm.read(), tm0);
		}
	}

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

int hpx_main(int argc, char *argv[]) {
	fftw_init_threads();
	for (integer N = 4; N <= 4 * 1024; N *= 2) {
		test_fft(N);
	}
	return hpx::finalize();
}

int main(int argc, char *argv[]) {
	hpx::init(argc, argv);
}

HPX_REGISTER_COMPONENT(hpx::components::managed_component<fft_server>, fft_server)
;

