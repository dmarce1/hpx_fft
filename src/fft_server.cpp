#include <fft/fft.hpp>

fft_server::fft_server(int N_, const std::array<int, NDIM>& pos_) :
		Nglobal(N_), lb(pos_) {
}

void fft_server::transpose_yxz() {
	std::vector<double> tmp(Nlocal);
	const int copy_sz = sizeof(double) * Nlocal;
	for (int i = 0; i < Nlocal; i++) {
		for (int j = i + 1; j < Nlocal; j++) {
			const int n = Nlocal * (j + Nlocal * i);
			const int m = Nlocal * (i + Nlocal * j);
			std::memcpy(tmp.data(), X.data() + n, copy_sz);
			std::memcpy(X.data() + n, X.data() + m, copy_sz);
			std::memcpy(X.data() + m, tmp.data(), copy_sz);
			std::memcpy(tmp.data(), Y.data() + n, copy_sz);
			std::memcpy(Y.data() + n, Y.data() + m, copy_sz);
			std::memcpy(Y.data() + m, tmp.data(), copy_sz);
		}
	}
	std::swap(lb[XDIM], lb[YDIM]);
	std::swap(ub[XDIM], ub[YDIM]);
	std::swap(servers[XDIM], servers[YDIM]);
}

void fft_server::transpose_zyx() {
	if (Nlocal >= SIMD_SIZE) {
		transpose_zyx_asm(X.data(), Nlocal);
		transpose_zyx_asm(Y.data(), Nlocal);
	} else {
		for (int i = 0; i < Nlocal; i++) {
			for (int j = 0; j < Nlocal; j++) {
				for (int k = 0; k < Nlocal; k++) {
					const int n = k + Nlocal * (j + Nlocal * i);
					const int m = i + Nlocal * (j + Nlocal * k);
					if (n > m) {
						std::swap(X[n], X[m]);
						std::swap(Y[n], Y[m]);
					}
				}
			}
		}
	}
	std::swap(lb[XDIM], lb[ZDIM]);
	std::swap(ub[XDIM], ub[ZDIM]);
	std::swap(servers[XDIM], servers[ZDIM]);
}

std::pair<swap_arc, swap_arc> fft_server::exchange(swap_arc&& x, swap_arc&& y, int xi) {
	const size_t copy_sz = sizeof(double) * x.size();
	std::pair<swap_arc, swap_arc> rc;
	const int n = Nlocal * Nlocal * (xi - lb[XDIM]);
	x.swap(X.data() + n);
	y.swap(Y.data() + n);
	rc.first = std::move(x);
	rc.second = std::move(y);
	return std::move(rc);
}

void fft_server::apply_fft() {
	const int N = 1 << (std::ilogb(Nglobal) >> 1);
	const int M = Nlocal * Nlocal;
	for (int i = 0; i < Nlocal; i += N) {
		const int n = M * i;
		apply_fft_1d(X.data() + n, Y.data() + n, N, M);
	}
}

void fft_server::apply_twiddles() {
	const int N = 1 << (std::ilogb(Nglobal) >> 1);
	const int log2N = std::ilogb(N);
	const int mask = N - 1;
	const int M = Nlocal * Nlocal;
	for (int i = 0; i < Nlocal; i++) {
		const int j = lb[XDIM] + i;
		const int k2 = j & mask;
		const int n1 = j >> log2N;
		const int n1r = bit_reverse(n1, N);
		if (n1r * k2) {
			const auto W = std::polar(1.0, -n1r * k2 * 2.0 * M_PI / Nglobal);
			apply_twiddles_1d(X.data() + M * i, Y.data() + M * i, W.real(), W.imag(), (size_t) M);
		}
	}
}

void fft_server::scramble() {
	transform([this](int i) {
		const int j = bit_reverse(i, Nglobal);
		return j;
	});
}

void fft_server::transpose_x() {
	transform([this](int ii) {
		const int N = 1 << (std::ilogb(Nglobal) >> 1);
		const int ix = ii >> std::ilogb(N);
		const int iy = ii & (N - 1);
		const int jj = iy * N + ix;
		return jj;
	});
}

void fft_server::transform(const std::function<int(int)>& P) {
	std::vector < hpx::future<std::pair<swap_arc, swap_arc>> > futs1;
	std::vector < hpx::future<void> > futs2;
	const int sz = Nlocal * Nlocal;
	for (int xi = lb[XDIM]; xi < ub[XDIM]; xi++) {
		const int xj = P(xi);
		const int n = sz * (xi - lb[XDIM]);
		if (xi < xj) {
			swap_arc x;
			swap_arc y;
			x.set(X.data() + n, sz);
			y.set(Y.data() + n, sz);
			const int orank = xj / Nlocal;
			futs1.push_back(hpx::async<typename fft_server::exchange_action>(servers[XDIM][orank], std::move(x), std::move(y), xj));
		}
	}
	int lll = 0;
	for (int xi = lb[XDIM]; xi < ub[XDIM]; xi++) {
		const int xj = P(xi);
		const int n = sz * (xi - lb[XDIM]);
		if (xi < xj) {
			futs2.push_back(futs1[lll++].then([n, this](hpx::future<std::pair<swap_arc, swap_arc>>&& fut) {
				const auto z = fut.get();
				z.first.get(X.data() + n);
				z.second.get(Y.data() + n);
			}));
		}
	}
	hpx::wait_all(futs2.begin(), futs2.end());
}

std::vector<std::complex<double>> fft_server::read(const std::array<int, NDIM>& xlb, const std::array<int, NDIM>& xub) {
	std::array<int, NDIM> dx;
	int sz = 1;
	for (int dim = 0; dim < NDIM; dim++) {
		dx[dim] = xub[dim] - xlb[dim];
		sz *= dx[dim];
	}
	std::vector<std::complex<double>> Z(sz);
	for (int i = xlb[XDIM]; i < xub[XDIM]; i++) {
		for (int j = xlb[YDIM]; j < xub[YDIM]; j++) {
			for (int k = xlb[ZDIM]; k < xub[ZDIM]; k++) {
				const int lll = (k - xlb[ZDIM]) + Nlocal * ((j - xlb[YDIM]) + Nlocal * (i - xlb[XDIM]));
				Z[lll].real(X[index(i, j, k)]);
				Z[lll].imag(Y[index(i, j, k)]);
			}
		}
	}
	return std::move(Z);
}

void fft_server::set_servers(std::array<std::vector<hpx::id_type>, NDIM>&& s) {
	servers = std::move(s);
	int sz = 1;
	Nrank = servers[0].size();
	Nlocal = Nglobal / Nrank;
	for (int dim = 0; dim < NDIM; dim++) {
		sz *= Nlocal;
		lb[dim] *= Nlocal;
		ub[dim] = lb[dim] + Nlocal;
	}
	X.resize(sz);
	Y.resize(sz);
}

int fft_server::index(int i, int j, int k) const {
	return (k - lb[ZDIM]) + Nlocal * ((j - lb[YDIM]) + Nlocal * (i - lb[XDIM]));
}

void fft_server::write(std::vector<std::complex<double>>&& Z, const std::array<int, NDIM>& xlb, const std::array<int, NDIM>& xub) {
	std::array<int, NDIM> dx;
	for (int dim = 0; dim < NDIM; dim++) {
		dx[dim] = xub[dim] - xlb[dim];
	}
	for (int i = xlb[XDIM]; i < xub[XDIM]; i++) {
		for (int j = xlb[YDIM]; j < xub[YDIM]; j++) {
			for (int k = xlb[ZDIM]; k < xub[ZDIM]; k++) {
				const int lll = (k - xlb[ZDIM]) + dx[ZDIM] * ((j - xlb[YDIM]) + dx[YDIM] * (i - xlb[XDIM]));
				X[index(i, j, k)] = Z[lll].real();
				Y[index(i, j, k)] = Z[lll].imag();
			}
		}
	}
}
