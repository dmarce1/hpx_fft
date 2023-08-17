#include <fft/fft.hpp>

fft_server::fft_server(int N_, const std::array<int, NDIM>& pos_) :
		Nglobal(N_), lb(pos_) {
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
				Z[lll].real(X(i - lb[XDIM], j - lb[YDIM], k - lb[ZDIM]));
				Z[lll].imag(Y(i - lb[XDIM], j - lb[YDIM], k - lb[ZDIM]));
			}
		}
	}
	return std::move(Z);
}

void fft_server::set_servers(std::array<std::vector<hpx::id_type>, NDIM>&& s) {
	servers = std::move(s);
	Nrank = servers[0].size();
	Nlocal = Nglobal / Nrank;
	for (int dim = 0; dim < NDIM; dim++) {
		lb[dim] *= Nlocal;
		ub[dim] = lb[dim] + Nlocal;
	}
	X.resize(Nlocal);
	Y.resize(Nlocal);
}

std::gslice fft_server::gen_slice(int xi, int dim) const {
	size_t start;
	std::valarray < size_t > sizes(NDIM - 1);
	std::valarray < size_t > strides(NDIM - 1);
	sizes[0] = sizes[1] = Nlocal;
	switch (dim) {
	case XDIM:
		start = Nlocal * Nlocal * xi;
		strides[0] = 1;
		strides[1] = Nlocal;
		break;
	case YDIM:
		start = Nlocal * xi;
		strides[0] = 1;
		strides[1] = Nlocal * Nlocal;
		break;
	case ZDIM:
		start = xi;
		strides[0] = Nlocal;
		strides[1] = Nlocal * Nlocal;
		break;
	default:
		assert(false);
	}
	return std::gslice(start, std::move(sizes), std::move(strides));
}

int fft_server::index(int i, int j, int k) const {
	return (k - lb[ZDIM]) + Nlocal * ((j - lb[YDIM]) + Nlocal * (i - lb[XDIM]));
}

std::pair<fft_vector_slice, fft_vector_slice> fft_server::exchange(fft_vector_slice&& Xslice, fft_vector_slice&& Yslice, int i, int dim) {
	const int j = i - lb[dim];
	Xslice.swap(X.get_slice(j, dim));
	Yslice.swap(Y.get_slice(j, dim));
	return std::make_pair<fft_vector_slice, fft_vector_slice>(std::move(Xslice), std::move(Yslice));
}

void fft_server::transform(const std::function<int(int)>& permutation, int dim) {
	std::vector<hpx::future<void>> futs;
	for (int i = 0; i < Nlocal; i++) {
		const int j = i + lb[dim];
		const int k = permutation(j);
		if (j < k) {
			auto xslice = X.get_slice(i, dim);
			auto yslice = Y.get_slice(i, dim);
			const int orank = k / Nlocal;
			const auto& oserver = servers[dim][orank];
			const bool local = hpx::get_colocation_id(oserver).get() == hpx::find_here();
			auto fut = hpx::async<typename fft_server::exchange_action>(oserver, std::move(xslice), std::move(yslice), k, dim);
			if (local) {
				futs.push_back(std::move(fut));
			} else {
				futs.push_back(fut.then([this, dim, i](hpx::future<std::pair<fft_vector_slice, fft_vector_slice>>&& fut) {
					auto arc = fut.get();
					arc.first.assign2(X.get_slice(i, dim));
					arc.second.assign2(Y.get_slice(i, dim));
				}));
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft_server::transpose(int dim) {
	transform([this](int i) {
		const int N = 1 << (std::ilogb(Nglobal) >> 1);
		const int mask = N - 1;
		const int shft = std::ilogb(N);
		return N * (i & mask) + (i >> shft);
	}, dim);
}

void fft_server::transpose_yz() {
	for (int xi = 0; xi < Nlocal; xi++) {
		const int n = xi * Nlocal * Nlocal;
		transpose_xy(X.data() + n, Nlocal);
		transpose_xy(Y.data() + n, Nlocal);
	}
	std::swap(lb[ZDIM], lb[YDIM]);
	std::swap(ub[ZDIM], ub[YDIM]);
	std::swap(servers[ZDIM], servers[YDIM]);
}

void fft_server::apply_fft_1d(int dim, bool tw) {
	const int N = 1 << (std::ilogb(Nglobal) >> 1);
	const int mask = N - 1;
	const int shft = std::ilogb(N);
	assert( N <= Nlocal);
	if (dim == XDIM) {
		for (int xi = 0; xi < Nlocal; xi += N) {
			const int n = xi * Nlocal * Nlocal;
			fft_1d(X.data() + n, Y.data() + n, N, Nlocal * Nlocal);
		}
		if (tw) {
			for (int xi = 0; xi < Nlocal; xi++) {
				const int n = xi * Nlocal * Nlocal;
				const int j = xi + lb[dim];
				const int n1 = bit_reverse(j >> mask, N);
				const int k2 = j & shft;
				if (n1 * k2) {
					const auto W = std::polar(1.0, -2.0 * M_PI * n1 * k2 / Nglobal);
					twiddles(X.data() + n, Y.data() + n, W.real(), W.imag(), Nlocal * Nlocal);

				}
			}
		}
	} else {
		for (int xi = 0; xi < Nlocal; xi++) {
			for (int yi = 0; yi < Nlocal; yi += N) {
				const int n = (yi + xi * Nlocal) * Nlocal;
				fft_1d(X.data() + n, Y.data() + n, N, Nlocal);
			}
			if (tw) {
				for (int yi = 0; yi < Nlocal; yi++) {
					const int n = Nlocal * (yi + xi * Nlocal);
					const int j = yi + lb[dim];
					const int n1 = bit_reverse(j >> mask, N);
					const int k2 = j & shft;
					if (n1 * k2) {
						const auto W = std::polar(1.0, -2.0 * M_PI * n1 * k2 / Nglobal);
						twiddles(X.data() + n, Y.data() + n, W.real(), W.imag(), Nlocal);

					}
				}
			}
		}
	}
}

void fft_server::scramble(int dim) {
	transform([this](int i) {
		return bit_reverse(i, Nglobal);
	}, dim);
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
				X(i - lb[XDIM], j - lb[YDIM], k - lb[ZDIM]) = Z[lll].real();
				Y(i - lb[XDIM], j - lb[YDIM], k - lb[ZDIM]) = Z[lll].imag();
			}
		}
	}
}
