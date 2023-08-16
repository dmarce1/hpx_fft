#include <fft/fft.hpp>

fft_server::fft_server(int N_, const std::array<int, NDIM>& pos_) :
		Nglobal(N_), lb(pos_) {
}

void fft_server::transpose_xy() {
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
}

void fft_server::transpose_xz() {
	transpose_xz_asm(X.data(), Nlocal);
	transpose_xz_asm(Y.data(), Nlocal);
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
		servers[dim] = std::move(s[dim]);
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
