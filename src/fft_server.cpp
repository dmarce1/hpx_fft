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
				Z[lll].real(X[index(i, j, k)]);
				Z[lll].imag(Y[index(i, j, k)]);
			}
		}
	}
	return std::move(Z);
}

void fft_server::set_servers(std::vector<hpx::id_type>&& s) {
	servers = std::move(s);
	Nrank = std::lround(std::pow(servers.size(), 1.0 / 3.0));
	Nlocal = Nglobal / Nrank;
	for (int dim = 0; dim < NDIM; dim++) {
		lb[dim] *= Nlocal;
		ub[dim] = lb[dim] + Nlocal;
	}
	X.resize(Nlocal * Nlocal * Nlocal);
	Y.resize(Nlocal * Nlocal * Nlocal);
}

int fft_server::index(int i, int j, int k) const {
	return (k - lb[ZDIM]) + Nlocal * ((j - lb[YDIM]) + Nlocal * (i - lb[XDIM]));
}

int swap_bits(int i, int bit1, int bit2) {
	const int mask = ((i >> bit1) & 1) ^ ((i >> bit2) & 1);
	i ^= mask << bit1;
	i ^= mask << bit2;
	return i;
}

std::array<std::vector<int>, NDIM> pbits2perms(const bitswap3d_t& pbits, int N) {
	std::array<std::vector<int>, NDIM> P;
	for (int dim = 0; dim < NDIM; dim++) {
		const auto& bits = pbits[dim];
		P[dim].resize(N);
		for (int n = 0; n < N; n++) {
			int m = n;
			for (int i = 0; i < bits.size(); i++) {
				m = swap_bits(m, bits[i].first, bits[i].second);
			}
			P[dim][n] = m;
		}
	}
	return std::move(P);
}

std::vector<double> fft_server::exchange(const bitswap3d_t& pbits, std::vector<double>&& Z, int xr, int yr, int zr) {
	const auto P = pbits2perms(pbits, Nglobal);
	const int xlb = xr * Nlocal;
	const int ylb = yr * Nlocal;
	const int zlb = zr * Nlocal;
	const int xub = xlb + Nlocal;
	const int yub = ylb + Nlocal;
	const int zub = zlb + Nlocal;
	const int rank_shft = std::ilogb(Nlocal);
	const int xri = lb[XDIM] >> rank_shft;
	const int yri = lb[YDIM] >> rank_shft;
	const int zri = lb[ZDIM] >> rank_shft;
	int l = 0;
	for (int xi = xlb; xi < xub; xi++) {
		const int xj = P[XDIM][xi];
		const int xrj = xj >> rank_shft;
		if (xrj == xri) {
			for (int yi = ylb; yi < yub; yi++) {
				const int yj = P[YDIM][yi];
				const int yrj = yj >> rank_shft;
				if (yrj == yri) {
					for (int zi = zlb; zi < zub; zi++) {
						const int zj = P[ZDIM][zi];
						const int zrj = zj >> rank_shft;
						if (zrj == zri) {
							const int m = index(xj, yj, zj);
							std::swap(X[m], Z[l++]);
							std::swap(Y[m], Z[l++]);
						}
					}
				}
			}
		}
	}
	return std::move(Z);
}

void fft_server::transform(const bitswap3d_t& pbits) {
	const auto P = pbits2perms(pbits, Nglobal);
	const int rank_shft = std::ilogb(Nlocal);
	const int ir = (lb[XDIM] >> rank_shft);
	const int jr = (lb[YDIM] >> rank_shft);
	const int kr = (lb[ZDIM] >> rank_shft);
	const int irank = kr + Nrank * (jr + Nrank * ir);
	std::unordered_map<int, std::vector<double>> sends;
	std::vector<hpx::future<void>> futs;
	for (int xi = lb[XDIM]; xi < ub[XDIM]; xi++) {
		const int xj = P[XDIM][xi];
		const int xrank = xj >> rank_shft;
		for (int yi = lb[YDIM]; yi < ub[YDIM]; yi++) {
			const int yj = P[YDIM][yi];
			const int yrank = yj >> rank_shft;
			for (int zi = lb[ZDIM]; zi < ub[ZDIM]; zi++) {
				const int zj = P[ZDIM][zi];
				const int zrank = zj >> rank_shft;
				const int jrank = zrank + Nrank * (yrank + xrank * Nrank);
				if (irank > jrank) {
					const int n = index(xi, yi, zi);
					sends[jrank].push_back(X[n]);
					sends[jrank].push_back(Y[n]);
				} else if (irank == jrank) {
					const int n = index(xi, yi, zi);
					const int m = index(xj, yj, zj);
					if (n > m) {
						std::swap(X[n], X[m]);
						std::swap(Y[n], Y[m]);
					}
				}
			}
		}
	}
	for (auto si = sends.begin(); si != sends.end(); si++) {
		const int xri = si->first / (Nrank * Nrank);
		const int yri = (si->first / Nrank) % Nrank;
		const int zri = si->first % Nrank;
		futs.push_back(
				hpx::async<typename fft_server::exchange_action>(servers[si->first], pbits, std::move(si->second), ir,
						jr, kr).then([this, rank_shft, xri, yri, zri, P](hpx::future<std::vector<double>>&& fut) {
					const auto Z = fut.get();
					int l = 0;
					for (int xi = lb[XDIM]; xi < ub[XDIM]; xi++) {
						const int xj = P[XDIM][xi];
						const int xrank = xj >> rank_shft;
						if( xrank == xri) {
							for (int yi = lb[YDIM]; yi < ub[YDIM]; yi++) {
								const int yj = P[YDIM][yi];
								const int yrank = yj >> rank_shft;
								if( yrank == yri ) {
									for (int zi = lb[ZDIM]; zi < ub[ZDIM]; zi++) {
										const int zj = P[ZDIM][zi];
										const int zrank = zj >> rank_shft;
										if( zrank == zri) {
											const int n = index(xi, yi, zi);
											X[n] = Z[l++];
											Y[n] = Z[l++];
										}
									}
								}
							}
						}
					}
				}));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft_server::transpose(int dim) {
	bitswap3d_t pbits;
	const int nbits = std::ilogb(Nglobal);
	for (int bit = 0; bit < nbits - bit; bit++) {
		pbits[dim].push_back(std::make_pair(bit, (bit + (Nglobal >> 1)) % Nglobal));
	}
	transform(pbits);
}

void fft_server::scramble(int dim) {
	bitswap3d_t pbits;
	const int nbits = std::ilogb(Nglobal);
	for (int bit = 0; bit < nbits - bit; bit++) {
		pbits[dim].push_back(std::make_pair(bit, nbits - 1 - bit));
	}
	transform(pbits);
}

void fft_server::write(std::vector<std::complex<double>>&& Z, const std::array<int, NDIM>& xlb,
		const std::array<int, NDIM>& xub) {
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
