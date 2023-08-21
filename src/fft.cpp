#include <fft/fft.hpp>

fft3d::fft3d(int N_, std::vector<hpx::id_type>&& localities) :
		Nglobal(N_) {
	std::vector<hpx::future<void>> futs;
	Nrank = 1 << (std::ilogb(localities.size()) / NDIM);
	Nlocal = Nglobal / Nrank;
	const int Nrank3 = Nrank * Nrank * Nrank;
	futs.resize(Nrank3);
	int lll = 0;
	std::array<int, NDIM> pos;
	servers.resize(Nrank3);
	for (pos[XDIM] = 0; pos[XDIM] < Nrank; pos[XDIM]++) {
		for (pos[YDIM] = 0; pos[YDIM] < Nrank; pos[YDIM]++) {
			for (pos[ZDIM] = 0; pos[ZDIM] < Nrank; pos[ZDIM]++) {
				futs[lll] = hpx::new_ < fft_server > (localities[lll], Nglobal, pos).then([this, lll](hpx::future<hpx::id_type>&& fut) {
					auto id = fut.get();
					servers[lll] = id;
				});
				lll++;
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	for (int i = 0; i < Nrank3; i++) {
		auto s = servers;
		futs[i] = hpx::async<typename fft_server::set_servers_action>(servers[i], std::move(s));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

std::vector<std::complex<double>> fft3d::read(std::array<int, NDIM> xlb, std::array<int, NDIM> xub) {
	std::array<int, NDIM> dx2;
	std::array<int, NDIM> dx1;
	int sz = 1;
	for (int dim = 0; dim < NDIM; dim++) {
		dx1[dim] = xub[dim] - xlb[dim];
		sz *= dx1[dim];
	}
	std::vector<std::complex<double>> Z(sz);
	std::array<int, NDIM> zlb, zub;
	std::array<int, NDIM> ylb, yub;
	std::vector < hpx::future<std::vector<std::complex<double>>> >futs1;
	std::vector < hpx::future<void> > futs2;
	for (int i = 0; i < Nrank; i++) {
		zlb[XDIM] = i * Nlocal;
		zub[XDIM] = zlb[XDIM] + Nlocal;
		for (int j = 0; j < Nrank; j++) {
			zlb[YDIM] = j * Nlocal;
			zub[YDIM] = zlb[YDIM] + Nlocal;
			for (int k = 0; k < Nrank; k++) {
				zlb[ZDIM] = k * Nlocal;
				zub[ZDIM] = zlb[ZDIM] + Nlocal;
				int vol = 1;
				for (int dim = 0; dim < NDIM; dim++) {
					ylb[dim] = std::max(xlb[dim], zlb[dim]);
					yub[dim] = std::min(xub[dim], zub[dim]);
					vol *= yub[dim] - ylb[dim];
				}
				if (vol) {
					const int iii = k + Nrank * (j + Nrank * i);
					futs1.push_back(hpx::async<typename fft_server::read_action>(servers[iii], ylb, yub));
				}
			}
		}
	}
	int lll = 0;
	for (int i = 0; i < Nrank; i++) {
		zlb[XDIM] = i * Nlocal;
		zub[XDIM] = zlb[XDIM] + Nlocal;
		for (int j = 0; j < Nrank; j++) {
			zlb[YDIM] = j * Nlocal;
			zub[YDIM] = zlb[YDIM] + Nlocal;
			for (int k = 0; k < Nrank; k++) {
				zlb[ZDIM] = k * Nlocal;
				zub[ZDIM] = zlb[ZDIM] + Nlocal;
				int vol = 1;
				for (int dim = 0; dim < NDIM; dim++) {
					ylb[dim] = std::max(xlb[dim], zlb[dim]);
					yub[dim] = std::min(xub[dim], zub[dim]);
					dx2[dim] = yub[dim] - ylb[dim];
					assert(dx2[dim] >= 0);
					vol *= dx2[dim];
				}
				if (vol) {
					futs2.push_back(futs1[lll++].then([&Z, this, ylb, xlb, yub, xub, dx1, dx2](hpx::future<std::vector<std::complex<double>>>&& fut) {
						const auto z = fut.get();
						for (int i = ylb[XDIM]; i < yub[XDIM]; i++) {
							for (int j = ylb[YDIM]; j < yub[YDIM]; j++) {
								for (int k = ylb[ZDIM]; k < yub[ZDIM]; k++) {
									const int lll = (k - ylb[ZDIM]) + dx2[ZDIM] * ((j - ylb[YDIM]) + dx2[YDIM] * (i - ylb[XDIM]));
									const int mmm = (k - xlb[ZDIM]) + dx1[ZDIM] * ((j - xlb[YDIM]) + dx1[YDIM] * (i - xlb[XDIM]));
									Z[mmm] = z[lll];
								}
							}
						}
					}));
				}
			}
		}
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	return std::move(Z);
}

void fft3d::write(std::vector<std::complex<double>>&& Z, std::array<int, NDIM> xlb, std::array<int, NDIM> xub) {
	std::array<int, NDIM> dx2;
	std::array<int, NDIM> dx1;
	int sz = 1;
	for (int dim = 0; dim < NDIM; dim++) {
		dx1[dim] = xub[dim] - xlb[dim];
		sz *= dx1[dim];
	}
	std::array<int, NDIM> zlb, zub;
	std::array<int, NDIM> ylb, yub;
	std::vector < hpx::future<void> > futs;
	for (int i = 0; i < Nrank; i++) {
		zlb[XDIM] = i * Nlocal;
		zub[XDIM] = zlb[XDIM] + Nlocal;
		for (int j = 0; j < Nrank; j++) {
			zlb[YDIM] = j * Nlocal;
			zub[YDIM] = zlb[YDIM] + Nlocal;
			for (int k = 0; k < Nrank; k++) {
				zlb[ZDIM] = k * Nlocal;
				zub[ZDIM] = zlb[ZDIM] + Nlocal;
				int vol = 1;
				for (int dim = 0; dim < NDIM; dim++) {
					ylb[dim] = std::max(xlb[dim], zlb[dim]);
					yub[dim] = std::min(xub[dim], zub[dim]);
					dx2[dim] = yub[dim] - ylb[dim];
					vol *= dx2[dim];
				}
				if (vol) {
					std::vector<std::complex<double>> z(vol);
					for (int i = ylb[XDIM]; i < yub[XDIM]; i++) {
						for (int j = ylb[YDIM]; j < yub[YDIM]; j++) {
							for (int k = ylb[ZDIM]; k < yub[ZDIM]; k++) {
								const int lll = (k - ylb[ZDIM]) + dx2[ZDIM] * ((j - ylb[YDIM]) + dx2[YDIM] * (i - ylb[XDIM]));
								const int mmm = (k - xlb[ZDIM]) + dx1[ZDIM] * ((j - xlb[YDIM]) + dx1[YDIM] * (i - xlb[XDIM]));
								z[lll] = Z[mmm];
							}
						}
					}
					const int iii = k + Nrank * (j + Nrank * i);
					futs.push_back(hpx::async<typename fft_server::write_action>(servers[iii], std::move(z), ylb, yub));
				}
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d::scramble(int dim) {
	std::vector<hpx::future<void>> futs;
	const int Nrank3 = Nrank * Nrank * Nrank;
	for (int i = 0; i < Nrank3; i++) {
		futs.push_back(hpx::async<typename fft_server::scramble_action>(servers[i], dim));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d::transpose(int dim) {
	std::vector<hpx::future<void>> futs;
	const int Nrank3 = Nrank * Nrank * Nrank;
	for (int i = 0; i < Nrank3; i++) {
		futs.push_back(hpx::async<typename fft_server::transpose_action>(servers[i], dim));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

