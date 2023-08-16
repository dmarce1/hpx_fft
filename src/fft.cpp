#include <fft/fft.hpp>

fft::fft(int N_, std::vector<hpx::id_type>&& localities) :
		Nglobal(N_) {
	std::vector < std::vector<std::vector<hpx::future<hpx::id_type>>> >cfuts;
	std::vector<hpx::future<void>> ifuts;
	const int nrank = 1 << NDIM * (std::ilogb(localities.size()) / NDIM);
	Nlocal = std::lround(std::pow(nrank, 1.0 / 3.0));
	Nrank = Nglobal / Nlocal;
	cfuts.resize(Nlocal);
	int lll = 0;
	std::array<int, NDIM> pos;
	for (pos[XDIM] = 0; pos[XDIM] < Nlocal; pos[XDIM]++) {
		cfuts[pos[XDIM]].resize(Nlocal);
		for (pos[YDIM] = 0; pos[YDIM] < Nlocal; pos[YDIM]++) {
			cfuts[pos[XDIM]][pos[YDIM]].resize(Nlocal);
			for (pos[ZDIM] = 0; pos[ZDIM] < Nlocal; pos[ZDIM]++) {
				cfuts[pos[XDIM]][pos[YDIM]][pos[ZDIM]] = hpx::new_ < fft_server > (localities[lll], Nglobal, pos);
				lll++;
			}
		}
	}
	servers.resize(Nlocal);
	for (int i = 0; i < Nlocal; i++) {
		servers[i].resize(Nlocal);
		for (int j = 0; j < Nlocal; j++) {
			servers[i][j].resize(Nlocal);
			for (int k = 0; k < Nlocal; k++) {
				servers[i][j][k] = cfuts[i][j][k].get();
			}
		}
	}
	for (int i = 0; i < Nlocal; i++) {
		for (int j = 0; j < Nlocal; j++) {
			for (int k = 0; k < Nlocal; k++) {
				std::array<std::vector<hpx::id_type>, NDIM> svrs;
				for (int dim = 0; dim < NDIM; dim++) {
					svrs[dim].resize(Nlocal);
				}
				for (int m = 0; m < Nlocal; m++) {
					svrs[XDIM][m] = servers[m][j][k];
				}
				for (int m = 0; m < Nlocal; m++) {
					svrs[YDIM][m] = servers[i][m][k];
				}
				for (int m = 0; m < Nlocal; m++) {
					svrs[ZDIM][m] = servers[i][j][m];
				}
				ifuts.push_back(hpx::async<typename fft_server::set_servers_action>(servers[i][j][k], std::move(svrs)));
			}
		}
	}
}

std::vector<std::complex<double>> fft::read(const std::array<int, NDIM>& xlb, const std::array<int, NDIM>& xub) {
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
	for (int i = 0; i < Nlocal; i++) {
		zlb[XDIM] = i * Nrank;
		zub[XDIM] = zlb[XDIM] + Nrank;
		for (int j = 0; j < Nlocal; j++) {
			zlb[YDIM] = j * Nrank;
			zub[YDIM] = zlb[YDIM] + Nrank;
			for (int k = 0; k < Nlocal; k++) {
				zlb[ZDIM] = k * Nrank;
				zub[ZDIM] = zlb[ZDIM] + Nrank;
				int vol = 1;
				for (int dim = 0; dim < NDIM; dim++) {
					ylb[dim] = std::max(xlb[dim], zlb[dim]);
					yub[dim] = std::min(yub[dim], yub[dim]);
					vol *= yub[dim] - ylb[dim];
				}
				if (vol) {
					futs1.push_back(hpx::async<typename fft_server::read_action>(servers[i][j][k], ylb, yub));
				}
			}
		}
	}
	int lll = 0;
	for (int i = 0; i < Nlocal; i++) {
		zlb[XDIM] = i * Nrank;
		zub[XDIM] = zlb[XDIM] + Nrank;
		for (int j = 0; j < Nlocal; j++) {
			zlb[YDIM] = j * Nrank;
			zub[YDIM] = zlb[YDIM] + Nrank;
			for (int k = 0; k < Nlocal; k++) {
				zlb[ZDIM] = k * Nrank;
				zub[ZDIM] = zlb[ZDIM] + Nrank;
				int vol = 1;
				for (int dim = 0; dim < NDIM; dim++) {
					ylb[dim] = std::max(xlb[dim], zlb[dim]);
					yub[dim] = std::min(yub[dim], yub[dim]);
					dx2[dim] = yub[dim] - ylb[dim];
					vol *= dx2[dim];
				}
				if (vol) {
					futs2.push_back(futs1[lll].then([&Z, this, ylb, xlb, yub, xub, dx1, dx2](hpx::future<std::vector<std::complex<double>>>&& fut) {
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

void fft::write(std::vector<std::complex<double>>&& Z, const std::array<int, NDIM>& xlb, const std::array<int, NDIM>& xub) {
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
	for (int i = 0; i < Nlocal; i++) {
		zlb[XDIM] = i * Nrank;
		zub[XDIM] = zlb[XDIM] + Nrank;
		for (int j = 0; j < Nlocal; j++) {
			zlb[YDIM] = j * Nrank;
			zub[YDIM] = zlb[YDIM] + Nrank;
			for (int k = 0; k < Nlocal; k++) {
				zlb[ZDIM] = k * Nrank;
				zub[ZDIM] = zlb[ZDIM] + Nrank;
				int vol = 1;
				for (int dim = 0; dim < NDIM; dim++) {
					ylb[dim] = std::max(xlb[dim], zlb[dim]);
					yub[dim] = std::min(yub[dim], yub[dim]);
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
					futs.push_back(hpx::async<typename fft_server::write_action>(servers[i][j][k], std::move(z), ylb, yub));
				}
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

