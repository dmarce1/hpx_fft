#include <fft/fft.hpp>

fft3d::fft3d(int N_, std::vector<hpx::id_type>&& localities) :
		Nglobal(N_) {
	std::vector < std::vector<std::vector<hpx::future<hpx::id_type>>> >cfuts;
	std::vector<hpx::future<void>> ifuts;
	Nrank = 1 << (std::ilogb(localities.size()) / NDIM);
	Nlocal = Nglobal / Nrank;
	cfuts.resize(Nrank);
	int lll = 0;
	std::array<int, NDIM> pos;
	for (pos[XDIM] = 0; pos[XDIM] < Nrank; pos[XDIM]++) {
		cfuts[pos[XDIM]].resize(Nrank);
		for (pos[YDIM] = 0; pos[YDIM] < Nrank; pos[YDIM]++) {
			cfuts[pos[XDIM]][pos[YDIM]].resize(Nrank);
			for (pos[ZDIM] = 0; pos[ZDIM] < Nrank; pos[ZDIM]++) {
				cfuts[pos[XDIM]][pos[YDIM]][pos[ZDIM]] = hpx::new_ < fft_server > (localities[lll], Nglobal, pos);
				lll++;
			}
		}
	}
	servers.resize(Nrank);
	for (int i = 0; i < Nrank; i++) {
		servers[i].resize(Nrank);
		for (int j = 0; j < Nrank; j++) {
			servers[i][j].resize(Nrank);
			for (int k = 0; k < Nrank; k++) {
				servers[i][j][k] = cfuts[i][j][k].get();
			}
		}
	}
	for (int i = 0; i < Nrank; i++) {
		for (int j = 0; j < Nrank; j++) {
			for (int k = 0; k < Nrank; k++) {
				std::array<std::vector<hpx::id_type>, NDIM> svrs;
				for (int dim = 0; dim < NDIM; dim++) {
					svrs[dim].resize(Nrank);
				}
				for (int m = 0; m < Nrank; m++) {
					svrs[XDIM][m] = servers[m][j][k];
				}
				for (int m = 0; m < Nrank; m++) {
					svrs[YDIM][m] = servers[i][m][k];
				}
				for (int m = 0; m < Nrank; m++) {
					svrs[ZDIM][m] = servers[i][j][m];
				}
				ifuts.push_back(hpx::async<typename fft_server::set_servers_action>(servers[i][j][k], std::move(svrs)));
			}
		}
	}
	hpx::wait_all(ifuts.begin(), ifuts.end());
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
					futs1.push_back(hpx::async<typename fft_server::read_action>(servers[i][j][k], ylb, yub));
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
					assert(dx2[dim]>=0);
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
					futs.push_back(hpx::async<typename fft_server::write_action>(servers[i][j][k], std::move(z), ylb, yub));
				}
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d::scramble() {
	std::vector<hpx::future<void>> futs;
	for (int i = 0; i < Nrank; i++) {
		for (int j = 0; j < Nrank; j++) {
			for (int k = 0; k < Nrank; k++) {
				futs.push_back(hpx::async<typename fft_server::scramble_action>(servers[i][j][k]));
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d::transpose_x() {
	std::vector<hpx::future<void>> futs;
	for (int i = 0; i < Nrank; i++) {
		for (int j = 0; j < Nrank; j++) {
			for (int k = 0; k < Nrank; k++) {
				futs.push_back(hpx::async<typename fft_server::transpose_x_action>(servers[i][j][k]));
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d::transpose_yxz() {
	std::vector<hpx::future<void>> futs;
	for (int i = 0; i < Nrank; i++) {
		for (int j = 0; j < Nrank; j++) {
			for (int k = 0; k < Nrank; k++) {
				futs.push_back(hpx::async<typename fft_server::transpose_yxz_action>(servers[i][j][k]));
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d::transpose_zyx() {
	std::vector<hpx::future<void>> futs;
	for (int i = 0; i < Nrank; i++) {
		for (int j = 0; j < Nrank; j++) {
			for (int k = 0; k < Nrank; k++) {
				futs.push_back(hpx::async<typename fft_server::transpose_zyx_action>(servers[i][j][k]));
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d::apply_fft() {
	std::vector<hpx::future<void>> futs;
	for (int i = 0; i < Nrank; i++) {
		for (int j = 0; j < Nrank; j++) {
			for (int k = 0; k < Nrank; k++) {
				futs.push_back(hpx::async<typename fft_server::apply_fft_action>(servers[i][j][k]));
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d::apply_twiddles() {
	std::vector<hpx::future<void>> futs;
	for (int i = 0; i < Nrank; i++) {
		for (int j = 0; j < Nrank; j++) {
			for (int k = 0; k < Nrank; k++) {
				futs.push_back(hpx::async<typename fft_server::apply_twiddles_action>(servers[i][j][k]));
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}
