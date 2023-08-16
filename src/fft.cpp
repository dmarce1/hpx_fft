#include <fft/fft.hpp>

fft::fft(integer N_, std::vector<hpx::id_type>&& localities) :
		N(N_) {
	std::vector < std::vector < std::vector<hpx::future<hpx::id_type>>> > cfuts;
	std::vector<hpx::future<void>> ifuts;
	std::array<integer, NDIM> M;
	const int nrank = 1 << std::ilogb(localities.size());
	M[XDIM] = std::lround(std::pow(nrank, 1.0 / 3.0));
	M[YDIM] = std::lround(std::sqrt(nrank / M[XDIM]));
	M[ZDIM] = nrank / (M[XDIM] * M[YDIM]);
	cfuts.resize(M[XDIM]);
	int lll = 0;
	for (int i = 0; i < M[XDIM]; i++) {
		cfuts[i].resize(M[YDIM]);
		for (int j = 0; j < M[YDIM]; j++) {
			cfuts[i][j].resize(M[ZDIM]);
			for (int k = 0; k < M[ZDIM]; k++) {
				cfuts[i][j][k] = hpx::new_ < fft_server > (localities[lll], N);
				lll++;
			}
		}
	}
	servers.resize(M[XDIM]);
	for (int i = 0; i < M[XDIM]; i++) {
		servers[i].resize(M[YDIM]);
		for (int j = 0; j < M[YDIM]; j++) {
			servers[i][j].resize(M[ZDIM]);
			for (int k = 0; k < M[ZDIM]; k++) {
				servers[i][j][k] = cfuts[i][j][k].get();
			}
		}
	}
	for (int i = 0; i < M[XDIM]; i++) {
		for (int j = 0; j < M[YDIM]; j++) {
			for (int k = 0; k < M[ZDIM]; k++) {
				std::array<std::vector<hpx::id_type>, NDIM> these_servers;
				for (int dim = 0; dim < NDIM; dim++) {
					these_servers[dim].resize(M[dim]);
				}
				for( int m = 0; m < M[XDIM]; m++) {
					these_servers[XDIM][m] = servers[m][j][k];
				}
				for( int m = 0; m < M[YDIM]; m++) {
					these_servers[YDIM][m] = servers[i][m][k];
				}
				for( int m = 0; m < M[ZDIM]; m++) {
					these_servers[ZDIM][m] = servers[i][j][m];
				}
				ifuts.push_back(hpx::async<typename fft_server::set_servers_action>(servers[i][j][k], std::move(these_servers)));
			}
		}
	}
}

