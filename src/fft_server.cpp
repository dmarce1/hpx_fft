#include <fft/fft.hpp>


fft_server::fft_server(integer N_) : N(N_) {

}


void fft_server::set_servers(std::array<std::vector<hpx::id_type>, NDIM>&& s) {
	servers = std::move(s);
	for( int dim = 0; dim < NDIM; dim++) {
		servers[dim] = std::move(s[dim]);
		M[dim] = servers[dim].size();
		L[dim] = N / M[dim];
	}
}
