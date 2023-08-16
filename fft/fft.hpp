/*
 * fft.hpp
 *
 *  Created on: Aug 14, 2023
 *      Author: dmarce1
 */

#ifndef FFT_HPP_
#define FFT_HPP_

#define SIMD_SIZE 4

#include <hpx/include/components.hpp>

#include <fftw3.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/channel.hpp>
#include <hpx/mutex.hpp>
#include <simd.hpp>

#include <array>
#include <chrono>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#define XDIM 0
#define YDIM 1
#define ZDIM 2
#define NDIM 3

class fft_server: public hpx::components::managed_component_base<fft_server> {
	const int Nglobal;
	int Nrank;
	int Nlocal;
	std::array<int, NDIM> lb;
	std::array<int, NDIM> ub;
	std::array<std::vector<hpx::id_type>, NDIM> servers;
	std::vector<double> X;
	std::vector<double> Y;
	int index(int, int, int) const;
public:
	fft_server(int N, const std::array<int, NDIM>& pos);
	void set_servers(std::array<std::vector<hpx::id_type>, NDIM>&&); //
	void write(std::vector<std::complex<double>>&&, const std::array<int, NDIM>&, const std::array<int, NDIM>&); //
	std::vector<std::complex<double>> read(const std::array<int, NDIM>&, const std::array<int, NDIM>&); //
	void transpose_xy();
	void transpose_xz();HPX_DEFINE_COMPONENT_ACTION(fft_server, read);HPX_DEFINE_COMPONENT_ACTION(fft_server, set_servers); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, write);HPX_DEFINE_COMPONENT_ACTION(fft_server, transpose_xy);HPX_DEFINE_COMPONENT_ACTION(fft_server, transpose_xz);
	//
};

class fft {
	const int Nglobal;
	int Nrank;
	int Nlocal;
	std::vector<std::vector<std::vector<hpx::id_type>>>servers;
	fft(int, std::vector<hpx::id_type>&& );
	void write(std::vector<std::complex<double>>&&, const std::array<int, NDIM>&, const std::array<int, NDIM>&); //
	std::vector<std::complex<double>> read(const std::array<int, NDIM>&, const std::array<int, NDIM>&);//
};

extern "C" {
void transpose_xz_asm(double* X, size_t N);
}

#endif /* FFT_HPP_ */
