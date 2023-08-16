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

using integer = unsigned long long;
using real = double;

#define XDIM 0
#define YDIM 1
#define ZDIM 2
#define NDIM 3

class fft_server: public hpx::components::managed_component_base<fft_server> {
	const integer N;
	std::array<integer, NDIM> M;
	std::array<integer, NDIM> L;
	std::array<std::vector<hpx::id_type>, NDIM> servers;
	std::vector<std::complex<real>> X;
public:
	fft_server(integer N);
	void set_servers(std::array<std::vector<hpx::id_type>, NDIM>&&);
	HPX_DEFINE_COMPONENT_ACTION(fft_server, set_servers);
};

class fft {
	const integer N;
	std::vector<std::vector<std::vector<hpx::id_type>>> servers;
	fft(integer, std::vector<hpx::id_type>&& );
};

#endif /* FFT_HPP_ */
