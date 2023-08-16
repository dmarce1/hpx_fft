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

using integer = long long;
using tint = signed char;
using real = double;

class fft {
	std::vector<hpx::id_type> servers;
	integer N;
	integer nrank;
	integer Nperrank;

public:

	fft(integer N_, std::vector<hpx::id_type> localities);
	void fft_2d();
	void apply_fft(integer M1, integer M0, integer L) const;
	std::vector<real> read(integer xib, integer xie);
	void transpose(std::vector<tint> P);
	void write(std::vector<real>&& Z, integer xib, integer xie);
};

class fft_server: public hpx::components::managed_component_base<fft_server> {

	std::vector<hpx::id_type> servers;
	std::vector<real> X;
	std::vector<hpx::channel<std::vector<real>>>channels;
	integer N;
	integer begin;
	integer end;
	integer rank;
	integer nrank;

public:

	fft_server();
	void apply_fft(integer M1, integer M0, integer L);
	void init(std::vector<hpx::id_type> servers_, integer rank_, integer N_);
	std::vector<real> read(integer xib, integer xie);
	void write(std::vector<real>&& Z, integer xib, integer xie);
	void set_channel(std::vector<real>&& Z, integer orank);
	void transpose(std::vector<tint> pindices);
	//
	HPX_DEFINE_COMPONENT_ACTION(fft_server, apply_fft);//
	HPX_DEFINE_COMPONENT_ACTION(fft_server, init);//
	HPX_DEFINE_COMPONENT_ACTION(fft_server, read);//
	HPX_DEFINE_COMPONENT_ACTION(fft_server, write);//
	HPX_DEFINE_COMPONENT_ACTION(fft_server, transpose);//
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(fft_server, set_channel);//
	//

};

extern "C" {
void transpose_re(double*, integer);
void scramble_hi(double*, integer, integer);
void fft_1d(double*, const double*, const double*, integer, integer, integer);
}

const std::vector<real>& cos_twiddles(int N);
const std::vector<real>& sin_twiddles(int N);

#endif /* FFT_HPP_ */
