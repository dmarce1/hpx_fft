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
	void apply_fft(integer M) const;
	std::vector<real> read(integer xib, integer xie);
	void transpose(std::vector<tint> P);
	void scramble();
	void write(std::vector<real>&& Z, integer xib, integer xie);
};

class fft_server: public hpx::components::managed_component_base<fft_server> {

	std::vector<hpx::id_type> servers;
	std::vector<real> X;
	std::vector<std::vector<real>> Y;
	integer N;
	integer begin;
	integer end;
	integer rank;
	integer nrank;

public:

	fft_server();
	void apply_fft(integer M);
	void init(std::vector<hpx::id_type> servers_, integer rank_, integer N_);
	std::vector<real> read(integer xib, integer xie);
	void write(std::vector<real>&& Z, integer xib, integer xie);
	void transpose_set(std::vector<real>&& Z, integer orank);
	void transpose_begin(std::vector<tint> pindices);
	void transpose_end(std::vector<tint> pindices);
	//
	HPX_DEFINE_COMPONENT_ACTION(fft_server, apply_fft); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, init); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, read); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, write); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, transpose_set); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, transpose_begin); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, transpose_end);
	//

};

extern "C" {
void transpose_re(double*, integer);
void scramble_hi(double*, integer, integer);
void fft_1d(double*, integer, const double*, const double*);
}

const std::vector<real>& cos_twiddles(int N);
const std::vector<real>& sin_twiddles(int N);

#endif /* FFT_HPP_ */
