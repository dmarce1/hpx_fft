/*
 * fft.hpp
 *
 *  Created on: Aug 14, 2023
 *      Author: dmarce1
 */

#ifndef FFT_HPP_
#define FFT_HPP_

#define SIMD_SIZE 4

#ifndef __CUDACC__
#include <hpx/include/components.hpp>
#include <fftw3.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/channel.hpp>
#include <hpx/mutex.hpp>
#include <hpx/serialization.hpp>
#else
#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))
#endif

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

#include <fft/fft_vector.hpp>

#ifndef __CUDACC__

class fft_server: public hpx::components::managed_component_base<fft_server> {
	const int Nglobal;
	int Nrank;
	int Nlocal;
	std::array<int, NDIM> lb;
	std::array<int, NDIM> ub;
	std::array<std::vector<hpx::id_type>, NDIM> servers;
	fft_vector X;
	fft_vector Y;
	std::gslice gen_slice(int xi, int dim) const;
	int index(int i, int j, int k) const;
	void transform(const std::function<int(int)>&, int dim);
	std::pair<fft_vector_slice, fft_vector_slice> exchange(fft_vector_slice&& Xslice, fft_vector_slice&& Yslice, int i, int dim);
public:
	fft_server(int N, const std::array<int, NDIM>& pos);
	void apply_fft_1d(int dim, bool tw);
	void transpose_yz();
	void transpose(int dim);
	void scramble(int dim);
	void set_servers(std::array<std::vector<hpx::id_type>, NDIM>&&);
	void write(std::vector<std::complex<double>>&&, const std::array<int, NDIM>&, const std::array<int, NDIM>&);
	std::vector<std::complex<double>> read(const std::array<int, NDIM>&, const std::array<int, NDIM>&); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, apply_fft_1d); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, exchange); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, read); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, set_servers); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, transpose); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, transpose_yz); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, scramble); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, write);
};

class fft3d {
	const int Nglobal;
	int Nrank;
	int Nlocal;
	std::vector<std::vector<std::vector<hpx::id_type>>>servers;
public:
	fft3d(int, std::vector<hpx::id_type>&& );
	void transpose_yz();
	void transpose(int dim);
	void apply_fft_1d(int dim, bool tw);
	void scramble(int dim);
	void write(std::vector<std::complex<double>>&&, std::array<int, NDIM>, std::array<int, NDIM>);	//
	std::vector<std::complex<double>> read(std::array<int, NDIM>, std::array<int, NDIM>);//
};

extern "C" {
void transpose_zyx_asm(double* X, size_t N);
size_t bit_reverse(size_t, size_t);
void fft_batch_1d(double* X, double* Y, size_t N, size_t M);
void twiddles(double* X, double* Y, double C, double S, size_t M);
void scramble(double* X, size_t N);
void scramble_hi(double* X, size_t NHI, size_t NLO);
void transpose_re(double* X, size_t N1, size_t N2);
void transpose_hi(double* X, size_t N);
void fft_vector_2d(double* X, double* Y, const double** Wr, const double** Wi, size_t N, size_t N0);
}


#endif

void fft_3d_local(double* X, double* Y, int N);

void yield();
#endif /* FFT_HPP_ */
