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
#include <hpx/serialization.hpp>
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

class swap_arc {
	std::vector<double> data;
	double* ptr;
	int len;
	bool remote;
public:
	inline swap_arc() {
		remote = false;
	}
	inline void set(double* p, int l) {
		ptr = p;
		len = l;
	}
	inline int size() const {
		return len;
	}
	inline void swap(double* other) {
		double* cur = remote ? data.data() : ptr;
		for (int i = 0; i < len; i++) {
			std::swap(cur[i], other[i]);
		}
	}
	inline void get(double* other) const {
		if (remote) {
			std::memcpy(other, data.data(), len * sizeof(double));
		}
	}
	template<class A>
	inline void save(A& arc, unsigned) const {
		const double* cur = remote ? data.data() : ptr;
		arc << len;
		for (int i = 0; i < len; i++) {
			arc << cur[i];
		}
	}
	template<class A>
	inline void load(A& arc, unsigned) {
		double* cur = remote ? data.data() : ptr;
		remote = true;
		arc >> len;
		data.resize(len);
		for (int i = 0; i < len; i++) {
			arc >> data[i];
		}
	}
	HPX_SERIALIZATION_SPLIT_MEMBER ();
};

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
	std::pair<swap_arc, swap_arc> exchange(swap_arc&&, swap_arc&&, int xi);
	void apply_fft();
	void apply_twiddles();
	void transform(const std::function<int(int)>&);
	void scramble();
	void transpose_x();
	void transpose_yxz();
	void transpose_zyx(); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, apply_twiddles); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, apply_fft); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, transpose_x); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, scramble); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, read); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, set_servers); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, write); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, transpose_yxz); //
	HPX_DEFINE_COMPONENT_ACTION(fft_server, transpose_zyx);HPX_DEFINE_COMPONENT_DIRECT_ACTION(fft_server, exchange);
	//
	//
};

class fft3d {
	const int Nglobal;
	int Nrank;
	int Nlocal;
	std::vector<std::vector<std::vector<hpx::id_type>>>servers;
public:
	fft3d(int, std::vector<hpx::id_type>&& );
	void apply_fft();
	void apply_twiddles();
	void scramble();
	void transpose_x();
	void transpose_yxz();
	void transpose_zyx(); //
	void write(std::vector<std::complex<double>>&&, std::array<int, NDIM>, std::array<int, NDIM>);//
	std::vector<std::complex<double>> read(std::array<int, NDIM>, std::array<int, NDIM>);//
};

extern "C" {
void transpose_zyx_asm(double* X, size_t N);
size_t bit_reverse(size_t, size_t);
void apply_fft_1d(double* X, double* Y, size_t N, size_t M);
void apply_twiddles_1d(double* X, double* Y, double C, double S, size_t M);
}

#endif /* FFT_HPP_ */
