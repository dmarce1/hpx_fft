/*
 * fft_vector.hpp
 *
 *  Created on: Aug 17, 2023
 *      Author: dmarce1
 */

#ifndef FFT_VECTOR_HPP_
#define FFT_VECTOR_HPP_

#define XDIM 0
#define YDIM 1
#define ZDIM 2
#define NDIM 3

#include <hpx/serialization.hpp>

class fft_vector_slice;

class fft_vector {
	std::vector<double> X;
	int N;
public:
	inline fft_vector();
	inline void resize(int n);
	inline double& operator()(int i, int j, int k);
	inline double operator()(int i, int j, int k) const;
	inline double* data();
	fft_vector_slice get_slice(int xi, int dim);
};

class fft_vector_slice {
	double* X;
	std::vector<double> data;
	int offset;
	int N;
	int stride1;
	int stride2;
	bool remote;
public:
	void assign2(const fft_vector_slice& other);
	void swap(const fft_vector_slice& other);
	template<class A>
	void serialize(A& arc, unsigned);
	friend class fft_vector;
};

inline fft_vector::fft_vector() {
}

inline void fft_vector::resize(int n) {
	N = n;
	X.resize(N * N * N);
}

inline double* fft_vector::data() {
	return X.data();
}

inline double& fft_vector::operator()(int i, int j, int k) {
	return X[k + N * (j + N * i)];
}

inline double fft_vector::operator()(int i, int j, int k) const {
	return X[k + N * (j + N * i)];
}

template<class A>
void fft_vector_slice::serialize(A& arc, unsigned) {
	if (!remote) {
		remote = true;
		data.reserve(N * N);
		for (int i1 = 0; i1 < N; i1++) {
			for (int i2 = 0; i2 < N; i2++) {
				data.push_back(X[offset + i1 * stride1 + i2 * stride2]);
			}
		}
	}
	arc & data;
	arc & offset;
	arc & N;
	arc & stride1;
	arc & stride2;
}

#endif /* FFT_VECTOR_HPP_ */
