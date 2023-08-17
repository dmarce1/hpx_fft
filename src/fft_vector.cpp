#include <fft/fft_vector.hpp>
#include <cassert>

fft_vector_slice fft_vector::get_slice(int i, int dim) {
	fft_vector_slice slice;
	slice.N = N;
	slice.X = X.data();
	slice.remote = false;
	switch (dim) {
	case XDIM:
		slice.offset = N * N * i;
		slice.stride1 = N;
		slice.stride2 = 1;
		break;
	case YDIM:
		slice.offset = N * i;
		slice.stride1 = N * N;
		slice.stride2 = 1;
		break;
	case ZDIM:
		slice.offset = i;
		slice.stride1 = N * N;
		slice.stride2 = N;
		break;
	}
	return slice;
}

void fft_vector_slice::assign2(const fft_vector_slice& other) {
	assert(!other.remote);
	if (remote) {
		int n = 0;
		for (int i1 = 0; i1 < N; i1++) {
			for (int i2 = 0; i2 < N; i2++) {
				const int m = other.offset + i1 * stride1 + i2 * stride2;
				other.X[m] = data[n++];
			}
		}
	} else {
		for (int i1 = 0; i1 < N; i1++) {
			for (int i2 = 0; i2 < N; i2++) {
				const int k = i1 * stride1 + i2 * stride2;
				const int n = offset + k;
				const int m = other.offset + k;
				other.X[m] = X[n];
			}
		}
	}
}

void fft_vector_slice::swap(const fft_vector_slice& other) {
	assert(!other.remote);
	if (remote) {
		int n = 0;
		for (int i1 = 0; i1 < N; i1++) {
			for (int i2 = 0; i2 < N; i2++) {
				const int m = other.offset + i1 * stride1 + i2 * stride2;
				std::swap(other.X[m], data[n++]);
			}
		}
	} else {
		for (int i1 = 0; i1 < N; i1++) {
			for (int i2 = 0; i2 < N; i2++) {
				const int k = i1 * stride1 + i2 * stride2;
				std::swap(X[offset + k], other.X[other.offset + k]);
			}
		}
	}
}
