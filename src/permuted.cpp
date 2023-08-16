#include <fft/fft.hpp>
#include <fft/permuted.hpp>

std::vector<tint> invert(const std::vector<tint>& I) {
	std::vector<tint> J(I.size());
	for (int i = 0; i < I.size(); i++) {
		J[I[i]] = i;
	}
	return std::move(J);
}

std::vector<tint> relto(const std::vector<tint>& J, const std::vector<tint>& I) {
	std::vector<tint> K(I.size());
	for (tint i = 0; i < I.size(); i++) {
		for (tint j = 0; j < I.size(); j++) {
			if (I[j] == J[i]) {
				K[i] = j;
				break;
			}
		}
	}
	return std::move(K);
}

permuted_index::permuted_index(const std::vector<tint>& K) {
	permutation = K;
	permuted_bits = std::bitset < 64 > (0);
	natural_bits = std::bitset < 64 > (0);
	is_locked = std::bitset < 64 > (0);
	inverse = invert(permutation);
}

void permuted_index::set_natural(integer n) {
	natural_bits = std::bitset < 64 > (n);
	permuted_bits = std::bitset < 64 > (0);
	for (tint i = 0; i < permutation.size(); i++) {
		if (natural_bits.test(i)) {
			permuted_bits.set(permutation[i]);
		}
	}
}

integer permuted_index::get_natural() const {
	return natural_bits.to_ullong();
}

integer permuted_index::get_permuted() const {
	return permuted_bits.to_ullong();
}

permuted_index& permuted_index::operator++() {
	bool is_end = true;
	for (tint i = 0; i < permutation.size(); i++) {
		if (!is_locked.test(i)) {
			const tint j = permutation[i];
			if (natural_bits.test(i)) {
				natural_bits.reset(i);
				permuted_bits.reset(j);
			} else {
				natural_bits.set(i);
				permuted_bits.set(j);
				is_end = false;
				break;
			}
		}
	}
	if (is_end) {
		permuted_bits = natural_bits = std::bitset < 64 > (0x7fffffffffffffffULL);
	} else {
	}
	return *this;
}

permuted_index& permuted_index::operator++(int) {
	auto& ref = *this;
	operator++();
	return ref;
}

void permuted_index::set_permuted_rank(integer rank, integer nrank) {
	const integer log2nrank = std::ilogb(nrank);
	std::bitset < 64 > bits(rank);
	for (tint i = 0; i < log2nrank; i++) {
		const tint bit = permutation.size() + i - log2nrank;
		is_locked.set(inverse[bit]);
		if (bits.test(i)) {
			natural_bits.set(inverse[bit]);
			permuted_bits.set(bit);
		} else {
			natural_bits.reset(inverse[bit]);
			permuted_bits.reset(bit);
		}
	}
}

void permuted_index::set_natural_rank(integer rank, integer nrank) {
	const integer log2nrank = std::ilogb(nrank);
	std::bitset < 64 > bits(rank);
	for (tint i = 0; i < log2nrank; i++) {
		const tint bit = permutation.size() + i - log2nrank;
		is_locked.set(bit);
		if (bits.test(i)) {
			natural_bits.set(bit);
			permuted_bits.set(permutation[bit]);
		} else {
			natural_bits.reset(bit);
			permuted_bits.reset(permutation[bit]);
		}
	}
}

