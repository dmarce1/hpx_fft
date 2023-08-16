/*
 * permuted.hpp
 *
 *  Created on: Aug 14, 2023
 *      Author: dmarce1
 */

#ifndef PERMUTED_HPP_
#define PERMUTED_HPP_

#include <bitset>

std::vector<tint> invert(const std::vector<tint>& I);
std::vector<tint> relto(const std::vector<tint>& J, const std::vector<tint>& I);

class permuted_index {
	std::bitset<64> permuted_bits;
	std::bitset<64> natural_bits;
	std::bitset<64> is_locked;
	std::vector<tint> permutation;
	std::vector<tint> inverse;
public:
	permuted_index(const std::vector<tint>& K);
	void set_natural(integer n);
	integer get_natural() const;
	integer get_permuted() const;
	permuted_index& operator++();
	permuted_index& operator++(int);
	void lock_bit(tint bit, bool value);
	void set_permuted_rank(integer rank, integer nrank);
	void set_natural_rank(integer rank, integer nrank);
};

#endif /* PERMUTED_HPP_ */
