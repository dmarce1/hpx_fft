#include <fft/fft.hpp>

const std::vector<real>& cos_twiddles(int N) {
	static hpx::mutex mtx;
	std::lock_guard < hpx::mutex > lock(mtx);
	static std::unordered_map<integer, std::shared_ptr<std::vector<real>>>values;
	auto i = values.find(N);
	if (i == values.end()) {
		std::vector<real> W(N);
		for (int n = 0; n < N; n++) {
			W[n] = std::cos(-2.0 * M_PI * n / N);
		}
		values[N] = std::make_shared < std::vector < real >> (std::move(W));
		i = values.find(N);
	}
	return *(i->second);
}

const std::vector<real>& sin_twiddles(int N) {
	static hpx::mutex mtx;
	std::lock_guard < hpx::mutex > lock(mtx);
	static std::unordered_map<integer, std::shared_ptr<std::vector<real>>>values;
	auto i = values.find(N);
	if (i == values.end()) {
		std::vector<real> W(N);
		for (int n = 0; n < N; n++) {
			W[n] = std::sin(-2.0 * M_PI * n / N);
		}
		values[N] = std::make_shared < std::vector < real >> (std::move(W));
		i = values.find(N);
	}
	return *(i->second);
}
