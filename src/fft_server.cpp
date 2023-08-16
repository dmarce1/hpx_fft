#include <fft/fft.hpp>
#include <fft/permuted.hpp>

fft_server::fft_server() {
}

hpx::future<void> fft_server::apply_fft(integer M) {
	return hpx::async([M, this]() {
		const auto& Wr = cos_twiddles(M);
		const auto& Wi = sin_twiddles(M);
		for (integer i = begin; i < end; i += 2 * M * SIMD_SIZE) {
			fft_1d(X.data() + i - begin, M, Wr.data(), Wi.data());
		}
	});
}

void fft_server::init(std::vector<hpx::id_type> servers_, integer rank_, integer N_) {
	servers = std::move(servers_);
	N = N_;
	rank = rank_;
	begin = rank * N / servers.size();
	end = (rank + 1) * N / servers.size();
	X.resize(N);
	nrank = servers.size();
	channels.resize(nrank);
}

std::vector<real> fft_server::read(integer xib, integer xie) {
	std::vector<real> Z(xie - xib);
	for (integer i = xib; i < xie; i++) {
		const integer j = i - xib;
		const integer k = i - begin;
		Z[j] = X[k];
	}
	return std::move(Z);
}

void fft_server::write(std::vector<real>&& Z, integer xib, integer xie) {
	for (integer i = xib; i < xie; i++) {
		const integer j = i - xib;
		const integer k = i - begin;
		X[k] = Z[j];
	}
}

void fft_server::set_channel(std::vector<real>&& Z, integer orank) {
	channels[orank].set(std::move(Z));
}

hpx::future<void> fft_server::transpose(std::vector<tint> pindices) {
	permuted_index p(std::move(pindices));
	std::unordered_map<integer, std::vector<real>> sends;
	std::vector<hpx::future<void>> futs;
	for (p.set_natural(begin); p.get_natural() < end; ++p) {
		const integer i = p.get_natural();
		const integer j = p.get_permuted();
		const integer orank = j * nrank / N;
		sends[orank].push_back(X[i - begin]);
	}
	for (auto si = sends.begin(); si != sends.end(); si++) {
		hpx::post<typename fft_server::set_channel_action>(servers[si->first], std::move(si->second), rank);
	}
	for (integer orank = 0; orank < nrank; orank++) {
		permuted_index p(pindices);
		const integer obegin = orank * N / nrank;
		const integer oend = (orank + 1) * N / nrank;
		p.set_natural(obegin);
		p.set_permuted_rank(rank, nrank);
		p.set_natural_rank(orank, nrank);
		if (p.get_natural() >= obegin && p.get_natural() < oend) {
			auto yfut = channels[orank].get();
			futs.push_back(yfut.then([p, obegin, oend, this](hpx::future<std::vector<real>>&& fut) {
				integer k = 0;
				const auto& y = fut.get();
				auto P = p;
				while (P.get_natural() < oend) {
					const integer i = P.get_natural();
					const integer j = P.get_permuted();
					assert(j < end);
					assert(j >= begin);
					assert(k >= 0);
					assert(k < y.size());
					assert(j - begin < X.size());
					X[j - begin] = y[k++];
					P++;
				}
			}));
		}
	}
	return hpx::when_all(futs.begin(), futs.end());
}

