#include <fft/fft.hpp>
#include <fft/permuted.hpp>

fft::fft(integer N_, std::vector<hpx::id_type> localities) {
	N = N_;
	nrank = localities.size();
	servers.resize(nrank);
	Nperrank = N / nrank;
	std::vector < hpx::future < hpx::id_type >> futs1(nrank);
	std::vector<hpx::future<void>> futs2(nrank);
	for (integer n = 0; n < nrank; n++) {
		futs1[n] = hpx::new_ < fft_server > (localities[n]);
	}
	for (integer n = 0; n < nrank; n++) {
		servers[n] = futs1[n].get();
	}
	for (integer n = 0; n < nrank; n++) {
		futs2[n] = hpx::async<typename fft_server::init_action>(servers[n], servers, n, N);
	}
	hpx::wait_all(futs2.begin(), futs2.end());
}

void fft::apply_fft(integer M1, integer M0, integer L) const {
	std::vector<hpx::future<void>> futs(nrank);
	for (integer n = 0; n < nrank; n++) {
		futs[n] = hpx::async<typename fft_server::apply_fft_action>(servers[n], M1, M0, L);
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft::fft_2d() {
	std::vector<tint> P0;
	std::vector<tint> P1;
	std::vector<tint> P2;
	const integer M = std::lround(std::sqrt(N >> 1));
	const tint log2N = std::ilogb(N);
	const tint log2M = std::ilogb(M);
	const tint log2S = std::ilogb(SIMD_SIZE);
	P0.resize(log2N);
	P1.resize(log2N);
	P2.resize(log2N);
	for (integer i = 0; i < log2S; i++) {
		P1[i] = i + 1;
		P0[i] = i + log2M + 1;
	}
	P0[log2S] = 0;
	P1[log2S] = 0;
	for (integer i = log2S + 1; i < log2S + 1 + log2M; i++) {
		P1[i] = log2N - i + log2S;
		P0[i] = 1 + log2M - i + log2S;
	}
	for (integer i = log2S + 1 + log2M; i < log2N; i++) {
		P1[i] = i - log2M;
		P0[i] = i;
	}
	P2[0] = 0;
	for (int i = 1; i < log2M + 1; i++) {
		P2[i] = log2M + 1 - i;
		P2[log2M + i] = log2N - i;
	}
	P2 = relto(P2, P1);
	P1 = relto(P1, P0);
	for (int i = 0; i < P0.size(); i++) {
	//	printf("%2i %2i %2i %2i\n", i, P0[i], P1[i], P2[i]);
	}
//	printf("!\n");
	for (int i = 0; i < P0.size(); i++) {
	//	printf("%2i %2i %2i %2i\n", i, P0[i], P0[P1[i]], P0[P1[P2[i]]]);
	}
	transpose(P0);
	apply_fft(M, 1, SIMD_SIZE);
	transpose(P1);
	apply_fft(M, 1, SIMD_SIZE);
	transpose(P2);
}

std::vector<real> fft::read(integer xib, integer xie) {
	const integer nx = xie - xib;
	std::vector<real> Z(nx);
	std::vector < hpx::future<std::vector<real>> > futs;
	const integer rb = xib / Nperrank;
	const integer re = xie / Nperrank;
	for (integer r = rb; r < re; r++) {
		const integer ib = std::max(xib, r * Nperrank);
		const integer ie = std::min(xie, (r + 1) * Nperrank);
		futs.push_back(hpx::async<typename fft_server::read_action>(servers[r], ib, ie));
	}
	for (integer r = rb; r < re; r++) {
		const integer ib = std::max(xib, r * Nperrank);
		const integer ie = std::min(xie, (r + 1) * Nperrank);
		const auto zr = futs[r - rb].get();
		for (integer i = ib; i < ie; i++) {
			Z[i - xib] = zr[i - r * Nperrank];
		}
	}
	return Z;
}

void fft::transpose(std::vector<tint> P) {
	P = invert(P);
	std::vector<hpx::future<void>> futs(servers.size());
	for (integer i = 0; i < servers.size(); i++) {
		futs[i] = hpx::async<typename fft_server::transpose_action>(servers[i], P);
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft::write(std::vector<real>&& Z, integer xib, integer xie) {
	const integer nx = xie - xib;
	std::vector < hpx::future<void> > futs;
	const integer rb = xib / Nperrank;
	const integer re = xie / Nperrank;
	for (integer r = rb; r < re; r++) {
		const integer ib = std::max(xib, r * Nperrank);
		const integer ie = std::min(xie, (r + 1) * Nperrank);
		std::vector<real> zr(ie - ib);
		for (integer i = ib; i < ie; i++) {
			zr[i - r * Nperrank] = Z[i - xib];
		}
		futs.push_back(hpx::async<typename fft_server::write_action>(servers[r], std::move(zr), ib, ie));
	}
	hpx::wait_all(futs.begin(), futs.end());
}
