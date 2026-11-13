// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// Research-only byte-access discriminator. This does not execute a search engine.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using Range = std::pair<uint64_t, uint64_t>;
using Clock = std::chrono::steady_clock;
constexpr uint64_t ALIGN = 4096;

static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static std::vector<Range> plan(const std::vector<uint64_t>& offsets,
                               const std::vector<size_t>& selected,
                               uint64_t window, uint64_t span) {
    if (window == 0) return {{0, span}};
    if (window % ALIGN || window < ALIGN) throw std::runtime_error("unaligned window");
    std::vector<Range> ranges;
    for (size_t b : selected) {
        if (b + 1 >= offsets.size()) throw std::runtime_error("block outside offset table");
        uint64_t lo = offsets[b] / window * window;
        uint64_t hi = std::min(span, (offsets[b + 1] + window - 1) / window * window);
        ranges.push_back({lo, hi});
    }
    std::sort(ranges.begin(), ranges.end());
    std::vector<Range> out;
    for (auto r : ranges) {
        if (!out.empty() && r.first <= out.back().second) out.back().second = std::max(r.second, out.back().second);
        else out.push_back(r);
    }
    return out;
}

static std::vector<uint64_t> load_offsets(const std::string& path) {
    // The native exporter writes a JSON integer array. Parse only this exact grammar.
    std::ifstream f(path);
    std::string text((std::istreambuf_iterator<char>(f)), {});
    if (text.empty() || text.front() != '[' || text.back() != ']') throw std::runtime_error("invalid offsets array");
    std::vector<uint64_t> out;
    size_t p = 1;
    while (p < text.size() - 1) {
        while (p < text.size() && (text[p] == ' ' || text[p] == '\n')) ++p;
        size_t used = 0;
        auto value = std::stoull(text.substr(p), &used);
        out.push_back(value); p += used;
        while (p < text.size() && (text[p] == ' ' || text[p] == '\n')) ++p;
        if (p < text.size() - 1 && text[p++] != ',') throw std::runtime_error("invalid offsets separator");
    }
    if (out.size() < 2 || out.front() != 0 || !std::is_sorted(out.begin(), out.end())) throw std::runtime_error("invalid offsets order");
    return out;
}

static uint64_t hash_bytes(const unsigned char* data, size_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; ++i) h = (h ^ data[i]) * 1099511628211ULL;
    return h;
}

static double cpu_seconds() {
    rusage r{}; getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_stime.tv_sec + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e6;
}

static uint64_t proc_read_bytes() {
    std::ifstream f("/proc/self/io"); std::string key; uint64_t value;
    while (f >> key >> value) if (key == "read_bytes:") return value;
    return 0;
}

static void self_check() {
    const std::vector<uint64_t> offsets{0,100,4090,4200,9000};
    if (plan(offsets, {0},4096,12288) != std::vector<Range>{{0,4096}}) throw std::runtime_error("first block");
    if (plan(offsets, {2},4096,12288) != std::vector<Range>{{0,8192}}) throw std::runtime_error("crossing block");
    if (plan(offsets, {3,0},4096,12288) != std::vector<Range>{{0,12288}}) throw std::runtime_error("coalescing");
    if (plan(offsets, {},4096,12288) != std::vector<Range>{}) throw std::runtime_error("empty selection");
    if (plan(offsets, {0,3},16384,12288) != std::vector<Range>{{0,12288}}) throw std::runtime_error("tail clamp");
    bool rejected = false;
    try { plan(offsets, {4},4096,12288); } catch (const std::runtime_error&) { rejected=true; }
    if (!rejected) throw std::runtime_error("invalid block accepted");
    std::cout << "range contract checks passed\n";
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-check") { self_check(); return 0; }
        if (argc != 11) throw std::runtime_error("data offsets term_count requests selected_blocks window_bytes concurrency direct skew cluster");
        auto offsets=load_offsets(argv[2]);
        const uint64_t span=(offsets.back()+ALIGN-1)/ALIGN*ALIGN;
        const size_t terms=std::stoull(argv[3]), requests=std::stoull(argv[4]);
        const size_t wanted=std::stoull(argv[5]);
        const uint64_t window=std::stoull(argv[6]);
        const size_t concurrency=std::stoull(argv[7]);
        const bool direct=std::stoi(argv[8]), skew=std::stoi(argv[9]), cluster=std::stoi(argv[10]);
        if (!terms || !requests || !concurrency || wanted>offsets.size()-1) throw std::runtime_error("invalid dimensions");
        int fd=open(argv[1],O_RDONLY | (direct?O_DIRECT:0));
        if (fd<0) throw std::runtime_error(std::strerror(errno));
        struct stat info{}; if(fstat(fd,&info) || uint64_t(info.st_size)!=terms*span) throw std::runtime_error("data length mismatch");
        // Each expected block hash is computed from the native template outside timing.
        std::vector<unsigned char> reference(offsets.back());
        std::string template_path=argv[2];
        auto suffix=template_path.rfind(".offsets.json");
        if(suffix==std::string::npos) throw std::runtime_error("offset filename suffix");
        template_path.replace(suffix,std::string::npos,".blocks");
        int ref_fd=open(template_path.c_str(),O_RDONLY);
        if (pread(ref_fd,reference.data(),reference.size(),0)!=ssize_t(reference.size())) throw std::runtime_error("reference read failed");
        close(ref_fd);
        std::vector<uint64_t> expected;
        for(size_t b=0;b+1<offsets.size();++b) expected.push_back(hash_bytes(reference.data()+offsets[b],offsets[b+1]-offsets[b]));
        reference.clear(); reference.shrink_to_fit();
        std::atomic<size_t> next{0};
        std::atomic<uint64_t> io_bytes{0},io_requests{0},logical_bytes{0},checksum{0};
        std::vector<double> latencies(requests);
        std::mutex error_mutex; std::string error;
        double cpu_start=cpu_seconds(); uint64_t device_start=proc_read_bytes();
        auto start=Clock::now();
        auto worker=[&] {
            void* allocation=nullptr;
            if(posix_memalign(&allocation,ALIGN,span)) {std::lock_guard<std::mutex> g(error_mutex);error="allocation failed";return;}
            auto* buffer=static_cast<unsigned char*>(allocation);
            try {
                for(size_t q;(q=next.fetch_add(1))<requests;) {
                    uint64_t state=mix(q+42);
                    size_t term=state%terms;
                    if(skew && state%10<9) term=state%std::min<size_t>(8,terms);
                    std::set<size_t> selection;
                    size_t first=mix(state)%expected.size();
                    for(size_t k=0;selection.size()<wanted;++k) selection.insert(cluster?(first+k)%expected.size():mix(state+k)%expected.size());
                    std::vector<size_t> blocks(selection.begin(),selection.end());
                    auto ranges=plan(offsets,blocks,window,span);
                    auto query_start=Clock::now();
                    uint64_t sum=0,read=0,calls=0,logical=0;
                    size_t cursor=0;
                    for(auto [lo,hi]:ranges) {
                        auto got=pread(fd,buffer,hi-lo,term*span+lo);
                        if(got!=ssize_t(hi-lo)) throw std::runtime_error("short/failed read: "+std::string(std::strerror(errno)));
                        read+=got; ++calls;
                        while(cursor<blocks.size() && offsets[blocks[cursor]]<hi) {
                            size_t b=blocks[cursor++];
                            auto h=hash_bytes(buffer+offsets[b]-lo,offsets[b+1]-offsets[b]);
                            if(h!=expected[b]) throw std::runtime_error("payload verification failed");
                            sum+=h; logical+=offsets[b+1]-offsets[b];
                        }
                    }
                    if(cursor!=blocks.size()) throw std::runtime_error("selected block not covered");
                    latencies[q]=std::chrono::duration<double,std::milli>(Clock::now()-query_start).count();
                    io_bytes+=read;io_requests+=calls;logical_bytes+=logical;checksum+=sum;
                }
            } catch(const std::exception& e) {std::lock_guard<std::mutex> g(error_mutex);error=e.what();}
            free(allocation);
        };
        std::vector<std::thread> workers;
        for(size_t t=0;t<concurrency;++t) workers.emplace_back(worker);
        for(auto& t:workers)t.join();
        double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
        double cpu=cpu_seconds()-cpu_start; auto physical=proc_read_bytes()-device_start;
        if(!error.empty()) throw std::runtime_error(error);
        auto ordered=latencies;std::sort(ordered.begin(),ordered.end());
        auto percentile=[&](double p){return ordered[std::min(requests-1,size_t(p*(requests-1)))];};
        rusage usage{};getrusage(RUSAGE_SELF,&usage);
        std::cout<<"{\"requests\":"<<requests<<",\"selected_blocks\":"<<wanted<<",\"total_blocks\":"<<expected.size()
                 <<",\"term_span_bytes\":"<<span<<",\"window_bytes\":"<<window<<",\"concurrency\":"<<concurrency
                 <<",\"direct\":"<<direct<<",\"skew\":"<<skew<<",\"cluster\":"<<cluster
                 <<",\"elapsed_s\":"<<elapsed<<",\"qps\":"<<requests/elapsed<<",\"p50_ms\":"<<percentile(.50)
                 <<",\"p95_ms\":"<<percentile(.95)<<",\"p99_ms\":"<<percentile(.99)<<",\"cpu_s\":"<<cpu
                 <<",\"logical_selected_bytes\":"<<logical_bytes<<",\"syscall_bytes\":"<<io_bytes
                 <<",\"syscall_requests\":"<<io_requests<<",\"proc_read_bytes\":"<<physical
                 <<",\"peak_rss_kib\":"<<usage.ru_maxrss<<",\"checksum\":"<<checksum<<",\"latency_ms\":[";
        for(size_t i=0;i<latencies.size();++i){if(i)std::cout<<",";std::cout<<latencies[i];}
        std::cout<<"]}\n";close(fd);return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
