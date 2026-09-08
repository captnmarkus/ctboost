#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>

extern "C" __attribute__((noinline)) void legacy_update(
    const float* leaves, const int* indices, int count, double rate, float* predictions) {
  for (int row = 0; row < count; ++row) {
    const float update = static_cast<float>(rate) * leaves[indices[row]];
    predictions[row] += update;
  }
}
extern "C" __attribute__((noinline)) void prepare_cache(
    const float* leaves, int count, double rate, float* cached) {
  for (int leaf = 0; leaf < count; ++leaf) {
    cached[leaf] = static_cast<float>(rate) * leaves[leaf];
  }
}
extern "C" __attribute__((noinline)) void cached_update(
    const float* cached, const int* indices, int count, float* predictions) {
  for (int row = 0; row < count; ++row) {
    predictions[row] += cached[indices[row]];
  }
}
static std::uint32_t bits(float x) { std::uint32_t b; std::memcpy(&b,&x,sizeof b);return b; }
int main() {
  constexpr int n=1000,k=37;
  float leaves[k],cache[k],before[n],old[n],now[n];int indices[n];
  std::uint32_t state=47;
  auto random=[&]() { state^=state<<13;state^=state>>17;state^=state<<5;return state; };
  for (int i=0;i<k;++i) leaves[i]=(static_cast<int>(random()%20001)-10000)/791.0F;
  for (int i=0;i<n;++i) { before[i]=old[i]=now[i]=(static_cast<int>(random()%20001)-10000)/577.0F;indices[i]=random()%k; }
  const double rate=.123456789;
  prepare_cache(leaves,k,rate,cache);
  legacy_update(leaves,indices,n,rate,old);
  cached_update(cache,indices,n,now);
  int mismatches=0;
  for(int i=0;i<n;++i) if(bits(old[i])!=bits(now[i])) {
    if(mismatches==0) std::printf("first row=%d initial=%08x weight=%08x scale=%08x legacy=%08x cached=%08x fma=%08x\n",i,bits(before[i]),bits(leaves[indices[i]]),bits(static_cast<float>(rate)),bits(old[i]),bits(now[i]),bits(std::fma(static_cast<float>(rate),leaves[indices[i]],before[i])));
    ++mismatches;
  }
  std::printf("mismatches=%d/%d\n",mismatches,n);
}
