#include "ctboost/tree.hpp"
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

int main() {
  int failures=0,checks=0;
  const auto check=[&](bool condition,const char* name) {
    ++checks;if(!condition){++failures;std::printf("FAIL %s\n",name);}
  };
  auto schema=std::make_shared<ctboost::QuantizationSchema>();
  schema->num_bins_per_feature={2,2};schema->cut_offsets={0,1,2};schema->cut_values={0,0};
  schema->categorical_mask={0,0};schema->missing_value_mask={0,0};
  std::vector<ctboost::Node> nodes(3);
  nodes[0].is_leaf=false;nodes[0].split_feature_id=1;nodes[0].split_bin_index=0;
  nodes[0].left_child=1;nodes[0].right_child=2;
  nodes[1].leaf_weight=2;nodes[2].leaf_weight=-3;
  ctboost::Tree tree;tree.LoadState(nodes,schema,{0,1});
  for(int bytes: {1,2}) {
    ctboost::HistMatrix hist;hist.num_rows=2;hist.num_cols=2;hist.bin_index_bytes=bytes;
    // Feature 0 is constant; feature 1 routes rows left/right. Keep both
    // columns allocated in the too-few-columns case so the pre-fix probe
    // safely detects missing validation without accessing unallocated memory.
    if(bytes==1) hist.compact_bin_indices={0,0,0,1};else hist.bin_indices={0,0,0,1};
    check(tree.PredictBinnedLeafIndex(hist,0)==1,"normal left leaf");
    check(tree.PredictBinnedLeafIndex(hist,1)==2,"normal right leaf");
    check(tree.PredictBinnedRow(hist,0)==2,"normal row value");
    std::vector<float> contrib(3,0);
    tree.AccumulateBinnedContributions(hist,1,0.5F,contrib);
    check(contrib==std::vector<float>({0,-1.5F,0}),"normal contribution");
    hist.num_cols=1;
    bool threw=false;try { tree.PredictBinnedLeafIndex(hist,0); }catch(const std::out_of_range&){threw=true;}
    check(threw,"too few columns leaf exception");
    threw=false;try { tree.PredictBinnedRow(hist,0); }catch(const std::out_of_range&){threw=true;}
    check(threw,"too few columns row exception");
    contrib.assign(3,0);threw=false;
    try { tree.AccumulateBinnedContributions(hist,0,1,contrib); }catch(const std::out_of_range&){threw=true;}
    check(threw,"too few columns contribution exception");
    // Fewer columns remain valid when this tree uses only an available one.
    nodes[0].split_feature_id=0;
    ctboost::Tree available;available.LoadState(nodes,schema,{1,0});
    check(available.PredictBinnedLeafIndex(hist,0)==1,"available feature checked fallback");
    contrib.assign(3,0);available.AccumulateBinnedContributions(hist,0,0.5F,contrib);
    check(contrib==std::vector<float>({1,0,0}),"available contribution checked fallback");
    nodes[0].split_feature_id=1;
    // Direct binned prediction also historically works without a tree schema.
    ctboost::Tree no_schema;no_schema.LoadState(nodes,nullptr,{0,1});hist.num_cols=2;
    check(no_schema.PredictBinnedLeafIndex(hist,1)==2,"no-schema checked fallback");
    contrib.assign(3,0);no_schema.AccumulateBinnedContributions(hist,1,0.5F,contrib);
    check(contrib==std::vector<float>({0,-1.5F,0}),"no-schema contribution fallback");
  }
  std::printf("checks=%d failures=%d\n",checks,failures);
  return failures ? 1 : 0;
}
