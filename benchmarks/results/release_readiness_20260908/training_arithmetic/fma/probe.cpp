
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
struct Node {
 bool is_leaf=true;
 bool is_categorical_split=false;
 int split_feature_id=-1,left_child=-1,right_child=-1;
 std::uint16_t split_bin_index=0;
 std::array<std::uint8_t,256> left_categories{};
 float leaf_weight=0;
 std::vector<float> leaf_weights;
};
struct Tree {std::vector<Node> values; const std::vector<Node>& nodes() const {return values;} bool is_vector_leaf() const {return !values.empty()&&!values.front().leaf_weights.empty();}};
struct LeafRowRange {std::size_t begin,end;};
void UpdatePredictionsFromLeafRanges(const Tree& tree,
                                     const std::vector<std::size_t>& row_indices,
                                     const std::vector<LeafRowRange>& leaf_row_ranges,
                                     double learning_rate,
                                     int prediction_dimension,
                                     int class_index,
                                     std::vector<float>& predictions) {
  const auto& nodes = tree.nodes();
  if (nodes.empty() || row_indices.empty() || leaf_row_ranges.size() < nodes.size()) {
    return;
  }
  if (tree.is_vector_leaf()) {
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      const Node& node = nodes[node_index];
      if (!node.is_leaf) continue;
      const LeafRowRange& range = leaf_row_ranges[node_index];
      for (std::size_t position = range.begin; position < range.end; ++position) {
        const std::size_t offset = row_indices[position] * static_cast<std::size_t>(prediction_dimension);
        for (int output = 0; output < prediction_dimension; ++output) {
          // Preserve the scalar path's separately rounded product. Fusing the
          // multiply/add changes the gradients and can select another split.
          const float update = static_cast<float>(learning_rate) *
                               node.leaf_weights[static_cast<std::size_t>(output)];
          predictions[offset + static_cast<std::size_t>(output)] += update;
        }
      }
    }
    return;
  }
  if (prediction_dimension == 1) {
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      const Node& node = nodes[node_index];
      if (!node.is_leaf) {
        continue;
      }
      const LeafRowRange& range = leaf_row_ranges[node_index];
      if (range.end <= range.begin) {
        continue;
      }
      const float update = static_cast<float>(learning_rate) * node.leaf_weight;
      for (std::size_t position = range.begin; position < range.end; ++position) {
        predictions[row_indices[position]] += update;
      }
    }
    return;
  }
  for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
    const Node& node = nodes[node_index];
    if (!node.is_leaf) {
      continue;
    }
    const LeafRowRange& range = leaf_row_ranges[node_index];
    if (range.end <= range.begin) {
      continue;
    }
    const float update = static_cast<float>(learning_rate) * node.leaf_weight;
    for (std::size_t position = range.begin; position < range.end; ++position) {
      const std::size_t row = row_indices[position];
      const std::size_t offset =
          row * static_cast<std::size_t>(prediction_dimension) + class_index;
      predictions[offset] += update;
    }
  }
}

void UpdatePredictionsFromLeafIndices(const Tree& tree,
                                      const std::vector<int>& leaf_indices,
                                      double learning_rate,
                                      int prediction_dimension,
                                      int class_index,
                                      std::vector<float>& predictions) {
  const auto& nodes = tree.nodes();
  if (nodes.empty() || leaf_indices.empty()) {
    return;
  }
  if (tree.is_vector_leaf()) {
    for (std::size_t row = 0; row < leaf_indices.size(); ++row) {
      const int leaf_index = leaf_indices[row];
      if (leaf_index < 0) continue;
      const Node& node = nodes[static_cast<std::size_t>(leaf_index)];
      const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension);
      for (int output = 0; output < prediction_dimension; ++output) {
        // Keep the same rounding boundary as scalar prediction, including the
        // external-memory, validation, and DART paths that reuse leaf indices.
        const float update = static_cast<float>(learning_rate) *
                             node.leaf_weights[static_cast<std::size_t>(output)];
        predictions[offset + static_cast<std::size_t>(output)] += update;
      }
    }
    return;
  }
  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < leaf_indices.size(); ++row) {
      const int leaf_index = leaf_indices[row];
      if (leaf_index >= 0) {
        const float update = static_cast<float>(learning_rate) *
                             nodes[static_cast<std::size_t>(leaf_index)].leaf_weight;
        predictions[row] += update;
      }
    }
    return;
  }
  for (std::size_t row = 0; row < leaf_indices.size(); ++row) {
    const int leaf_index = leaf_indices[row];
    if (leaf_index < 0) {
      continue;
    }
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension) + class_index;
    const float update = static_cast<float>(learning_rate) *
                         nodes[static_cast<std::size_t>(leaf_index)].leaf_weight;
    predictions[offset] += update;
  }
}

template <typename BinType>
int PredictContiguousLeafIndex(const std::vector<Node>& nodes,
                               const BinType* bin_indices,
                               std::size_t num_rows,
                               std::size_t row) noexcept {
  int node_index = 0;
  while (!nodes[static_cast<std::size_t>(node_index)].is_leaf) {
    const Node& node = nodes[static_cast<std::size_t>(node_index)];
    const std::size_t offset =
        static_cast<std::size_t>(node.split_feature_id) * num_rows + row;
    const std::uint16_t bin = static_cast<std::uint16_t>(bin_indices[offset]);
    node_index = node.is_categorical_split
                     ? (node.left_categories[bin] != 0 ? node.left_child : node.right_child)
                     : (bin <= node.split_bin_index ? node.left_child : node.right_child);
  }
  return node_index;
}

template <typename BinType>
void UpdatePredictionsFromContiguousBins(const Tree& tree,
                                         const BinType* bin_indices,
                                         std::size_t num_rows,
                                         double learning_rate,
                                         int prediction_dimension,
                                         int class_index,
                                         std::vector<float>& predictions) {
  const std::vector<Node>& nodes = tree.nodes();
  if (nodes.empty()) {
    return;
  }
  if (tree.is_vector_leaf()) {
    for (std::size_t row = 0; row < num_rows; ++row) {
      const int leaf_index = PredictContiguousLeafIndex(nodes, bin_indices, num_rows, row);
      const Node& node = nodes[static_cast<std::size_t>(leaf_index)];
      const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension);
      for (int output = 0; output < prediction_dimension; ++output) {
        // Match the scalar path's float rounding before addition. Combining
        // these expressions permits FMA contraction on targets such as ARM64.
        const float update = static_cast<float>(learning_rate) *
                             node.leaf_weights[static_cast<std::size_t>(output)];
        predictions[offset + static_cast<std::size_t>(output)] += update;
      }
    }
    return;
  }
  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < num_rows; ++row) {
      const int leaf_index = PredictContiguousLeafIndex(nodes, bin_indices, num_rows, row);
      const float update = static_cast<float>(learning_rate) *
                           nodes[static_cast<std::size_t>(leaf_index)].leaf_weight;
      predictions[row] += update;
    }
    return;
  }
  for (std::size_t row = 0; row < num_rows; ++row) {
    const int leaf_index = PredictContiguousLeafIndex(nodes, bin_indices, num_rows, row);
    const std::size_t offset =
        row * static_cast<std::size_t>(prediction_dimension) +
        static_cast<std::size_t>(class_index);
    const float update = static_cast<float>(learning_rate) *
                         nodes[static_cast<std::size_t>(leaf_index)].leaf_weight;
    predictions[offset] += update;
  }
}


std::uint32_t bits(float x){std::uint32_t b;std::memcpy(&b,&x,sizeof b);return b;}
float value(std::uint32_t b){float x;std::memcpy(&x,&b,sizeof x);return x;}
int main(){
 constexpr std::size_t count=1000; constexpr int dimension=3;
 const double learning_rate=.123456789;
 std::vector<float> initial(count*dimension);
 std::uint32_t rng=47;
 for(float& x:initial){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;x=(static_cast<int>(rng%20001)-10000)/577.0F;}
 const std::vector<float> weights{value(0x4129776d),value(0x412277bf),-0.34948291F};
 Tree scalar;scalar.values.resize(1); scalar.values[0].leaf_weight=weights[0];
 Tree vector=scalar;vector.values[0].leaf_weights=weights;
 std::vector<std::size_t> rows(count); for(std::size_t i=0;i<count;++i)rows[i]=i;
 const std::vector<LeafRowRange> ranges{{0,count}};
 const std::vector<int> indices(count,0);
 const std::vector<std::uint8_t> bins(count,0);
 auto scalar_ranges=initial,vector_ranges=initial,scalar_indices=initial,vector_indices=initial,scalar_hist=initial,vector_hist=initial,explicit_fma=initial;
 for(int k=0;k<dimension;++k){
   scalar.values[0].leaf_weight=weights[k];
   UpdatePredictionsFromLeafRanges(scalar,rows,ranges,learning_rate,dimension,k,scalar_ranges);
   UpdatePredictionsFromLeafIndices(scalar,indices,learning_rate,dimension,k,scalar_indices);
   UpdatePredictionsFromContiguousBins(scalar,bins.data(),count,learning_rate,dimension,k,scalar_hist);
   for(std::size_t row=0;row<count;++row)explicit_fma[row*dimension+k]=std::fma(static_cast<float>(learning_rate),weights[k],explicit_fma[row*dimension+k]);
 }
 UpdatePredictionsFromLeafRanges(vector,rows,ranges,learning_rate,dimension,0,vector_ranges);
 UpdatePredictionsFromLeafIndices(vector,indices,learning_rate,dimension,0,vector_indices);
 UpdatePredictionsFromContiguousBins(vector,bins.data(),count,learning_rate,dimension,0,vector_hist);
 auto compare=[&](const char* name,const std::vector<float>& a,const std::vector<float>& b){std::size_t mismatch=0;for(std::size_t i=0;i<a.size();++i)mismatch+=bits(a[i])!=bits(b[i]);std::printf("%s=%zu/%zu\n",name,mismatch,a.size());};
 compare("scalar_ranges_vs_hist",scalar_ranges,scalar_hist);
 compare("vector_ranges_vs_hist",vector_ranges,vector_hist);
 compare("scalar_indices_vs_hist",scalar_indices,scalar_hist);
 compare("vector_indices_vs_hist",vector_indices,vector_hist);
 compare("scalar_vs_vector_hist",scalar_hist,vector_hist);
 compare("explicit_fma_vs_hist",explicit_fma,scalar_hist);
 compare("scalar_vs_vector_ranges",scalar_ranges,vector_ranges);
}
