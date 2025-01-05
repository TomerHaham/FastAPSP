// Copyright 2023 The Fap Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <assert.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <set>
#include <stdexcept>

#include "fap/fap.h"
#include "fap/utils.h"
#include "fap/kernel.h"
#include <cuda.h>




#define checkCudaErrors(call)                                                                   \
{                                                                                               \
    cudaError_t cudaStatus = call;                                                              \
    if (cudaStatus != cudaSuccess)                                                              \
    {                                                                                           \
        std::cerr << "CUDA API error: " << cudaGetErrorString(cudaStatus) << " at "            \
                  << __FILE__ << " line " << __LINE__ << "." << std::endl;                      \
        exit(EXIT_FAILURE);                                                                     \
    }                                                                                           \
}

namespace fap {
std::atomic<int>* fapGraph::current_run = new std::atomic<int>(0);
fapGraph::~fapGraph() {
    // ... any existing cleanup
    delete current_run;  // Clean up the static atomic
}
fapGraph::fapGraph(std::string input_graph,
                  bool directed, bool weighted, int32_t K, std::string partitioner, bool version) {
  // build graph from file.
  int vertexs;
  int edges;

  readVerEdges(vertexs, edges, input_graph, directed, weighted);
  this->num_vertexs = vertexs;
  this->num_edges = edges;
  this->K = K;
  this->partitioner = partitioner;
  this->version = version;

  this->adj_size.resize(vertexs);
  this->row_offset.resize(vertexs + 1);
  this->col_val.resize(edges);
  this->weight.resize(edges);

  this->graph_id.assign(vertexs, 0);
  this->st2ed.assign(vertexs, 0);
  this->ed2st.assign(vertexs, 0);
  this->C_BlockVer_num.assign(K + 1, 0);
  this->C_BlockVer_offset.assign(K + 1, 0);
  this->C_BlockBdy_num.assign(K + 1, 0);
  this->C_BlockBdy_offset.assign(K + 1, 0);

  // get message of Graph from input file.
  readMatFile(vertexs, edges,
      adj_size.data(), row_offset.data(), col_val.data(), weight.data(),
      input_graph, directed, weighted);
  // get subgraph message from metis.
  readIdFile_METIS(
      graph_id.data(), BlockVer, K, directed, vertexs, edges,
      adj_size.data(), row_offset.data(), col_val.data(), weight.data(), partitioner);
}

fapGraph::fapGraph(int32_t num_vertexs, int64_t num_edges,
                std::vector<int32_t> row_offset, std::vector<int32_t> col_val,
                std::vector<float> weight, int32_t K) {
  // build graph from metadata.
}

void fapGraph::preCondition() {
  findBoundry(this->K, this->num_vertexs, this->graph_id.data(),
            this->adj_size.data(), this->row_offset.data(),
            this->col_val.data(), this->weight.data(),
            this->BlockVer, this->BlockBoundary, this->isBoundry);

  sort_and_encode(this->K, this->num_vertexs,
                  this->graph_id.data(), this->isBoundry,
                  this->C_BlockVer_num.data(), this->C_BlockVer_offset.data(),
                  this->C_BlockBdy_num.data(), this->C_BlockBdy_offset.data(),
                  this->BlockVer, this->BlockBoundary,
                  this->st2ed.data(), this->ed2st.data());

  this->is_split = (this->num_vertexs > this->max_vertexs_num_limit);
}

bool fapGraph::isSplit() {
  return this->is_split;
}

std::vector<int> fapGraph::getGraphId() {
  return this->graph_id;
}

// Add this to your utility functions
inline bool isPointerAligned(const void* ptr, size_t alignment = 16) {
    return ((uintptr_t)ptr % alignment) == 0;
}
template<typename T>
T* fapGraph::calculateDeviceOffset(T* base_ptr, size_t offset, size_t total_elements, const char* ptr_name) {
    // Ensure offset is aligned to 16 bytes
    const size_t alignment = 16 / sizeof(T);
    size_t aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
    
    if (!base_ptr || aligned_offset >= total_elements) {
        printf("Invalid parameters for %s: base=%p, offset=%zu, aligned=%zu, total=%zu\n",
               ptr_name, (void*)base_ptr, offset, aligned_offset, total_elements);
        return nullptr;
    }

    T* offset_ptr = base_ptr + aligned_offset;

    // Verify alignment
    if ((uintptr_t)offset_ptr % 16 != 0) {
        printf("Alignment error for %s: result pointer not 16-byte aligned\n", ptr_name);
        return nullptr;
    }

    // Verify it's a valid device pointer
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, offset_ptr) != cudaSuccess) {
        printf("Invalid device pointer for %s\n", ptr_name);
        return nullptr;
    }

    printf("Offset calculation for %s: base=%p, offset=%zu, aligned=%zu, result=%p\n",
           ptr_name, (void*)base_ptr, offset, aligned_offset, (void*)offset_ptr);
    return offset_ptr;
}
// Explicit instanti// worked but with problem in the kernel 
/* 
template<typename T>
    T* fapGraph::calculateDeviceOffset(T* base_ptr, size_t offset_elements, 
                                     size_t total_elements, const char* ptr_name) {
        // Validate base pointer
        cudaPointerAttributes attr;
        if (cudaPointerGetAttributes(&attr, base_ptr) != cudaSuccess || 
            attr.type != cudaMemoryTypeDevice) {
            printf("Invalid base pointer %s\n", ptr_name);
            return nullptr;
        }

        // Check if offset would exceed allocated memory
        if (offset_elements >= total_elements) {
            printf("Offset exceeds total elements for %s: offset=%zu, total=%zu\n",
                   ptr_name, offset_elements, total_elements);
            return nullptr;
        }

        T* offset_ptr = base_ptr + offset_elements;

        // Validate resulting pointer
        if (cudaPointerGetAttributes(&attr, offset_ptr) != cudaSuccess || 
            attr.type != cudaMemoryTypeDevice) {
            printf("Invalid offset pointer for %s\n", ptr_name);
            return nullptr;
        }

        printf("Calculated offset for %s: base=%p, offset=%zu, result=%p\n",
               ptr_name, (void*)base_ptr, offset_elements, (void*)offset_ptr);

        return offset_ptr;
    }
*/
// run fast APSP algorithm.
void fapGraph::run(float *subgraph_dist,
                      int *subgraph_path,
                      const int32_t sub_graph_id) {
    

  const int bdy_num = C_BlockBdy_num[sub_graph_id];
  const int sub_vertexs = C_BlockVer_num[sub_graph_id];
  const int inner_num = sub_vertexs - bdy_num;
  int64_t subgraph_dist_size = (int64_t)sub_vertexs * this->num_vertexs;
  int64_t inner_to_bdy_size = (int64_t)inner_num * bdy_num;
  int64_t inner_to_inner_size = (int64_t)sub_vertexs * sub_vertexs;
  vector<float> inner_to_bdy_dist(inner_to_bdy_size, fap::MAXVALUE);
  vector<float> inner_to_inner_dist(inner_to_inner_size, fap::MAXVALUE);
  vector<int> inner_to_inner_path(inner_to_inner_size, -1);
      // new version
      if (this->version){
 float *d_res = nullptr;
    float *d_mat1 = nullptr;
    int *d_rowOffsetArc = nullptr, *d_colValueArc = nullptr;
    float *d_weightArc = nullptr;
    float *d_subMat = nullptr;
    int *d_subMat_path = nullptr;
    int *d_graph_id = nullptr, *d_st2ed = nullptr, *d_ed2st = nullptr;
    int *d_adj_size = nullptr;
    int *d_subGraph_path = nullptr;
        try {

        checkCudaErrors(cudaMalloc((void **)&d_res, subgraph_dist_size * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_rowOffsetArc, this->num_vertexs * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_colValueArc, this->num_edges * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_weightArc, this->num_edges * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_rowOffsetArc, row_offset.data(), this->num_vertexs * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_colValueArc, col_val.data(), this->num_edges * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_weightArc, weight.data(), this->num_edges * sizeof(float), cudaMemcpyHostToDevice));


        // Handle boundary data on GPU
        handle_boundry_path_data_on_gpu(subgraph_dist, subgraph_path,
                            this->num_vertexs, this->num_edges, bdy_num,
                            adj_size.data(), row_offset.data(), col_val.data(),
                            weight.data(), st2ed.data(), C_BlockVer_offset[sub_graph_id],
                            d_res, d_rowOffsetArc, d_colValueArc, d_weightArc);
        checkCudaErrors(cudaDeviceSynchronize());

        // Allocate and copy data for graph arrays
        checkCudaErrors(cudaMalloc((void**)&d_graph_id, graph_id.size() * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&d_st2ed, st2ed.size() * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&d_ed2st, ed2st.size() * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&d_adj_size, adj_size.size() * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&d_subMat, sub_vertexs * sub_vertexs * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&d_subMat_path, sub_vertexs * sub_vertexs * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&d_subGraph_path, subgraph_dist_size * sizeof(int)));
        checkCudaErrors(cudaMemcpy(d_subGraph_path, subgraph_path, sub_vertexs * this->num_vertexs * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_graph_id, graph_id.data(), graph_id.size() * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_st2ed, st2ed.data(), st2ed.size() * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_ed2st, ed2st.data(), ed2st.size() * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_adj_size, adj_size.data(), adj_size.size() * sizeof(int), cudaMemcpyHostToDevice));

        int start = C_BlockVer_offset[sub_graph_id];

        // Launch kernel
        LaunchMysubMatBuildKernel(
            d_subMat, d_subMat_path, d_res,
            d_subGraph_path, d_rowOffsetArc, d_colValueArc,
            d_weightArc, sub_vertexs, bdy_num, this->num_vertexs,
            d_graph_id, d_st2ed, d_ed2st, d_adj_size, graph_id, start, this->num_edges);

        checkCudaErrors(cudaFree(d_graph_id));
        checkCudaErrors(cudaFree(d_ed2st));
        checkCudaErrors(cudaFree(d_adj_size));
        checkCudaErrors(cudaFree(d_rowOffsetArc));
        checkCudaErrors(cudaFree(d_weightArc));
        checkCudaErrors(cudaFree(d_colValueArc));

        // Note: d_subMat, d_subMat_path, d_res, and d_subGraph_path are kept alive
        // for use in subsequent operations

    } catch (const std::runtime_error& e) {
        printf("Error in fapGraph::run: %s\n", e.what());
        // Clean up any allocated memory in case of error
        if (d_subMat) checkCudaErrors(cudaFree(d_subMat));
        if (d_subMat_path) checkCudaErrors(cudaFree(d_subMat_path));
        if (d_subGraph_path) checkCudaErrors(cudaFree(d_subGraph_path));
        if (d_res) checkCudaErrors(cudaFree(d_res));
        throw;
    }

    // 2.2 run floyd algorithm
// 2.2 run floyd algorithm
        fap::floyd_path_gpu(sub_vertexs, d_subMat, d_subMat_path);
        
        // Copy floyd results back to host



        // 3.1 Setup mat1 for min-plus
        checkCudaErrors(cudaMalloc((void**)&d_mat1, inner_num * bdy_num * sizeof(float)));
        LaunchMyMat1BuildKernel(d_mat1, d_subMat, inner_num, bdy_num, sub_vertexs);
        

        // 3.2 Min-plus computation
        int64_t offset = (int64_t)bdy_num * this->num_vertexs;
        const double GPU_MAX_NUM = 4e9;
        const double MEM_NUM = GPU_MAX_NUM / this->num_vertexs - bdy_num;
        int part_num = 1;
        #ifdef WITH_GPU
        part_num = static_cast<int>(ceil(static_cast<double>(inner_num) / MEM_NUM));
        #endif

if (part_num == 1) {
    fap::min_plus_path_advanced_gpu(
        d_mat1,           // GPU pointer for inner_to_bdy_dist
        d_res,            // GPU pointer for subgraph_dist
        d_subGraph_path,  // GPU pointer for subgraph_path
        d_res + offset,   // GPU pointer for subgraph_dist + offset
        d_subGraph_path + offset, // GPU pointer for subgraph_path + offset
        inner_num, this->num_vertexs, bdy_num);
} else {
    int block_size = inner_num / part_num;
    int last_size = inner_num - block_size * (part_num - 1);

    for (int i = 0; i < part_num; i++) {
        int64_t offset_value = offset + (int64_t)i * block_size * this->num_vertexs;
        if (i == part_num - 1) {
            fap::min_plus_path_advanced_gpu(
                d_mat1 + i * block_size * bdy_num,
                d_res,
                d_subGraph_path,
                d_res + offset_value,
                d_subGraph_path + offset_value,
                last_size, this->num_vertexs, bdy_num);
        } else {
            fap::min_plus_path_advanced_gpu(
                d_mat1 + i * block_size * bdy_num,
                d_res,
                d_subGraph_path,
                d_res + offset_value,
                d_subGraph_path + offset_value,
                block_size, this->num_vertexs, bdy_num);
        }
    }
}

        // 3.3 Final decode
LaunchMysubMatDecodePathKernel(
    d_subMat, d_subMat_path,
    d_res, d_subGraph_path,
    d_st2ed, C_BlockVer_offset[sub_graph_id],
    sub_vertexs, sub_vertexs, this->num_vertexs);

            checkCudaErrors(cudaMemcpy(subgraph_dist, d_res, 
                              subgraph_dist_size * sizeof(float), 
                              cudaMemcpyDeviceToHost));
    
    checkCudaErrors(cudaMemcpy(subgraph_path, d_subGraph_path,
                              subgraph_dist_size * sizeof(int),
                              cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(d_res));
    checkCudaErrors(cudaFree(d_subGraph_path));

        // Cleanup GPU resources
        if (d_mat1) checkCudaErrors(cudaFree(d_mat1));
        if (d_subMat) checkCudaErrors(cudaFree(d_subMat));
        if (d_subMat_path) checkCudaErrors(cudaFree(d_subMat_path));
        if (d_st2ed) checkCudaErrors(cudaFree(d_st2ed));

}else{

  // stage 1. run sssp algorithm in boundry points.
  fap::handle_boundry_path(
            subgraph_dist, subgraph_path,
            this->num_vertexs, this->num_edges, bdy_num,
            adj_size.data(), row_offset.data(), col_val.data(), weight.data(),
            st2ed.data(), C_BlockVer_offset[sub_graph_id]);

  // stage 2. run floyd algorithm in all points of subgraph.
  // 2.1 move data from subgraph matrix to floyd matrix.
  fap::MysubMatBuild_path(
      inner_to_inner_dist.data(), inner_to_inner_path.data(),
      subgraph_dist, subgraph_path,
      graph_id.data(), C_BlockVer_offset[sub_graph_id],
      sub_vertexs, bdy_num,
      this->num_vertexs, st2ed.data(), ed2st.data(),
      adj_size.data(), row_offset.data(), col_val.data(), weight.data());

  // 2.2 run floyd algorithm
  fap::floyd_path(sub_vertexs,
    inner_to_inner_dist.data(), inner_to_inner_path.data());

  // stage 3. run min-plus algorithm in boundry points to all other points.
  // 3.1 move data from subMat to a thin matrix.
  fap::MyMat1Build(inner_to_bdy_dist.data(), inner_to_inner_dist.data(),
            inner_num, bdy_num, sub_vertexs);
  int64_t offset = (int64_t)bdy_num * this->num_vertexs;

  // 3.2 run min-plus
  // GPU global mem / sizeof(float)
  const double GPU_MAX_NUM = 4e9;
  const double MEM_NUM = GPU_MAX_NUM / this->num_vertexs - bdy_num;
  int part_num = 1;
#ifdef WITH_GPU
  part_num = static_cast<int>(
      ceil(static_cast<double>(inner_num) / MEM_NUM));
#endif
  // GPU memory is expansive and maybe too big.
  if (part_num == 1) {
      fap::min_plus_path_advanced(
          inner_to_bdy_dist.data(),
          subgraph_dist, subgraph_path,
          subgraph_dist + offset,
          subgraph_path + offset,
          inner_num, this->num_vertexs, bdy_num);
  } else {
      int block_size = inner_num / part_num;
      int last_size = inner_num - block_size * (part_num - 1);

      for (int i = 0; i < part_num; i++) {
        int64_t offset_value = offset +
          (int64_t)i * block_size * this->num_vertexs;
        if (i == part_num - 1) {
            fap::min_plus_path_advanced(
              inner_to_bdy_dist.data() + i * block_size * bdy_num,
              subgraph_dist, subgraph_path,
              subgraph_dist + offset_value,
              subgraph_path + offset_value,
              last_size, this->num_vertexs, bdy_num);
        } else {
            fap::min_plus_path_advanced(
              inner_to_bdy_dist.data() + i * block_size * bdy_num,
              subgraph_dist, subgraph_path,
              subgraph_dist + offset_value,
              subgraph_path + offset_value,
              block_size, this->num_vertexs, bdy_num);
        }
      }
  }

  // 3.3 move data from floyd matrix to subgraph matrix.
  fap::MysubMatDecode_path(
      inner_to_inner_dist.data(), inner_to_inner_path.data(),
      subgraph_dist, subgraph_path,
      C_BlockVer_offset[sub_graph_id], sub_vertexs,
      sub_vertexs, this->num_vertexs, st2ed.data());
}
}


// run fast APSP algorithm.
int32_t fapGraph::solve(bool is_path_needed) {  // <-- Need ; here
    int64_t graph_size = (int64_t)num_vertexs * num_vertexs;
    dist.resize(graph_size);
    path.resize(graph_size);
    int64_t offset = 0;
    for (int i = 1; i <= K; i++) {
        run(dist.data() + offset, path.data() + offset, i);
        int64_t size = (int64_t)num_vertexs * C_BlockVer_num[i];
        offset += size;
    }
    return 0;
}
// run fast APSP algorithm in one subgraph.
int32_t fapGraph::solveSubGraph(int32_t sub_graph_id, bool is_path_needed) {
    // Add a static set to track which subgraphs have been processed
    static std::set<int32_t> processed_subgraphs;
    
    // Check if this subgraph has already been processed
    if (processed_subgraphs.find(sub_graph_id) != processed_subgraphs.end()) {
        printf("Warning: Subgraph %d has already been processed, skipping.\n", sub_graph_id);
        return 0;
    }

    // init data.
    const int sub_vertexs = C_BlockVer_num[sub_graph_id];
    int64_t subgraph_dist_size = (int64_t)sub_vertexs * this->num_vertexs;
    this->subgraph_dist.resize(subgraph_dist_size);
    this->subgraph_path.resize(subgraph_dist_size);
    this->current_subgraph_id = sub_graph_id;

    // Mark as processed before running
    processed_subgraphs.insert(sub_graph_id);
    
    //printf("Starting execution of subgraph %d\n", sub_graph_id);
    run(subgraph_dist.data(), subgraph_path.data(), sub_graph_id);
    //printf("Completed execution of subgraph %d\n", sub_graph_id);
    
    return 0;
}
std::vector<int32_t> fapGraph::getMapping() {
  return this->st2ed;
}

std::vector<float> fapGraph::getAllDistance() {
  assert(this->is_split == false);
  return this->dist;
}

std::vector<int32_t> fapGraph::getAllPath() {
  assert(this->is_split == false);
  return this->path;
}

std::vector<int32_t> fapGraph::getSubGraphIndex(int32_t sub_graph_id) {
  int offset = C_BlockVer_offset[sub_graph_id];
  int size = C_BlockVer_num[sub_graph_id];
  std::vector<int32_t> sub_graph_st2ed(st2ed.begin() + offset,
                                        st2ed.begin() + offset + size);
  return sub_graph_st2ed;
}

int32_t fapGraph::getCurrentSubGraphId() {
  return this->current_subgraph_id;
}

std::vector<float> fapGraph::getSubGraphDistance(int32_t sub_graph_id) {
  assert(this->current_subgraph_id == sub_graph_id);
  return this->subgraph_dist;
}

std::vector<int32_t> fapGraph::getSubGraphPath(int32_t sub_graph_id) {
  assert(this->current_subgraph_id == sub_graph_id);
  return this->subgraph_path;
}

std::vector<float> fapGraph::getDistanceFromOnePoint(int32_t vectex_id) {
  assert(this->graph_id[vectex_id] == this->current_subgraph_id);
  int32_t current_index = this->ed2st[vectex_id]
                                  - this->C_BlockVer_offset[vectex_id];
  std::vector<float> result(subgraph_dist.begin() + current_index,
      subgraph_dist.begin() + current_index + this->num_vertexs);
  return result;
}

std::vector<int32_t> fapGraph::getPathFromOnePoint(int32_t vectex_id) {
  assert(this->graph_id[vectex_id] == this->current_subgraph_id);
  int32_t current_index = this->ed2st[vectex_id]
                                  - this->C_BlockVer_offset[vectex_id];
  std::vector<int32_t> result(subgraph_path.begin() + current_index,
      subgraph_path.begin() + current_index + this->num_vertexs);
  return result;
}

// Tomer added
void fapGraph::printResalts(){
  for (int32_t i = 0; i < num_vertexs; ++i){
    for (int32_t j = 0; j < num_vertexs; ++j ){
      printf("Distances from node %d: ", i);
      printf(" to node %d: ", j);
      printf(" to node %d: ", j);
      printf("%.2f ", this->dist[num_vertexs*i + j]);
      printf("\n");
    }
  }
}
/*
void fapGraph::printResalts(){
  for (int32_t i = 0; i < num_vertexs; ++i){
    printf("Distances from node %d:\n", i + 1);
    for (int32_t j = 0; j < num_vertexs; ++j ){
      printf("  to node %d: %.2f\n", j + 1, dist[num_vertexs * i + j]);
    }
  }
}
*/

float fapGraph::getDistanceP2P() {
  // TODO(Liu-xiandong):
}

std::vector<int32_t> fapGraph::getPathP2P() {
  // TODO(Liu-xiandong):
}

bool fapGraph::check_result(float *dist, int *path,
               int *source, int source_num, int vertexs,
               int *adj_size, int *row_offset,
               int *col_val, float *weight, int *graph_id) {
  return fap::check_ans(dist, path, source, source_num, vertexs,
          adj_size, row_offset, col_val, weight, graph_id);
}
}  // namespace fap