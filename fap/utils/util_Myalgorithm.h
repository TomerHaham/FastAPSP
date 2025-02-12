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

#ifndef UTIL_IMPROVED_H_
#define UTIL_IMPROVED_H_

#pragma once

#include <string.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <assert.h>
#include <stdio.h> // For debugging purposes

#include "fap/utils/parameter.h"

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

void MysubMatBuild_path(float *subMat, int *subMat_path,
    float *subGraph, int *subGraph_path, int *graph_id,
    int start, int sub_vertexs, int bdy_vertexs, int vertexs,
    int *st2ed, int *ed2st,
    int *adj_size, int *row_offset, int *col_val, float *weight) {
    // subMat build from csr Graph
    for (int i = 0; i < sub_vertexs; i++) {
        int ver = st2ed[i + start];
        int adjcount = adj_size[ver];
        int offset = row_offset[ver];
        for (int j = 0; j < adjcount; j++) {
            int neighbor = col_val[offset + j];
            if (graph_id[ver] != graph_id[neighbor])
                continue;
            float w = weight[offset + j];
            int index = ed2st[neighbor] - start;
            if (index >= 0 && index < sub_vertexs) {
                subMat[(int64_t)i * sub_vertexs + index] = w;
                subMat_path[(int64_t)i * sub_vertexs + index] = ver;
            }
        }
    }

    // Build from subGraph
    for (int i = 0; i < bdy_vertexs; i++) {
        for (int j = 0; j < sub_vertexs; j++) {
            int ver = st2ed[j + start];
            int64_t src = (int64_t)i * vertexs + ver;
            int64_t dst = (int64_t)i * sub_vertexs + j;
            subMat[dst] = subGraph[src];
            subMat_path[dst] = subGraph_path[src];
        }
    }

    // Diagonal numbers
    for (int i = 0; i < sub_vertexs; i++) {
        int64_t dst = (int64_t)i * sub_vertexs + i;
        subMat[dst] = 0;
        subMat_path[dst] = st2ed[i + start];
    }
}


__global__ void buildFromCSR(float* d_subMat, int* d_subMat_path, 
                            int* d_graph_id, int start, int sub_vertexs,
                            int vertexs, int* d_st2ed, int* d_ed2st,
                            int* d_adj_size, int* d_rowOffsetArc,
                            int* d_colValueArc, float* d_weightArc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sub_vertexs) return;
    
    int ver = d_st2ed[i + start];
    int adjcount = d_adj_size[ver];
    int offset = d_rowOffsetArc[ver];
    
    for (int j = 0; j < adjcount; j++) {
        int neighbor = d_colValueArc[offset + j];
        if (neighbor < 0 || neighbor >= vertexs) continue;
        if (d_graph_id[ver] != d_graph_id[neighbor]) continue;
        
        float w = d_weightArc[offset + j];
        int index = d_ed2st[neighbor] - start;
        
        if (index >= 0 && index < sub_vertexs) {
            int64_t mat_idx = (int64_t)i * sub_vertexs + index;
            d_subMat[mat_idx] = w;
            d_subMat_path[mat_idx] = ver;
        }
    }
}

__global__ void buildFromSubGraph(float* d_subMat, int* d_subMat_path,
                                 float* d_subGraph, int* d_subGraph_path,
                                 int start, int sub_vertexs, int bdy_vertexs,
                                 int vertexs, int* d_st2ed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bdy_vertexs) return;
    
    for (int j = 0; j < sub_vertexs; j++) {
        int ver = d_st2ed[j + start];
        int64_t src = (int64_t)i * vertexs + ver;
        int64_t dst = (int64_t)i * sub_vertexs + j;
                
        d_subMat[dst] = d_subGraph[src];
        d_subMat_path[dst] = d_subGraph_path[src];
    }
}

__global__ void setDiagonal(float* d_subMat, int* d_subMat_path,
                           int start, int sub_vertexs, int* d_st2ed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sub_vertexs) return;
    
    int64_t diag = (int64_t)i * sub_vertexs + i;
    d_subMat[diag] = 0;
    d_subMat_path[diag] = d_st2ed[i + start];
}


void LaunchMysubMatBuildKernel(float *d_subMat, int *d_subMat_path,
                              float *d_res, int *d_subGraph_path,
                              int *d_rowOffsetArc, int *d_colValueArc,
                              float *d_weightArc, int sub_vertexs, int bdy_vertexs,
                              int vertexs, int *d_graph_id, int *d_st2ed, 
                              int *d_ed2st, int *d_adj_size,
                              const std::vector<int> &graph_id,
                              int start, int num_edges) {
    // Initialize with MAXVALUE
    float* init = new float[sub_vertexs * sub_vertexs];
    for(int i = 0; i < sub_vertexs * sub_vertexs; i++) {
        init[i] = MAXVALUE;
    }
    cudaMemcpy(d_subMat, init, sub_vertexs * sub_vertexs * sizeof(float),
               cudaMemcpyHostToDevice);
    delete[] init;

    int* init_path = new int[sub_vertexs * sub_vertexs];
    memset(init_path, -1, sub_vertexs * sub_vertexs * sizeof(int));
    cudaMemcpy(d_subMat_path, init_path, sub_vertexs * sub_vertexs * sizeof(int),
               cudaMemcpyHostToDevice);
    delete[] init_path;


    const int blockSize = 256;

    // Step 1: Build from CSR Graph
    int numBlocks1 = (sub_vertexs + blockSize - 1) / blockSize;
    buildFromCSR<<<numBlocks1, blockSize>>>(d_subMat, d_subMat_path,
                                           d_graph_id, start, sub_vertexs,
                                           vertexs, d_st2ed, d_ed2st,
                                           d_adj_size, d_rowOffsetArc,
                                           d_colValueArc, d_weightArc);
    cudaDeviceSynchronize();

    // Step 2: Build from subGraph
    int numBlocks2 = (bdy_vertexs + blockSize - 1) / blockSize;
    buildFromSubGraph<<<numBlocks2, blockSize>>>(d_subMat, d_subMat_path,
                                                d_res, d_subGraph_path,
                                                start, sub_vertexs, bdy_vertexs,
                                                vertexs, d_st2ed);
    cudaDeviceSynchronize();

    // Step 3: Set diagonal elements
    int numBlocks3 = (sub_vertexs + blockSize - 1) / blockSize;
    setDiagonal<<<numBlocks3, blockSize>>>(d_subMat, d_subMat_path,
                                          start, sub_vertexs, d_st2ed);
    cudaDeviceSynchronize();
}

__global__ void MysubMatDecode_path_kernel(
    const float* d_subMat, const int* d_subMat_path,
    float* d_res, int* d_subGraph_path,
    const int* d_ed2st, int start, int row, int col, int vertexs) {
    
    // Calculate the row (i) and column (j) indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds checking
    if (i < row && j < col) {
        int ver = d_ed2st[j + start];
        int64_t dst = (int64_t)i * vertexs + ver;

        // Update subGraph and subGraph_path
        d_res[dst] = d_subMat[i * col + j];
        d_subGraph_path[dst] = d_subMat_path[i * col + j];
    }
}



void LaunchMysubMatDecodePathKernel(
    const float* d_subMat, const int* d_subMat_path,
    float* d_res, int* d_subGraph_path,
    const int* d_ed2st, int start, int row, int col, int vertexs) {
    
    // Define block and grid dimensions
    dim3 blockDim(16, 16); // Adjust as needed for your GPU architecture
    dim3 gridDim((row + blockDim.x - 1) / blockDim.x,
                 (col + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    MysubMatDecode_path_kernel<<<gridDim, blockDim>>>(
        d_subMat, d_subMat_path,
        d_res, d_subGraph_path,
        d_ed2st, start, row, col, vertexs);

    // Synchronize to ensure kernel completion
    checkCudaErrors(cudaDeviceSynchronize());
}


void MysubMatDecode_path(float *subMat, int *subMat_path,
    float *subGraph, int *subGraph_path,
    int start, int row, int col, int vertexs, int *st2ed) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int ver = st2ed[j + start];
            int64_t dst = (int64_t)i * vertexs + ver;
            subGraph[dst] = subMat[i * col + j];
            subGraph_path[dst] = subMat_path[i * col + j];
        }
    }
}


__global__ void MyMat1BuildKernel(float *d_mat1, const float* inner_to_inner_dist, 
                                 int inner_num, int bdy_num, int sub_vertexs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inner_num) return;
    
    int index = i + bdy_num;
    
    // Copy from inner_to_inner_dist to create the thin matrix
    for (int j = 0; j < bdy_num; j++) {
        float val = inner_to_inner_dist[index * sub_vertexs + j];
        d_mat1[i * bdy_num + j] = val;
    }
}

void LaunchMyMat1BuildKernel(float* d_mat1, const float* d_inner_to_inner_dist, 
                            int inner_num, int bdy_num, int sub_vertexs) {
    // Print source data
    const int blockSize = 256;
    const int numBlocks = (inner_num + blockSize - 1) / blockSize;
    
    MyMat1BuildKernel<<<numBlocks, blockSize>>>(
        d_mat1, d_inner_to_inner_dist, inner_num, bdy_num, sub_vertexs);

    cudaDeviceSynchronize();
    
    // Verify output
}

void MyMat1Build(float *mat1, float *subMat,
    int inner_num, int bdy_num, int sub_vertexs) {
    for (int i = 0; i < inner_num; i++) {
        int index = i + bdy_num;
        memcpy(mat1 + i * bdy_num, subMat + index * sub_vertexs,
            bdy_num * sizeof(float));
    }
}

}  // namespace fap

#endif