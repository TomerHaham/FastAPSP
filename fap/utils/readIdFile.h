#ifndef UTIL_READIDFILE_H_
#define UTIL_READIDFILE_H_

#pragma once

#include <metis.h>
#include <kaHIP_interface.h>
// Include Kokkos headers
#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>  // For KokkosSparse::CrsMatrix

#include <unordered_map>
#include <map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "fap/utils/debug.h"
#include "/home/thaham/bachelor_project/Jet-Partitioner/jet.hpp"
using std::ifstream;
using std::string;
using std::to_string;
using std::unordered_map;
using std::vector;

//#define DEBUG

namespace fap {

void readIdFile(int n, int *id,
    std::unordered_map<int, std::vector<int>> &BlockVer,
    int K, bool directed, bool weighted) {
    std::ifstream input;
#ifdef DEBUG
    input.open("../test/subtest.txt");

    int a, b;
    string line;
    while (input >> line) {
        int pre = 0, cnt = 0;
        for (int i = 0; i < line.size(); i++) {
            if (cnt == 0 && line[i] == ',') {
                a = stoi(line.substr(pre, i - pre));
                cnt++;
                pre = i + 1;
            }
            if (cnt == 1 && line[i] == ',') {
                b = stoi(line.substr(pre));
                break;
            }
        }
        id[a] = b;
        BlockVer[b].push_back(a);
    }
#endif
}

// if the input csr is directed graph, need to change it to undirected graph
void Graph_decomposition(int *id,
    std::unordered_map<int, std::vector<int>> &BlockVer,
    bool directed, int K, int vertexs, int edges,
    int *adj_size, int *row_offset, int *col_val)
{
    using matrix_t = KokkosSparse::CrsMatrix<value_t, ordinal_t, Device, void, edge_offset_t>;
    using graph_t = typename matrix_t::staticcrsgraph_type;
/*
    // Debug: Print input parameters
    std::cout << "Input Parameters:" << std::endl;
    std::cout << "Number of vertices: " << vertexs << std::endl;
    std::cout << "Number of edges: " << edges << std::endl;
    std::cout << "Number of partitions: " << K << std::endl;
    std::cout << "Directed: " << (directed ? "true" : "false") << std::endl;
    std::cout << "Adjacency sizes (adj_size): ";
    for (int i = 0; i < vertexs; ++i) {
        std::cout << adj_size[i] << " ";
    }

    std::cout << std::endl;
    std::cout << "Row offsets (row_offset): ";
    for (int i = 0; i <= vertexs; ++i) {
        std::cout << row_offset[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Column values (col_val): ";
    for (int i = 0; i < edges; ++i) {
        std::cout << col_val[i] << " ";
    }
    std::cout << std::endl;
*/
/*    // Convert input data to 0-based indexing for column indices
    for (int i = 0; i < edges; ++i) {
        col_val[i]--; // Convert column indices to 0-based
    }
*/
    // Allocate and populate Kokkos Views for CSR representation
    edge_view_t row_map(Kokkos::ViewAllocateWithoutInitializing("row_map"), vertexs + 1);
    edge_mirror_t row_map_m = Kokkos::create_mirror_view(row_map);
    vtx_view_t entries(Kokkos::ViewAllocateWithoutInitializing("entries"), edges);
    vtx_mirror_t entries_m = Kokkos::create_mirror_view(entries);
    wgt_view_t values(Kokkos::ViewAllocateWithoutInitializing("values"), edges);
    wgt_mirror_t values_m = Kokkos::create_mirror_view(values);

    // Fill row_map and entries from input data
    for (int i = 0; i <= vertexs; ++i) {
        row_map_m(i) = row_offset[i];
    }
    for (int i = 0; i < edges; ++i) {
        entries_m(i) = col_val[i];
    }

    // Assuming unweighted graph, set all edge weights to 1
    Kokkos::deep_copy(values_m, 1);

    // Copy data to device views
    Kokkos::deep_copy(row_map, row_map_m);
    Kokkos::deep_copy(entries, entries_m);
    Kokkos::deep_copy(values, values_m);

/*    // Debug: Print CSR format
    std::cout << "Number of vertices: " << vertexs << std::endl;
    std::cout << "Number of edges: " << edges << std::endl;
    std::cout << "Row offsets (xadj): ";
    for (int i = 0; i <= vertexs; ++i) {
        std::cout << row_map_m(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Column indices (adjncy): ";
    for (int i = 0; i < edges; ++i) {
        std::cout << entries_m(i) << " ";
    }
    std::cout << std::endl;
*/
    // Create the graph object
    graph_t g_graph(entries, row_map);
    matrix_t graph("input graph", vertexs, values, g_graph);

    // Jet partitioning setup
    Kokkos::View<int*, Device> vertex_weights("vertex_weights", vertexs);
    jet_partitioner::ExperimentLoggerUtil<int> experiment;

    // Print information before partitioning
  //  std::cout << "Starting partitioning with " << K << " parts..." << std::endl;

    // Perform the partition
    auto part_view = jet_partitioner::partitioner<matrix_t, int>::partition(graph, vertex_weights, K, 1.001, false, experiment);

    //std::cout << "Partitioning finished." << std::endl;

    // Copy results back to host
    auto part_host = Kokkos::create_mirror_view(part_view);
    Kokkos::deep_copy(part_host, part_view);

    // Assign partition IDs to `id` and fill `BlockVer`
    for (unsigned i = 0; i < part_host.size(); i++) {
        int subGraph_id = part_host(i) + 1; // Convert to 1-based indexing for output
        id[i] = subGraph_id;
        BlockVer[subGraph_id].push_back(i);
    }
/*
    // Debug: Print partition results
    std::cout << "Partition results:" << std::endl;
    for (unsigned i = 0; i < part_host.size(); i++) {
        std::cout << "Vertex " << i << " -> Partition " << part_host(i)  << std::endl;
    }*/
}

void readIdFile_METIS(
        int *id, std::unordered_map<int, std::vector<int>> &BlockVer,
        int K, bool directed, int vertexs,
        int edges, int *adj_size,
        int *row_offset, int *col_val, float *weight) {
#ifdef DEBUG
    readIdFile(vertexs, id, BlockVer, K, directed, true);
#else
    Graph_decomposition(id, BlockVer, directed, K,
        vertexs, edges, adj_size, row_offset, col_val);
#endif
}

}  // namespace fap

#endif
