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
    int *adj_size, int *row_offset, int *col_val, std::string partitioner)
{
    if (partitioner == "jet") {
        using matrix_t = KokkosSparse::CrsMatrix<value_t, ordinal_t, Device, void, edge_offset_t>;
        using graph_t = typename matrix_t::staticcrsgraph_type;

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

        // Create the graph object
        graph_t g_graph(entries, row_map);
        matrix_t graph("input graph", vertexs, values, g_graph);

        // Jet partitioning setup
        Kokkos::View<int*, Device> vertex_weights("vertex_weights", vertexs);
	// added 28.11
	Kokkos::deep_copy(vertex_weights, 1);

        jet_partitioner::ExperimentLoggerUtil<int> experiment;

        // Perform the partition
        auto part_view = jet_partitioner::partitioner<matrix_t, int>::partition(graph, vertex_weights, K, 1.03, false, experiment);

	// After partitioning
	//experiment.verboseReport();

        // Copy results back to host
        auto part_host = Kokkos::create_mirror_view(part_view);
        Kokkos::deep_copy(part_host, part_view);

        // Assign partition IDs to `id` and fill `BlockVer`
        for (unsigned i = 0; i < part_host.size(); i++) {
            int subGraph_id = part_host(i) + 1; // Convert to 1-based indexing for output
            id[i] = subGraph_id;
            BlockVer[subGraph_id].push_back(i);
        }
    } 
    else if (partitioner == "KaHIP" || partitioner == "KaHIP_eco" || partitioner == "KaHIP_soc_eco" || partitioner == "KaHIP_soc_s" || partitioner == "KaHIP_soc_f" ||partitioner == "KaHIP_f" || partitioner == "metis") {
        std::vector<idx_t> xadj(1);
        std::vector<idx_t> adjncy;

        if (directed) {
            // Convert to undirected graph
            std::map<int, std::unordered_set<int>> graph;
            for (int i = 0; i < vertexs; i++) {
                int ver = i;
                int adjcount = adj_size[ver];
                int offset = row_offset[ver];
                for (int j = 0; j < adjcount; j++) {
                    int nextNode = col_val[offset + j];
                    if (ver == nextNode) {
                        continue;
                    }
                    graph[ver].insert(nextNode);
                    graph[nextNode].insert(ver);
                }
            }

            for (auto it = graph.begin(); it != graph.end(); it++) {
                auto ver = it->first;
                int count = 0;
                for (auto neighbor = graph[ver].begin(); neighbor != graph[ver].end(); neighbor++) {
                    int nextNode = *neighbor;
                    count++;
                    adjncy.push_back(nextNode);
                }
                xadj.push_back(xadj.back() + count);
            }
        } else {
            std::vector<idx_t> xadj_tmp(row_offset, row_offset + vertexs + 1);
            std::vector<idx_t> adjncy_tmp(col_val, col_val + edges);
            xadj.assign(xadj_tmp.begin(), xadj_tmp.end());
            adjncy.assign(adjncy_tmp.begin(), adjncy_tmp.end());
        }

        idx_t nVertices = vertexs;
        idx_t nWeights = 1;
        idx_t nParts = K;
        idx_t objval;
        std::vector<idx_t> part(nVertices, 0);

        if (partitioner == "KaHIP" || partitioner == "KaHIP_soc_s" || partitioner == "KaHIP_soc_f" || partitioner == "KaHIP_f" || partitioner == "KaHIP_eco" || partitioner == "KaHIP_soc_eco") {
            double imbalance = 0.003;  // 3% imbalance
            bool suppress_output = true;  // Suppress output to stdout
            int seed = 42;  // Random seed
            int mode = 2;  // Mode (e.g., FAST mode in KaHIP)
            int edgecut = 0;
	    if(partitioner == "KaHIP_f"){
	    	mode = 0;
	    }else if(partitioner == "KaHIP_soc_f"){
		mode = 3;
	    }else if(partitioner == "KaHIP_soc_s"){
		mode = 5;
            }else if(partitioner == "KaHIP_soc_eco"){
                mode = 4;
            }else if(partitioner == "KaHIP_eco"){
                mode = 2;
            }
            kaffpa(&nVertices, nullptr, xadj.data(), nullptr, adjncy.data(), &nParts,
                   &imbalance, suppress_output, seed, mode, &edgecut, part.data());
        } 
        else if (partitioner == "metis") {
            int ret = METIS_PartGraphKway(&nVertices, &nWeights,
                                          xadj.data(), adjncy.data(),
                                          nullptr, nullptr, nullptr, &nParts,
                                          nullptr, nullptr, nullptr, &objval, part.data());

            if (ret != METIS_OK) {
                std::cerr << "METIS_ERROR" << std::endl;
                return;
            }
        }

        // Assign partition IDs to `id` and fill `BlockVer`
        for (unsigned i = 0; i < part.size(); i++) {
            int subGraph_id = part[i] + 1; // Convert to 1-based indexing
            id[i] = subGraph_id;
            BlockVer[subGraph_id].push_back(i);
        }
    } 
    else {
        std::cerr << "Unknown partitioner type: " << partitioner << std::endl;
        throw std::invalid_argument("Invalid partitioner type");
    }
}

void readIdFile_METIS(
        int *id, std::unordered_map<int, std::vector<int>> &BlockVer,
        int K, bool directed, int vertexs,
        int edges, int *adj_size,
        int *row_offset, int *col_val, float *weight, std::string partitioner) {
#ifdef DEBUG
    readIdFile(vertexs, id, BlockVer, K, directed, true);
#else
    Graph_decomposition(id, BlockVer, directed, K,
        vertexs, edges, adj_size, row_offset, col_val, partitioner);
#endif
}

}  // namespace fap

#endif
