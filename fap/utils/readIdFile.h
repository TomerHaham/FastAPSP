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
#include <chrono>
#include <time.h>
#include <sys/time.h>


#include "fap/utils/debug.h"
//#include "/home/hd/hd_hd/hd_nh271/bachelor_project/Jet-Partitioner/jet.hpp"
#include "/home/thaham/Jet-Partitioner/header/jet.h"
#include "/home/thaham/Jet-Partitioner/header/experiment_data.hpp"

using std::ifstream;
using std::string;
using std::to_string;
using std::unordered_map;
using std::vector;

//#define DEBUG

namespace fap {
using edge_offset_t = int32_t;  // The type for edge offsets
using ordinal_t = int32_t;      // The type for vertex indices
using value_t = int32_t;        // The type for edge weights (e.g., for unweighted graphs set to 1)

// Define Kokkos Views to hold graph data
using edge_view_t = Kokkos::View<edge_offset_t*, Kokkos::Cuda>;
using edge_mirror_t = typename edge_view_t::HostMirror;
using vtx_view_t = Kokkos::View<ordinal_t*, Kokkos::Cuda>;
using vtx_mirror_t = typename vtx_view_t::HostMirror;
using wgt_view_t = Kokkos::View<value_t*, Kokkos::Cuda>;
using wgt_mirror_t = typename wgt_view_t::HostMirror;

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
    int *adj_size, int *row_offset, int *col_val, std::string partitioner, double &time_partitioning)
{
    auto start_time_partitioning = std::chrono::high_resolution_clock::now();
if (partitioner == "jet") {
    using matrix_t = jet_partitioner::matrix_t;
    using edge_view_t = jet_partitioner::edge_vt;
    using edge_mirror_t = jet_partitioner::edge_mt;
    using wgt_view_t = jet_partitioner::wgt_vt;
    using wgt_mirror_t = jet_partitioner::wgt_mt;
    using Device = Kokkos::Cuda;

    edge_view_t row_map("row_map", vertexs + 1);
    edge_view_t entries("entries", edges);
    wgt_view_t values("values", edges);

    edge_mirror_t h_row_map = Kokkos::create_mirror_view(row_map);
    edge_mirror_t h_entries = Kokkos::create_mirror_view(entries);
    wgt_mirror_t h_values = Kokkos::create_mirror_view(values);

    for (int i = 0; i <= vertexs; ++i) {
        h_row_map(i) = row_offset[i];
    }
    for (int i = 0; i < edges; ++i) {
        h_entries(i) = col_val[i];
        h_values(i) = 1;
    }

    Kokkos::deep_copy(row_map, h_row_map);
    Kokkos::deep_copy(entries, h_entries);
    Kokkos::deep_copy(values, h_values);

    matrix_t graph("input_graph", vertexs, vertexs, edges, values, row_map, entries);

    wgt_view_t vertex_weights("vertex_weights", vertexs);
    wgt_mirror_t h_vertex_weights = Kokkos::create_mirror_view(vertex_weights);
    Kokkos::deep_copy(h_vertex_weights, 1);
    Kokkos::deep_copy(vertex_weights, h_vertex_weights);

    jet_partitioner::config_t config;
    config.coarsening_alg = 0;
    config.num_parts = K;
    config.max_imb_ratio = 1.03;
    
    jet_partitioner::experiment_data<value_t> experiment;
    value_t edge_cut = 0;
    
    auto part_view = jet_partitioner::partition(
        edge_cut,
        config,
        graph,
        vertex_weights,
        true,
        experiment
    );

    auto h_part_view = Kokkos::create_mirror_view(part_view);
    Kokkos::deep_copy(h_part_view, part_view);

    for (int i = 0; i < vertexs; i++) {
        int subGraph_id = h_part_view(i) + 1;
        id[i] = subGraph_id;
        BlockVer[subGraph_id].push_back(i);
    }
}
 // work on host
 /*   
if (partitioner == "jet") {
    using matrix_t = jet_partitioner::serial_matrix_t;
    using wgt_view_t = jet_partitioner::wgt_serial_vt;
    using ordinal_t = jet_partitioner::ordinal_t;
    using Device = Kokkos::Serial;
    
    Kokkos::View<ordinal_t*, Device> row_map("row_map", vertexs + 1);
    Kokkos::View<ordinal_t*, Device> entries("entries", edges);
    wgt_view_t values("values", edges);

    for (int i = 0; i <= vertexs; ++i) {
        row_map(i) = row_offset[i];
    }
    for (int i = 0; i < edges; ++i) {
        entries(i) = col_val[i];
        values(i) = 1;
    }

    matrix_t graph("input_graph", vertexs, vertexs, edges, values, row_map, entries);

    wgt_view_t vertex_weights("vertex_weights", vertexs);
    Kokkos::deep_copy(vertex_weights, 1);

    jet_partitioner::config_t config;
    config.coarsening_alg = 0;
    config.num_parts = K;
    config.max_imb_ratio = 1.03;
    
    jet_partitioner::experiment_data<value_t> experiment;
    value_t edge_cut = 0;
    
    auto part_view = jet_partitioner::partition_serial(
        edge_cut,
        config,
        graph,
        vertex_weights,
        true,
        experiment
    );

    for (int i = 0; i < vertexs; i++) {
        int subGraph_id = part_view(i) + 1;
        id[i] = subGraph_id;
        BlockVer[subGraph_id].push_back(i);
    }
}
*/

/*    if (partitioner == "jet") {
	using Device = Kokkos::Cuda;

        using matrix_t = KokkosSparse::CrsMatrix<int, int, Device, void, edge_offset_t>;
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

        jet_partitioner::experiment_data<int> experiment;

        jet_partitioner::config_t config;
config.coarsening_alg = 0;  // Set coarsening algorithm
config.num_parts = K;       // Set the number of parts (partitions)
config.max_imb_ratio = 1.03; // Set imbalance ratio, etc.
config.num_iter = 1;
config.refine_tolerance = 0.999;
config.dump_coarse = false;
config.ultra_settings = false;
value_t edge_cut = 0;  // or float depending on your weight type

        // Perform the partition
        //auto part_view = jet_partitioner::partitioner<matrix_t, int>::partition(graph, vertex_weights, K, 1.03, false, experiment);
       auto part_view = jet_partitioner::partition(edge_cut, config, graph, vertex_weights, false, experiment);

	// After partitioning
	//experiment.verboseReport();

        // Copy results back to host

//	 auto part_host = Kokkos::create_mirror_view(part_view);
  //      Kokkos::deep_copy(part_host, part_view);

	vtx_view_t part_host("part_host", part_view.size());
	Kokkos::deep_copy(part_host, part_view);

        // Assign partition IDs to `id` and fill `BlockVer`
        for (unsigned i = 0; i < part_host.size(); i++) {
            int subGraph_id = part_host(i) + 1; // Convert to 1-based indexing for output
            id[i] = subGraph_id;
            BlockVer[subGraph_id].push_back(i);
        }
    }*/
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
        auto end_time_partitioning = std::chrono::high_resolution_clock::now();
        time_partitioning = std::chrono::duration<double>(end_time_partitioning - start_time_partitioning).count();
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
        int *row_offset, int *col_val, float *weight, std::string partitioner, double &time_partitioning) {
#ifdef DEBUG
    readIdFile(vertexs, id, BlockVer, K, directed, true);
#else
    Graph_decomposition(id, BlockVer, directed, K,
        vertexs, edges, adj_size, row_offset, col_val, partitioner, time_partitioning);
#endif
}

}  // namespace fap

#endif
