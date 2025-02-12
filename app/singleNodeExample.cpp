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

#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "fap/fap.h"
#include <chrono>
#include <Kokkos_Core.hpp>  // Include Kokkos after MPI



int main(int argc, char **argv)
{
        Kokkos::initialize();    // Initialize Kokkos

    {
        // Start timing for the complete runtime
        auto start_time = std::chrono::high_resolution_clock::now();

        std::string file;
        int K;
        std::string partitioner;
        bool directed = false, weighted = false, version = false;

        // Parse command line arguments
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-f") == 0) {
                file = argv[i + 1];
            } else if (strcmp(argv[i], "-k") == 0) {
                K = std::stoi(argv[i + 1]);
            } else if (strcmp(argv[i], "-direct") == 0) {
                directed = (strcmp(argv[i + 1], "true") == 0);
            } else if (strcmp(argv[i], "-weight") == 0) {
                weighted = (strcmp(argv[i + 1], "true") == 0);
            } else if (strcmp(argv[i], "-partitioner") == 0) {
                partitioner = argv[i + 1];
            }else if (strcmp(argv[i], "-version") == 0) {
                version = (strcmp(argv[i + 1], "true") == 0);
            }
        }
// run the kernel
        std::chrono::time_point<std::chrono::high_resolution_clock> endTimeIo;       
    fap::fapGraph G(file, directed, weighted, K, partitioner, version, endTimeIo);
    G.preCondition();
if (G.isSplit() && K >= 3) {
    printf("Processing first three subgraphs in split mode\n");
    G.solveSubGraph(1, true);
    printf("Subgraph 1 processing complete\n");
    G.solveSubGraph(2, true);
    printf("Subgraph 2 processing complete\n");
    G.solveSubGraph(3, true);
    printf("Subgraph 3 processing complete\n");
} else {
    printf("Processing all subgraphs in non-split mode\n");
    G.solve();
    printf("All subgraphs processing complete\n");
}            auto end_time = std::chrono::high_resolution_clock::now();
        double total_runtime = std::chrono::duration<double>(end_time - start_time).count();
            std::cout << "Total runtime: " << total_runtime << " seconds." << std::endl;

    // get the result
    auto current_subgraph_id = G.getCurrentSubGraphId();
    auto subgraph_dist = G.getSubGraphDistance(current_subgraph_id);
    auto subgraph_path = G.getSubGraphPath(current_subgraph_id);
    auto source = G.getSubGraphIndex(current_subgraph_id);

    // verify
    bool check = G.check_result(
        subgraph_dist.data(), subgraph_path.data(),
        source.data(), source.size(), G.adj_size.size(),
        G.adj_size.data(), G.row_offset.data(),
        G.col_val.data(), G.weight.data(),
        G.getGraphId().data());

    int32_t verify_id = current_subgraph_id;
    if (check == false)
        printf("the %d subGraph is wrong !!!\n", verify_id);
    else
        printf("the %d subGraph is right\n", verify_id);       
}
    Kokkos::finalize();  // Finalize Kokkos
        return 0;


}