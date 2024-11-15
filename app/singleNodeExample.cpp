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

int main(int argc, char **argv)
{
    std::string file;
    std::string partitioner;
    int K;
    bool directed, weighted;
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
        }
    }

    // run the kernel
    fap::fapGraph G(file, directed, weighted, K, partitioner);
    G.preCondition();
    if (G.isSplit() && K >= 3) {
        G.solveSubGraph(1, true);
        G.solveSubGraph(2, true);
        G.solveSubGraph(3, true);
    } else {
        G.solve();
    }

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
