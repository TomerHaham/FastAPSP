#include <chrono>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <mpi.h>  // Explicitly include MPI before Kokkos

#include <iostream>
#include <fstream>
#include <sstream>

#include "fap/fap.h"
#include "fap/fap_more_task_mpi.h"
#include <Kokkos_Core.hpp>  // Include Kokkos after MPI

#include "/home/thaham/bachelor_project/FlatBufferWriter.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);  // Initialize MPI
    Kokkos::initialize();    // Initialize Kokkos

    {
        // Start timing for the complete runtime
        auto start_time = std::chrono::high_resolution_clock::now();

        // FlatBufferWriter instance to collect metrics
        FlatBufferWriter writer;
        PartitionConfig config;

        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        int num_process = world_size;

        int myProcess;
        MPI_Comm_rank(MPI_COMM_WORLD, &myProcess);

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

        // Initialize graph and partition
        fap::fapGraphMoreTaskMPI G(file, directed, weighted, K, partitioner, version);
        G.init(num_process, K);

        if (myProcess == 0) {
            G.preCondition();
            G.loadBalance(num_process);
        }

        // Broadcast the meta data.
        G.Meta_MPI_Bcast(MPI_COMM_WORLD);

        // Run the kernel
        auto current_task_array = G.getCurrentTask(myProcess);
        for (int i = 0; i < current_task_array.size(); i++) {
//	    std::cout <<  current_task_array.size() << std::endl;
            int32_t current_task_id = current_task_array[i];
            G.solveSubGraph(current_task_id, true);
//	    std::cout << "finish" << std::endl;
        }
/*
// get the result of last task
    auto current_subgraph_id = G.getCurrentSubGraphId();
    auto subgraph_dist = G.getSubGraphDistance(current_subgraph_id);
    auto subgraph_path = G.getSubGraphPath(current_subgraph_id);
    auto source = G.getSubGraphIndex(current_subgraph_id);

    // verify the last task
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
//	std::cout << "finish" << std::endl;
*/
        // Synchronize Kokkos tasks
        Kokkos::fence(); // Ensure all Kokkos tasks are completed

        // Synchronize CUDA (if applicable)
        cudaDeviceSynchronize(); // Ensure all CUDA kernels are completed

        // Synchronize MPI processes
        MPI_Barrier(MPI_COMM_WORLD); // Wait for all processes to finish

        // Collect total runtime at the end
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_runtime = std::chrono::duration<double>(end_time - start_time).count();

        // Write runtime to FlatBufferWriter
        if (myProcess == 0) {
            // Create variables for the arguments to avoid reference issues
            double buffer_io_time = 0.0;
            double mapping_time = 0.0;
            long maxRSS = 0;
            std::cout << "Total runtime: " << total_runtime << " seconds." << std::endl;
            // Update FlatBufferWriter with runtime metrics
            writer.updateResourceConsumption(buffer_io_time, mapping_time, total_runtime, maxRSS);

            // Also update with partition configuration details
            uint64_t total_edge_cut = 0;  // Example placeholder
            double balance = 0.0;         // Example placeholder
            writer.updatePartitionMetrics(total_edge_cut, balance);

            // Update config with partitioning parameters
            config.k = K;
            config.partitioner = partitioner; // Add partitioner information
            config.output_path = "./"; // Set output path, adjust as needed
            config.write_results = true;

            // Set a unique experiment ID using current timestamp
            auto now = std::chrono::system_clock::now();
            auto now_time_t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << now_time_t;
            config.experiment_id = ss.str();  // Set a unique experiment ID

            // Write the final data to the output file
            std::string baseFilename = FlatBufferWriter::extractBaseFilename(file);
            writer.write(baseFilename, config);
            
        }
    }

    Kokkos::finalize();  // Finalize Kokkos
    MPI_Finalize();      // Finalize MPI

    return 0;
}