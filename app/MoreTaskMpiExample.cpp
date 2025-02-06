
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
if (myProcess == 0) {
    std::cout << "Total MPI processes: " << world_size << std::endl;
}

        std::string file;
        int K = 0; // Initialize to avoid uninitialized variable usage
        std::string partitioner;
        bool directed = false, weighted = false, version = false;
        auto start_time_io = std::chrono::high_resolution_clock::now();

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
            } else if (strcmp(argv[i], "-version") == 0) {
                version = (strcmp(argv[i + 1], "true") == 0);
            }
        }

        // Check for required inputs
        if (file.empty() || K <= 0) {
            if (myProcess == 0) {
                std::cerr << "Error: Missing required inputs. Ensure -f <file> and -k <partitions> are provided.\n";
            }
            MPI_Finalize();
            return -1;
        }

        std::chrono::time_point<std::chrono::high_resolution_clock> end_io_time;       
        // Initialize graph and partition
        fap::fapGraphMoreTaskMPI G(file, directed, weighted, K, partitioner, version, end_io_time);
        G.init(num_process, K);

        if (myProcess == 0) {
            G.preCondition();
            G.loadBalance(num_process);
        }

        // Broadcast the meta data
        G.Meta_MPI_Bcast(MPI_COMM_WORLD);

        // Run the kernel
        auto current_task_array = G.getCurrentTask(myProcess);
        if (current_task_array.empty()) {
            printf("Process %d: No tasks assigned.\n", myProcess);
        } else {
            for (int i = 0; i < current_task_array.size(); i++) {
                int32_t current_task_id = current_task_array[i];
                //printf("Process %d: Solving task %d\n", myProcess, current_task_id);
                G.solveSubGraph(current_task_id, true);
                //printf("Process %d: Finished solving task %d\n", myProcess, current_task_id);
            }
        }
    
        

        // Synchronize Kokkos tasks
        Kokkos::fence(); // Ensure all Kokkos tasks are completed

        // Synchronize CUDA (if applicable)
        cudaDeviceSynchronize(); // Ensure all CUDA kernels are completed

        // Synchronize MPI processes
        MPI_Barrier(MPI_COMM_WORLD); // Wait for all processes to finish

        // Collect total runtime at the end
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_runtime = std::chrono::duration<double>(end_time - start_time).count();
        double total_io_time = std::chrono::duration<double>(end_io_time - start_time_io).count();

       /* // Get the result of the last task
        auto current_subgraph_id = G.getCurrentSubGraphId();
        if (current_subgraph_id < 0 || current_subgraph_id >= K) {
            printf("Process %d: Invalid subgraph ID %d\n", myProcess, current_subgraph_id);
        } else {
            auto subgraph_dist = G.getSubGraphDistance(current_subgraph_id);
            auto subgraph_path = G.getSubGraphPath(current_subgraph_id);
            auto source = G.getSubGraphIndex(current_subgraph_id);

            if (subgraph_dist.empty() || subgraph_path.empty() || source.empty()) {
                printf("Process %d: Invalid data for subgraph %d\n", myProcess, current_subgraph_id);
            } else {
                // Verify the last task
                bool check = G.check_result(
                    subgraph_dist.data(), subgraph_path.data(),
                    source.data(), source.size(), G.adj_size.size(),
                    G.adj_size.data(), G.row_offset.data(),
                    G.col_val.data(), G.weight.data(),
                    G.getGraphId().data());

                if (check) {
                    printf("Process %d: Subgraph %d is correct.\n", myProcess, current_subgraph_id);
                } else {
                    printf("Process %d: Subgraph %d is incorrect!\n", myProcess, current_subgraph_id);

                }
            }
        }*/
        // Write runtime to FlatBufferWriter
        if (myProcess == 0) {
            double mapping_time = 0.0;
            long maxRSS = 0;

            std::cout << "Total runtime: " << total_runtime << " seconds.\n";
            std::cout << "Total runtime: " << total_io_time << " seconds.\n";
            std::cout << "paritition runtime: " << G.getTimePartitioning() << " seconds.\n";
            std::cout << "sssp runtime: " << G.getTimeSSSP() << " seconds.\n";
            std::cout << "floyd runtime: " << G.getTimeFloyd() << " seconds.\n";
            std::cout << "min plus runtime: " << G.getTimeMinPlus() << " seconds.\n";
            std::cout << "data runtime: " << G.getTimeData() << " seconds.\n";
            std::cout << "Gpu memory consumption sssp: " << G.getMemConsSSSP() << " GB.\n";
            std::cout << "Gpu memory consumption floyd: " << G.getMemConsFloyd() << " GB.\n";
            std::cout << "Gpu memory consumption min-plus: " << G.getMemConsMinPlus() << "GB\n";
            std::cout << "boundary vertices " << G.getBoundaryNum() << "\n";
            std::cout << "inner vertices " << G.getInnerNum() << "\n";
            std::cout << "edge cut " << G.getEdgeCut() << "\n";
            // Update basic runtime and resource consumption values.
            writer.updateResourceConsumption(total_io_time, mapping_time, total_runtime, maxRSS);

            uint64_t total_edge_cut = G.getEdgeCut();  
            double balance = 0.0;         // Example placeholder
            writer.updatePartitionMetrics(total_edge_cut, balance);

            // Update detailed metrics with timing, memory and vertex counts.
    writer.updateDetailedMetrics(
        G.getTimePartitioning(),  // time_partitioning
        G.getTimeSSSP(),          // time_sssp
        G.getTimeFloyd(),         // time_floyd
        G.getTimeMinPlus(),       // time_min_plus
        G.getTimeData(),          // time_data_transformation
        G.getMemConsSSSP(),       // mem_cons_sssp
        G.getMemConsFloyd(),      // mem_cons_floyd
        G.getMemConsMinPlus(),    // mem_cons_min_plus
        G.getBoundaryNum(),      // boundary_vertices
        G.getInnerNum()          // inner_vertices
    );

            config.k = K;
            config.partitioner = partitioner;
            config.output_path = "./";
            config.write_results = true;
            config.version = version;

            auto now = std::chrono::system_clock::now();
            auto now_time_t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << now_time_t;
            config.experiment_id = ss.str();

            std::string baseFilename = FlatBufferWriter::extractBaseFilename(file);
            writer.write(baseFilename, config);
        }
    }

    Kokkos::finalize();  // Finalize Kokkos
    MPI_Finalize();      // Finalize MPI

    return 0;
}
