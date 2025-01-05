/*#include <chrono>
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

int main(int argc, char **argv) {
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
            } else if (strcmp(argv[i], "-version") == 0) {
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
        std::set<int32_t> processed_tasks;

        // Print initial task assignment
        if (myProcess == 0) {
            printf("Task distribution:\n");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        printf("Process %d assigned tasks:", myProcess);
        for (const auto& task : current_task_array) {
            printf(" %d", task);
        }
        printf("\n");
        fflush(stdout);

        MPI_Barrier(MPI_COMM_WORLD);

        // Create arrays for task ownership
        std::vector<int> task_owner(K + 1, -1);  // +1 since tasks are 1-based
        for (const auto& task : current_task_array) {
            task_owner[task] = myProcess;
        }

        // Synchronize task ownership across all processes
        for (int i = 1; i <= K; i++) {
            int my_ownership = (task_owner[i] == myProcess) ? myProcess : -1;
            int owner;
            MPI_Allreduce(&my_ownership, &owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            task_owner[i] = owner;
        }

        // Process only owned tasks
// Process only owned tasks
for (const auto& current_task_id : current_task_array) {
    // Verify ownership
    if (task_owner[current_task_id] != myProcess) {
        printf("Process %d: Task %d ownership verification failed\n", myProcess, current_task_id);
        continue;
    }

    // Execute the task
    printf("Process %d executing task %d\n", myProcess, current_task_id);
    fflush(stdout);
    
    G.solveSubGraph(current_task_id, true);
    
    // Mark as completed
    processed_tasks.insert(current_task_id);

    // Verify the task results right after execution
    auto current_subgraph_id = G.getCurrentSubGraphId();
    printf("Process %d: Task %d, Subgraph ID: %d (K = %d)\n", 
           myProcess, current_task_id, current_subgraph_id, K);


    // Synchronize after task completion and verification
    MPI_Barrier(MPI_COMM_WORLD);
}

        // Final verification stats
        int my_verified = processed_tasks.size();
        int total_verified;
        MPI_Reduce(&my_verified, &total_verified, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (myProcess == 0) {
            printf("Total tasks completed and verified: %d/%d\n", total_verified, K);
        }

        // Final synchronization
        MPI_Barrier(MPI_COMM_WORLD);

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
            double buffer_io_time = 0.0;
            double mapping_time = 0.0;
            long maxRSS = 0;
            std::cout << "Total runtime: " << total_runtime << " seconds." << std::endl;
            
            writer.updateResourceConsumption(buffer_io_time, mapping_time, total_runtime, maxRSS);

            uint64_t total_edge_cut = 0;
            double balance = 0.0;
            writer.updatePartitionMetrics(total_edge_cut, balance);

            config.k = K;
            config.partitioner = partitioner;
            config.output_path = "./";
            config.write_results = true;

            auto now = std::chrono::system_clock::now();
            auto now_time_t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << now_time_t;
            config.experiment_id = ss.str();

            std::string baseFilename = FlatBufferWriter::extractBaseFilename(file);
            writer.write(baseFilename, config);
        }
    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}*/

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
        int K = 0; // Initialize to avoid uninitialized variable usage
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

        // Initialize graph and partition
        fap::fapGraphMoreTaskMPI G(file, directed, weighted, K, partitioner, version);
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
            double buffer_io_time = 0.0;
            double mapping_time = 0.0;
            long maxRSS = 0;
            std::cout << "Total runtime: " << total_runtime << " seconds.\n";
            writer.updateResourceConsumption(buffer_io_time, mapping_time, total_runtime, maxRSS);

            uint64_t total_edge_cut = 0;  // Example placeholder
            double balance = 0.0;         // Example placeholder
            writer.updatePartitionMetrics(total_edge_cut, balance);

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