cmake_minimum_required(VERSION 3.20)

# Set the project name and version
project(fapApp VERSION 0.1 LANGUAGES CXX CUDA)

# Specify the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

option(WITH_CUDA "Enable the CUDA backend for supported interfaces" ON)

# Set paths for KaHIP, Metis, Jet Partitioner (use Spack paths or set manually)
set(KAHIP_DIR "/home/thaham/bachelor_project/spack/opt/spack/linux-ubuntu22.04-sapphirerapids/gcc-11.4.0/kahip-3.14-drjnai36m4zbt7k2y46yn5xiz5toknvs")
set(METIS_DIR "/home/thaham/bachelor_project/spack/opt/spack/linux-ubuntu22.04-sapphirerapids/gcc-11.4.0/metis-5.1.0-qdyf5j343pnyx43w2s3egtrk7v4j3f3n")
set(LOC "/home/thaham/bachelor_project/Jet-Partitioner/container")
set(KOKKOS_DIR "${LOC}/install/kokkos")
set(KOKKOS_KERNELS_DIR "${LOC}/install/kokkos-kernels")
set(JET_PARTITIONER_DIR "/home/thaham/bachelor_project/Jet-Partitioner")

# FlatBuffers Spack Path
set(FLATBUFFERS_DIR "/home/thaham/bachelor_project/spack/opt/spack/linux-ubuntu22.04-sapphirerapids/gcc-11.4.0/flatbuffers-24.3.7-63wczour3cdlvcj4j6ze4y3daixbpxqq")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Add debug info even in Release mode
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -g")
    
    # Enable some runtime checks even in Release
    add_definitions(-D_GLIBCXX_ASSERTIONS)
    
    # Add specific optimization flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-aggressive-loop-optimizations")
endif()

# Set the compiler to nvcc_wrapper if CUDA is enabled
if (WITH_CUDA)
    set(CMAKE_CXX_COMPILER "${KOKKOS_DIR}/bin/nvcc_wrapper")
endif()

# Find and enable OpenMP support
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
endif()

# Set CUDA NVCC flags to enable extended lambdas and set architecture
if (WITH_CUDA)
    set(CUDA_ARCHITECTURE "89")  # Compute capability 8.9 for H100 GPU
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${CUDA_ARCHITECTURE}")
endif()

# Find MPI
find_package(MPI REQUIRED)

# Include directories for KaHIP, Metis, Jet Partitioner, Kokkos, KokkosKernels, FlatBuffers
include_directories("${KAHIP_DIR}/include")
include_directories("${METIS_DIR}/include")
include_directories("${KOKKOS_DIR}/include")
include_directories("${KOKKOS_KERNELS_DIR}/include")
include_directories("${JET_PARTITIONER_DIR}")  # Include the directory containing jet.hpp
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")  # Include project root for experiment_data_generated.h
include_directories("${FLATBUFFERS_DIR}/include")  # FlatBuffers include path

# Link directories for KaHIP, Metis, Jet Partitioner, and dependencies
link_directories("${KAHIP_DIR}/lib")
link_directories("${METIS_DIR}/lib")
link_directories("${KOKKOS_DIR}/lib")
link_directories("${KOKKOS_KERNELS_DIR}/lib")
link_directories("${JET_PARTITIONER_DIR}/lib")
link_directories("${FLATBUFFERS_DIR}/lib")  # FlatBuffers library path

# Add the fap library
add_subdirectory(fap)
list(APPEND EXTRA_LIBS fap)
list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/fap")

# Set compiler flags to include necessary include paths and link flags for Kokkos, KaHIP, and Metis
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${KOKKOS_DIR}/include -I${KOKKOS_KERNELS_DIR}/include -I${KAHIP_DIR}/include -I${METIS_DIR}/include")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${KOKKOS_DIR}/lib -L${KOKKOS_KERNELS_DIR}/lib -L${KAHIP_DIR}/lib -L${METIS_DIR}/lib")

# Find all *Example.cpp files and store in list examples
file(GLOB_RECURSE examples RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*Example.cpp")

# Add each example file as a target
foreach(examplefile IN LISTS examples)
    get_filename_component(examplename ${examplefile} NAME_WE)
    add_executable(${examplename} ${examplefile})

    # Link libraries for each example
    target_link_libraries(${examplename} PUBLIC kahip metis fap MPI::MPI_CXX
        ${KOKKOS_DIR}/lib/libkokkoscore.a
        ${KOKKOS_DIR}/lib/libkokkoscontainers.a
        ${KOKKOS_DIR}/lib/libkokkossimd.a
        ${KOKKOS_KERNELS_DIR}/lib/libkokkoskernels.a
        ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX cuda
        "${FLATBUFFERS_DIR}/lib/libflatbuffers.so")  # Link FlatBuffers library

    # Add include paths for each example
    target_include_directories(${examplename} PUBLIC
        "${PROJECT_SOURCE_DIR}"
        "${EXTRA_INCLUDES}"
        "${KAHIP_DIR}/include"
        "${METIS_DIR}/include"
        "${KOKKOS_DIR}/include"
        "${KOKKOS_KERNELS_DIR}/include"
        "${JET_PARTITIONER_DIR}"
        "${CUDA_INCLUDE_DIRS}"
        "${FLATBUFFERS_DIR}/include"  # Include FlatBuffers
    )

    # Set CUDA-specific properties if CUDA is enabled
    if (WITH_CUDA)
        set_target_properties(${examplename} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_target_properties(${examplename} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE})
        target_compile_options(${examplename} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    endif()
endforeach()
