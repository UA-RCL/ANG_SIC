cmake_minimum_required(VERSION 3.10)

project(SIC
  VERSION 1.0
  DESCRIPTION "Synaptic Input Consolidation"
  LANGUAGES C)

# Architectural flags
set(ARCH_TYPES LINUX WINDOWS)
set(ARCH_TYPE "LINUX" CACHE STRING "Compilation flag to compile for Linux!")

if(NOT ARCH_TYPE IN_LIST ARCH_TYPES)
  message(FATAL_ERROR "ARCH_TYPE must be one of ${ARCH_TYPES}")
endif()

if(${ARCH_TYPE} STREQUAL WINDOWS)
  message(FATAL_ERROR "Compilation support for WINDOWS is currently not integrated.")
else()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_LINUX")
endif()

# Debug flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -ggdb")

# Sources
set(SIC_SRCDIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(SIC_SOURCES
  ${SIC_SRCDIR}/main.c
  ${SIC_SRCDIR}/Activation.c
  ${SIC_SRCDIR}/AnalyzeInputs.c
  ${SIC_SRCDIR}/Bicubic.c
  ${SIC_SRCDIR}/Classes.c
  ${SIC_SRCDIR}/ClassLevelNetworks.c
  ${SIC_SRCDIR}/Config.c
  ${SIC_SRCDIR}/ConstructNetworks.c
  ${SIC_SRCDIR}/Convolution.c
  ${SIC_SRCDIR}/FixedPoint.c
  ${SIC_SRCDIR}/Functions.c
  ${SIC_SRCDIR}/GaborFilter.c
  ${SIC_SRCDIR}/Inference.c
  ${SIC_SRCDIR}/InputData.c
  ${SIC_SRCDIR}/InputFileCIFAR.c
  ${SIC_SRCDIR}/InputFileImagenette.c
  ${SIC_SRCDIR}/InputFileIR.c
  ${SIC_SRCDIR}/InputFileISAR.c
  ${SIC_SRCDIR}/JenkFish.c
  ${SIC_SRCDIR}/layer.c
  ${SIC_SRCDIR}/LRI.c
  ${SIC_SRCDIR}/MAC.c
  ${SIC_SRCDIR}/Network.c
  ${SIC_SRCDIR}/NeuroGenesis.c
  ${SIC_SRCDIR}/Perceptron.c
  ${SIC_SRCDIR}/Prune.c
  ${SIC_SRCDIR}/Synapse.c
  ${SIC_SRCDIR}/Train.c
)

# Executable
add_executable(sic ${SIC_SOURCES})

# Includes & libs
target_include_directories(sic PRIVATE ${SIC_SRCDIR}/include)
target_link_libraries(sic PRIVATE m)
