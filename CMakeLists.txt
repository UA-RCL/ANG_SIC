cmake_minimum_required(VERSION 3.10)
project (ANG VERSION 1.0
	DESCRIPTION "Artificaial Neuro Genesis"
	LANGUAGES C)

# Setting proper architectural flags
set(ARCH_TYPES LINUX WINDOWS)
set (ARCH_TYPE "LINUX" CACHE STRING "Compilation flag to compile for Linux!" )

if (NOT ARCH_TYPE IN_LIST ARCH_TYPES)
  message(FATAL_ERROR "ARCH_TYPE must be one of ${ARCH_TYPES}")
endif()

if (${ARCH_TYPE} STREQUAL WINDOWS)
	message(FATAL_ERROR "Compilation support for WINDOWS is currently not integrated. Sowwy :(")
else()
	set(CMAKE_C_FLAGS "-D_LINUX")
endif()

# The flags that are set when DEBUG compilation mode is selected
set(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb")

# Add sources
set (ANG_SRCDIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set (ANG_SOURCES ${ANG_SRCDIR}/main.c ${ANG_SRCDIR}/Activation.c ${ANG_SRCDIR}/AnalyzeInputs.c
		${ANG_SRCDIR}/Bicubic.c ${ANG_SRCDIR}/Classes.c ${ANG_SRCDIR}/ClassLevelNetworks.c 
		${ANG_SRCDIR}/Config.c ${ANG_SRCDIR}/ConstructNetworks.c ${ANG_SRCDIR}/Convolution.c
		${ANG_SRCDIR}/FixedPoint.c ${ANG_SRCDIR}/Functions.c ${ANG_SRCDIR}/GaborFilter.c
		${ANG_SRCDIR}/Inference.c ${ANG_SRCDIR}/InputData.c ${ANG_SRCDIR}/InputFileCIFAR.c 
		${ANG_SRCDIR}/InputFileImagenette.c ${ANG_SRCDIR}/InputFileIR.c ${ANG_SRCDIR}/InputFileISAR.c
		${ANG_SRCDIR}/JenkFish.c ${ANG_SRCDIR}/layer.c ${ANG_SRCDIR}/LRI.c ${ANG_SRCDIR}/MAC.c
		${ANG_SRCDIR}/Network.c ${ANG_SRCDIR}/NeuroGenesis.c ${ANG_SRCDIR}/Perceptron.c
		${ANG_SRCDIR}/Prune.c ${ANG_SRCDIR}/Synapse.c ${ANG_SRCDIR}/Train.c)

# Specify executable to be generated
add_executable(ang ${ANG_SOURCES})

# Specify include directories and libraries
target_include_directories(ang PRIVATE ${ANG_SRCDIR}/include)
target_link_libraries(ang PRIVATE m)

