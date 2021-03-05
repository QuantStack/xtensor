############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)
project(xtensor)

set(XTENSOR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versionning
# ===========

file(STRINGS "${XTENSOR_INCLUDE_DIR}/xtensor/xtensor_config.hpp" xtensor_version_defines
     REGEX "#define XTENSOR_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${xtensor_version_defines})
    if(ver MATCHES "#define XTENSOR_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(XTENSOR_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
    ${XTENSOR_VERSION_MAJOR}.${XTENSOR_VERSION_MINOR}.${XTENSOR_VERSION_PATCH})
message(STATUS "Building xtensor v${${PROJECT_NAME}_VERSION}")

# Dependencies
# ============

set(xtl_REQUIRED_VERSION 0.7.0)
if(TARGET xtl)
    set(xtl_VERSION ${XTL_VERSION_MAJOR}.${XTL_VERSION_MINOR}.${XTL_VERSION_PATCH})
    # Note: This is not SEMVER compatible comparison
    if( NOT ${xtl_VERSION} VERSION_GREATER_EQUAL ${xtl_REQUIRED_VERSION})
        message(ERROR "Mismatch xtl versions. Found '${xtl_VERSION}' but requires: '${xtl_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xtl v${xtl_VERSION}")
    endif()
else()
    find_package(xtl ${xtl_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xtl: ${xtl_INCLUDE_DIRS}/xtl")
endif()

find_package(nlohmann_json 3.1.1 QUIET)

# Optional dependencies
# =====================

OPTION(XTENSOR_USE_XSIMD "simd acceleration for xtensor" OFF)
OPTION(XTENSOR_USE_TBB "enable parallelization using intel TBB" OFF)
OPTION(XTENSOR_USE_OPENMP "enable parallelization using OpenMP" OFF)
if(XTENSOR_USE_TBB AND XTENSOR_USE_OPENMP)
    message(
        FATAL
        "XTENSOR_USE_TBB and XTENSOR_USE_OPENMP cannot both be active at once"
    )
endif()

if(XTENSOR_USE_XSIMD)
    set(xsimd_REQUIRED_VERSION 7.4.4)
    if(TARGET xsimd)
        set(xsimd_VERSION ${XSIMD_VERSION_MAJOR}.${XSIMD_VERSION_MINOR}.${XSIMD_VERSION_PATCH})
        # Note: This is not SEMVER compatible comparison
        if( NOT ${xsimd_VERSION} VERSION_GREATER_EQUAL ${xsimd_REQUIRED_VERSION})
            message(ERROR "Mismatch xsimd versions. Found '${xsimd_VERSION}' but requires: '${xsimd_REQUIRED_VERSION}'")
        else()
            message(STATUS "Found xsimd v${xsimd_VERSION}")
        endif()
    else()
        find_package(xsimd ${xsimd_REQUIRED_VERSION} REQUIRED)
        message(STATUS "Found xsimd: ${xsimd_INCLUDE_DIRS}/xsimd")
    endif()
endif()

if(XTENSOR_USE_TBB)
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
    find_package(TBB REQUIRED)
    message(STATUS "Found intel TBB: ${TBB_INCLUDE_DIRS}")
endif()

if(XTENSOR_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        # Set openmp variables now

        # Create private target just for this lib
        # https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
        # Probably not safe for cmake < 3.4 ..
        find_package(Threads REQUIRED)
        add_library(OpenMP::OpenMP_CXX_xtensor IMPORTED INTERFACE)
        set_property(
            TARGET
            OpenMP::OpenMP_CXX_xtensor
            PROPERTY
            INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS}
        )
        # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
        set_property(
            TARGET
            OpenMP::OpenMP_CXX_xtensor
            PROPERTY
            INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)

        message(STATUS "OpenMP Found")
    else()
        message(FATAL "Failed to locate OpenMP")
    endif()
endif()

# Build
# =====

set(XTENSOR_HEADERS
    ${XTENSOR_INCLUDE_DIR}/xtensor/xaccessible.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xaccumulator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xadapt.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xarray.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xassign.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xaxis_iterator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xaxis_slice_iterator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xbroadcast.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xbuffer_adaptor.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xbuilder.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xchunked_array.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xcomplex.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xcontainer.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xcsv.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xdynamic_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xeval.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xexception.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xexpression.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xexpression_holder.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xexpression_traits.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xfixed.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xfunction.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xfunctor_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xgenerator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xhistogram.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xindex_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xinfo.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xio.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xiterable.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xiterator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xjson.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xlayout.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xmanipulation.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xmasked_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xmath.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xmime.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xnoalias.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xnorm.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xnpy.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoffset_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoperation.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoptional.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoptional_assembly.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoptional_assembly_base.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoptional_assembly_storage.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xpad.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xrandom.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xreducer.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xrepeat.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xscalar.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xsemantic.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xset_operation.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xshape.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xslice.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xsort.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xstorage.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xstrided_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xstrided_view_base.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xstrides.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xtensor.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xtensor_config.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xtensor_forward.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xtensor_simd.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xutils.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xvector.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xvectorize.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xview.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xview_utils.hpp
)

add_library(xtensor INTERFACE)

target_include_directories(xtensor INTERFACE
    $<BUILD_INTERFACE:${XTENSOR_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

target_compile_features(xtensor INTERFACE cxx_std_14)

target_link_libraries(xtensor INTERFACE xtl)

OPTION(XTENSOR_ENABLE_ASSERT "xtensor bound check" OFF)
OPTION(XTENSOR_CHECK_DIMENSION "xtensor dimension check" OFF)
OPTION(BUILD_TESTS "xtensor test suite" OFF)
OPTION(BUILD_BENCHMARK "xtensor benchmark" OFF)
OPTION(DOWNLOAD_GTEST "build gtest from downloaded sources" OFF)
OPTION(DOWNLOAD_GBENCHMARK "download google benchmark and build from source" ON)
OPTION(DEFAULT_COLUMN_MAJOR "set default layout to column major" OFF)
OPTION(DISABLE_VS2017 "disables the compilation of some test with Visual Studio 2017" OFF)
OPTION(CPP17 "enables C++17" OFF)
OPTION(CPP20 "enables C++20 (experimental)" OFF)
OPTION(XTENSOR_DISABLE_EXCEPTIONS "Disable C++ exceptions" OFF)
OPTION(DISABLE_MSVC_ITERATOR_CHECK "Disable the MVSC iterator check" ON)

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    set(BUILD_TESTS ON)
endif()

if(XTENSOR_ENABLE_ASSERT OR XTENSOR_CHECK_DIMENSION)
    add_definitions(-DXTENSOR_ENABLE_ASSERT)
endif()

if(XTENSOR_CHECK_DIMENSION)
    add_definitions(-DXTENSOR_ENABLE_CHECK_DIMENSION)
endif()

if(DEFAULT_COLUMN_MAJOR)
    add_definitions(-DXTENSOR_DEFAULT_LAYOUT=layout_type::column_major)
endif()

if(DISABLE_VS2017)
    add_definitions(-DDISABLE_VS2017)
endif()

if(MSVC AND DISABLE_MSVC_ITERATOR_CHECK)
    add_compile_definitions($<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=0>)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()

if(XTENSOR_USE_OPENMP)
    # Link xtensor itself to OpenMP to propagate to user projects
    target_link_libraries(xtensor INTERFACE OpenMP::OpenMP_CXX_xtensor)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS xtensor
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(FILES ${XTENSOR_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xtensor)

set(XTENSOR_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for xtensorConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${XTENSOR_CMAKECONFIG_INSTALL_DIR})

# xtensor is header-only and does not depend on the architecture.
# Remove CMAKE_SIZEOF_VOID_P from xtensorConfigVersion.cmake so that an xtensorConfig.cmake
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
set(_XTENSOR_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_XTENSOR_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${XTENSOR_CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${XTENSOR_CMAKECONFIG_INSTALL_DIR})

configure_file(${PROJECT_NAME}.pc.in
               "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
                @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")

# Write single include
# ====================

function(PREPEND var prefix)
   set(listVar "")
   foreach(f ${ARGN})
      list(APPEND listVar "${prefix}${f}")
   endforeach(f)
   set(${var} "${listVar}" PARENT_SCOPE)
endfunction()

function(POSTFIX var postfix)
   set(listVar "")
   foreach(f ${ARGN})
      list(APPEND listVar "${f}${postfix}")
   endforeach(f)
   set(${var} "${listVar}" PARENT_SCOPE)
endfunction()

set(XTENSOR_SINGLE_INCLUDE ${XTENSOR_HEADERS})
string(REPLACE "${XTENSOR_INCLUDE_DIR}/" "" XTENSOR_SINGLE_INCLUDE "${XTENSOR_SINGLE_INCLUDE}")
list(REMOVE_ITEM XTENSOR_SINGLE_INCLUDE
    xtensor/xexpression_holder.hpp
    xtensor/xjson.hpp
    xtensor/xmime.hpp
    xtensor/xnpy.hpp)

PREPEND(XTENSOR_SINGLE_INCLUDE "#include <" ${XTENSOR_SINGLE_INCLUDE})
POSTFIX(XTENSOR_SINGLE_INCLUDE ">" ${XTENSOR_SINGLE_INCLUDE})
string(REPLACE ";" "\n" XTENSOR_SINGLE_INCLUDE "${XTENSOR_SINGLE_INCLUDE}")
string(CONCAT XTENSOR_SINGLE_INCLUDE "#ifndef XTENSOR\n" "#define XTENSOR\n\n" "${XTENSOR_SINGLE_INCLUDE}" "\n\n#endif\n")

file(WRITE "${CMAKE_BINARY_DIR}/xtensor.hpp" "${XTENSOR_SINGLE_INCLUDE}")

install(FILES "${CMAKE_BINARY_DIR}/xtensor.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
