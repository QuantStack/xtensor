############################################################################
# Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xtensor-test)

    find_package(xtensor REQUIRED CONFIG)
    set(XTENSOR_INCLUDE_DIR ${xtensor_INCLUDE_DIRS})
endif ()

message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)

    if (HAS_CPP14_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
        message(FATAL_ERROR "Unsupported compiler -- xtensor requires C++14 support!")
    endif()
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj /wd4800")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    if(DOWNLOAD_GTEST)
        # Download and unpack googletest at configure time
        configure_file(downloadGTest.cmake.in googletest-download/CMakeLists.txt)
    else()
        # Copy local source of googletest at configure time
        configure_file(copyGTest.cmake.in googletest-download/CMakeLists.txt)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                     ${CMAKE_CURRENT_BINARY_DIR}/googletest-build)

    set(GTEST_INCLUDE_DIRS "${gtest_SOURCE_DIR}/include")
    set(GTEST_BOTH_LIBRARIES  gtest_main gtest)
else()
    find_package(GTest REQUIRED)
endif()

find_package(Threads)

include_directories(${GTEST_INCLUDE_DIRS} SYSTEM)

set(COMMON_BASE
    main.cpp
    test_common.hpp
    test_xsemantic.hpp
)

set(XTENSOR_TESTS
    test_xaccumulator.cpp
    test_xadapt.cpp
    test_xadaptor_semantic.cpp
    test_xarray.cpp
    test_xarray_adaptor.cpp
    test_xaxis_iterator.cpp
    test_xbroadcast.cpp
    test_xbuffer_adaptor.cpp
    test_xbuilder.cpp
    test_xconcepts.cpp
    test_xcontainer_semantic.cpp
    test_xcomplex.cpp
    test_xcsv.cpp
    test_xdatesupport.cpp
    test_xdynamic_view.cpp
    test_xeval.cpp
    test_xexception.cpp
    test_xexpression.cpp
    test_xfunction.cpp
    test_xfixed.cpp
    test_xhistogram.cpp
    test_xindex_view.cpp
    test_xinfo.cpp
    test_xiterator.cpp
    test_xio.cpp
    test_xlayout.cpp
    test_xmanipulation.cpp
    test_xmasked_value.cpp
    test_xmasked_view.cpp
    test_xmath.cpp
    test_xnan_functions.cpp
    test_xnoalias.cpp
    test_xnorm.cpp
    test_xnpy.cpp
    test_xoperation.cpp
    test_xoptional.cpp
    test_xoptional_assembly.cpp
    test_xoptional_assembly_adaptor.cpp
    test_xoptional_assembly_storage.cpp
    test_xrandom.cpp
    test_xreducer.cpp
    test_xscalar.cpp
    test_xscalar_semantic.cpp
    test_xshape.cpp
    test_xsort.cpp
    test_xstorage.cpp
    test_xstrided_view.cpp
    test_xstrides.cpp
    test_xtensor.cpp
    test_xtensor_adaptor.cpp
    test_xtensor_semantic.cpp
    test_xvectorize.cpp
    test_xview.cpp
    test_xview_semantic.cpp
    test_xutils.cpp
    test_extended_xmath_interp.cpp
    test_extended_broadcast_view.cpp
    test_extended_xmath_reducers.cpp
    test_extended_xhistogram.cpp
)

if(nlohmann_json_FOUND)
    list(APPEND XTENSOR_TESTS test_xjson.cpp)
    list(APPEND XTENSOR_TESTS test_xexpression_holder.cpp)
endif()

# remove xinfo tests for compilers < GCC 5
if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5))
    list(REMOVE_ITEM XTENSOR_TESTS test_xinfo.cpp)
endif()

set(XTENSOR_TARGET test_xtensor_lib)

# Add files for npy tests
set(XNPY_FILES
    bool.npy
    bool_fortran.npy
    double.npy
    double_fortran.npy
    unsignedlong.npy
    unsignedlong_fortran.npy
)
foreach(filename IN LISTS XNPY_FILES)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/files/xnpy_files/${filename}
        ${CMAKE_CURRENT_BINARY_DIR}/files/xnpy_files/${filename} COPYONLY)
endforeach()

set(XTENSOR_PREPROCESS_FILES
    files/cppy_source/test_xstrided_view_extended.cppy
)

add_custom_target(
    preprocess_cppy
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/files/preprocess.py
    DEPENDS ${XTENSOR_PREPROCESS_FILES}
)

foreach(filename IN LISTS XTENSOR_TESTS)
    string(REPLACE ".cpp" "" targetname ${filename})
    add_executable(${targetname} ${COMMON_BASE} ${filename} ${XTENSOR_HEADERS})
    target_link_libraries(${targetname} xtensor ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
        add_dependencies(${targetname} gtest_main)
    endif()
    add_custom_target(
        x${targetname}
        COMMAND ${targetname}
        DEPENDS ${targetname} ${filename} ${XTENSOR_HEADERS})
endforeach()

add_executable(${XTENSOR_TARGET} ${COMMON_BASE} ${XTENSOR_TESTS} ${XTENSOR_HEADERS})
if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    add_dependencies(${XTENSOR_TARGET} gtest_main)
endif()
target_link_libraries(${XTENSOR_TARGET} xtensor ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_custom_target(xtest COMMAND ${XTENSOR_TARGET} DEPENDS ${XTENSOR_TARGET})
