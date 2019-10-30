/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xnpy.hpp"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

#include <fstream>
#include <cstdint>

namespace xt
{
    TEST(xnpy, load)
    {
        xarray<double> darr = {{{ 0.29731723,  0.04380157,  0.94748308},
                                { 0.85020643,  0.52958618,  0.0598172 },
                                { 0.77253259,  0.47564231,  0.70274005}},
                               {{ 0.85998447,  0.61160158,  0.44432939},
                                { 0.25506765,  0.97420976,  0.15455842},
                                { 0.05873659,  0.66191764,  0.01448838}},
                               {{ 0.175919  ,  0.13850365,  0.94059426},
                                { 0.79941809,  0.5124432 ,  0.51364796},
                                { 0.25721979,  0.41608858,  0.06255319}}};

        xarray<bool> barr = {{{ 0, 0, 1},
                              { 1, 1, 0},
                              { 1, 0, 1}},
                             {{ 1, 1, 0},
                              { 0, 1, 0},
                              { 0, 1, 0}},
                             {{ 0, 0, 1},
                              { 1, 1, 1},
                              { 0, 0, 0}}};

        xarray<int> iarr1d = {3, 4, 5, 6, 7};

        auto darr_loaded = load_npy<double>("files/xnpy_files/double.npy");
        EXPECT_TRUE(all(isclose(darr, darr_loaded)));

        std::ifstream dstream("files/xnpy_files/double.npy");
        auto darr_loaded_stream = load_npy<double>(dstream);
        EXPECT_TRUE(all(isclose(darr, darr_loaded_stream)))
            << "Loading double numpy array from stream failed";
        dstream.close();

        auto barr_loaded = load_npy<bool>("files/xnpy_files/bool.npy");
        EXPECT_TRUE(all(equal(barr, barr_loaded)));

        std::ifstream bstream("files/xnpy_files/bool.npy");
        auto barr_loaded_stream = load_npy<bool>(bstream);
        EXPECT_TRUE(all(equal(barr, barr_loaded_stream)))
            << "Loading boolean numpy array from stream failed";
        bstream.close();

        auto dfarr_loaded = load_npy<double, layout_type::column_major>("files/xnpy_files/double_fortran.npy");
        EXPECT_TRUE(all(isclose(darr, dfarr_loaded)));

        auto iarr1d_loaded = load_npy<int>("files/xnpy_files/int.npy");
        EXPECT_TRUE(all(equal(iarr1d, iarr1d_loaded)));
    }

    bool compare_binary_files(std::string fn1, std::string fn2)
    {
        std::ifstream stream1(fn1, std::ios::in | std::ios::binary);
        std::vector<uint8_t> fn1_contents((std::istreambuf_iterator<char>(stream1)),
                                          std::istreambuf_iterator<char>());

        std::ifstream stream2(fn2, std::ios::in | std::ios::binary);
        std::vector<uint8_t> fn2_contents((std::istreambuf_iterator<char>(stream2)),
                                          std::istreambuf_iterator<char>());
        return std::equal(fn1_contents.begin(), fn1_contents.end(), fn2_contents.begin()) &&
            fn1_contents.size() == fn2_contents.size();
    }

    std::string get_filename(int n)
    {
        std::string filename = "files/xnpy_files/test_dump_" + std::to_string(n) + ".npy";
        return filename;
    }

    std::string read_file(const std::string& name)
    {
        return static_cast<std::stringstream const&>(std::stringstream() << std::ifstream(name).rdbuf()).str();
    }

    TEST(xnpy, dump)
    {
        std::string filename = get_filename(0);
        xarray<bool> barr = {{{0, 0, 1},
                              {1, 1, 0},
                              {1, 0, 1}},
                             {{1, 1, 0},
                              {0, 1, 0},
                              {0, 1, 0}},
                             {{0, 0, 1},
                              {1, 1, 1},
                              {0, 0, 0}}};

        xtensor<uint64_t, 1> ularr = {12ul, 14ul, 16ul, 18ul, 1234321ul};
        dump_npy(filename, barr);

        std::string compare_name = "files/xnpy_files/bool.npy";
        if (barr.layout() == layout_type::column_major)
        {
            compare_name = "files/xnpy_files/bool_fortran.npy";
        }

        EXPECT_TRUE(compare_binary_files(filename, compare_name));

        std::string barr_str = dump_npy(barr);
        std::string barr_disk = read_file(compare_name);
        EXPECT_EQ(barr_str, barr_disk) << "Dumping boolean numpy file to string failed";

        std::remove(filename.c_str());

        filename = get_filename(1);
        dump_npy(filename, ularr);
        auto ularrcpy = load_npy<uint64_t>(filename);
        EXPECT_TRUE(all(equal(ularr, ularrcpy)));

        compare_name = "files/xnpy_files/unsignedlong.npy";
        if (barr.layout() == layout_type::column_major)
        {
            compare_name = "files/xnpy_files/unsignedlong_fortran.npy";
        }

        EXPECT_TRUE(compare_binary_files(filename, compare_name));

        std::string ularr_str = dump_npy(ularr);
        std::string ularr_disk = read_file(compare_name);
        EXPECT_EQ(ularr_str, ularr_disk) << "Dumping boolean numpy file to string failed";

        std::remove(filename.c_str());
    }

    TEST(xnpy, xfunction_cast)
    {
        // compilation test, cf: https://github.com/xtensor-stack/xtensor/issues/1070
        auto dc = cast<char>(load_npy<double>("files/xnpy_files/double.npy"));
        EXPECT_EQ(dc(0, 0), 0);
        xarray<char> adc = dc;
        EXPECT_EQ(adc(0, 0), 0);
    }
}
