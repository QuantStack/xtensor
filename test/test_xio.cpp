/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xio.hpp"

#include <vector>
#include <algorithm>
#include <sstream>
#include <limits>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xdynamic_view.hpp"
#include "xtensor/xview.hpp"

#include "files/xio_expected_results.hpp"

namespace xt
{
    TEST(xio, xtensor_one_d)
    {
        xtensor<double, 1> e = xt::arange<double>(1, 6);
        std::stringstream out;
        out << e;
        EXPECT_EQ("{ 1.,  2.,  3.,  4.,  5.}", out.str());
    }

    TEST(xio, xarray_one_d)
    {
        xarray<double> e{1, 2, 3, 4, 5};
        std::stringstream out;
        out << e;
        EXPECT_EQ("{ 1.,  2.,  3.,  4.,  5.}", out.str());
    }

    TEST(xio, xarray_two_d)
    {
        xarray<double> e{{1, 2, 3, 4},
                         {5, 6, 7, 8},
                         {9, 10, 11, 12}};
        std::stringstream out;
        out << e;
        EXPECT_EQ(twod_double, out.str());
    }

    TEST(xio, view)
    {
        xarray<double> e{{1, 2, 3, 4},
                         {5, 6, 7, 8},
                         {9, 10, 11, 12}};

        auto v_1 = view(e, 1, xt::all());
        auto v_2 = view(e, xt::all(), 1);
        auto v_new_axis = view(e, 1, xt::newaxis(), xt::all());

        xarray<int> c = {1, 2, 3, 4};
        auto v_just_new_axis = view(c, xt::newaxis());

        std::stringstream out_1;
        out_1 << v_1;
        EXPECT_EQ("{ 5.,  6.,  7.,  8.}", out_1.str());

        std::stringstream out_2;
        out_2 << v_2;
        EXPECT_EQ("{  2.,   6.,  10.}", out_2.str());

        std::stringstream out_3;
        out_3 << v_new_axis;
        EXPECT_EQ("{{ 5.,  6.,  7.,  8.}}", out_3.str());

        std::stringstream out_4;
        out_4 << v_just_new_axis;
        EXPECT_EQ("{{1, 2, 3, 4}}", out_4.str());
    }

    TEST(xio, xdynamic_view)
    {
        xarray<int> e{{1, 2, 3, 4},
                      {5, 6, 7, 8},
                      {9, 10, 11, 12}};
        auto v = xt::dynamic_view(e, { 1, keep(0, 2, 3)});
        std::stringstream out;
        out << v;
        EXPECT_EQ("{5, 7, 8}", out.str());
    }

    TEST(xio, random_nan_inf)
    {
        xt::random::seed(123);
        xt::xarray<double, layout_type::row_major> rn = xt::random::rand<double>({20, 20}, -10, 10);
        rn(1, 1) = -1;
        rn(1, 2) = +1;
        rn(1, 1) = -1;
        rn(2, 2) = std::numeric_limits<double>::infinity();  //  inf
        rn(2, 3) = -std::numeric_limits<double>::infinity();  // -inf
        rn(4, 4) = std::nan("xnan");
        std::stringstream out;

        out << rn;
        EXPECT_EQ(random_nan_inf, out.str());
    }

    TEST(xio, big_exp)
    {
        xt::random::seed(123);
        xt::xarray<double, layout_type::row_major> rn = xt::random::rand<double>({5, 4}, -10, 10);
        rn(1, 1) = 1e220;
        rn(1, 2) = 1e-124;

        std::stringstream out;
        out << rn;

        EXPECT_EQ(big_exp, out.str());
    }

    TEST(xio, precision)
    {
        xt::random::seed(123);
        xt::xarray<double, layout_type::row_major> rn = xt::random::rand<double>({5, 4}, -10, 10);

        std::stringstream out;
        out << std::setprecision(12) << rn;
        EXPECT_EQ(precision, out.str());
    }

    TEST(xio, bool_fn)
    {
        xt::random::seed(123);
        xt::xarray<double, layout_type::row_major> rn = xt::random::rand<double>({5, 5}, -10, 10);

        std::stringstream out;
        out << (rn > 0);
        std::string res = bool_fn;
        EXPECT_EQ(res, out.str());
    }

    TEST(xio, cutoff)
    {
        xt::xarray<int> rn = xt::ones<int>({1, 1001});

        std::stringstream out;
        out << rn;
        EXPECT_EQ("{{1, 1, 1, ..., 1, 1, 1}}", out.str());

        std::stringstream out2;
        xt::xarray<int> rn2 = xt::ones<int>({1001, 1});
        out2 << rn2;
        EXPECT_EQ(cut_high, out2.str());
    }

    TEST(xio, cut_longwise)
    {
        xt::xarray<unsigned int> a = xt::ones<unsigned int>({5, 1000});

        std::stringstream out;
        out << a;
        EXPECT_EQ(cut_long, out.str());

        xt::xarray<int> b = xt::ones<int>({7, 1000});

        std::stringstream outb;
        outb << b;
        EXPECT_EQ(cut_both, outb.str());

        xt::xarray<int> c = xt::ones<int>({7, 7, 7, 1000});

        std::stringstream outc;
        outc << c;
        EXPECT_EQ(cut_4d, outc.str());
    }

    TEST(xio, options)
    {
        xt::random::seed(123);
        xt::xarray<double, layout_type::row_major> rn = xt::random::rand<double>({100, 100}, -10, 10);

        xt::print_options::set_line_width(150);
        xt::print_options::set_edge_items(10);
        xt::print_options::set_precision(10);
        xt::print_options::set_threshold(100);

        std::stringstream out;
        out << rn;
        EXPECT_EQ(print_options_result, out.str());

        // reset back to default
        xt::print_options::set_line_width(75);
        xt::print_options::set_edge_items(3);
        xt::print_options::set_precision(-1);
        xt::print_options::set_threshold(1000);
    }

    namespace po = xt::print_options;

    TEST(xio, local_options)
    {
        xt::random::seed(123);
        xt::xarray<double, layout_type::row_major> rn = xt::random::rand<double>({100, 100}, -10, 10);

        std::stringstream out;
        out << po::line_width(150)
            << po::edge_items(10)
            << po::precision(10)
            << po::threshold(100)
            << rn;

        EXPECT_EQ(print_options_result, out.str());

        EXPECT_EQ(out.iword(po::edge_items::id()), long(0));
        EXPECT_EQ(out.iword(po::line_width::id()), long(0));
        EXPECT_EQ(out.iword(po::threshold::id()), long(0));
        EXPECT_EQ(out.iword(po::precision::id()), long(0));
    }

    TEST(xio, three_d)
    {
        xarray<double> e{{{1, 2},
                          {3, 4},
                          {5, 6},
                          {7, 8}},
                         {{9, 10},
                          {11, 12},
                          {7, 9},
                          {11, 14}},
                         {{5, 26},
                          {7, 8},
                          {10, 8},
                          {4, 3}}};
        std::stringstream out;
        out << e;
        EXPECT_EQ(threed_double, out.str());
    }

    TEST(xio, strings)
    {
        xt::xarray<std::string> e = {{"some", "random", "boring"}, {"strings", "in", "xtensor xarray"}};
        std::stringstream out;
        out << e;
        EXPECT_EQ(random_strings, out.str());
    }

    TEST(xio, long_strings)
    {
        xt::xarray<std::string> e = {{"some", "random very long and very very", "boring"}, {"strings", "in", "xtensor xarray"}};
        std::stringstream out;
        out << e;
        EXPECT_EQ(long_strings, out.str());
    }

    TEST(xio, complex)
    {
        xt::random::seed(123);
        xt::xarray<double, layout_type::row_major> real = xt::random::rand<double>({10, 10}, -10, 10);
        xt::xarray<double, layout_type::row_major> imag = xt::random::rand<double>({10, 10}, -5, 5);
        xt::xarray<std::complex<double>, layout_type::row_major> e = real + (imag * std::complex<double>(0, 1));

        std::stringstream out;
        out << e;
        EXPECT_EQ(complex_numbers, out.str());
    }

    TEST(xio, complex_zero_erasing)
    {
        xt::random::seed(123);
        xt::xarray<double, layout_type::row_major> real = xt::random::rand<double>({10, 10}) - 0.5;
        xt::xarray<double, layout_type::row_major> imag = xt::random::rand<double>({10, 10}) - 0.5;
        xt::xarray<std::complex<double>, layout_type::row_major> e = real + (imag * std::complex<double>(0, 1));

        std::stringstream out;
        out << e;
        EXPECT_EQ(complex_zero_erasing, out.str());
    }

    TEST(xio, float_leading_zero)
    {
        xt::random::seed(123);
        std::stringstream out;
        out << xt::random::rand<double>({10, 10}) - 0.5;
        EXPECT_EQ(float_leading_zero, out.str());
    }

    TEST(xio, custom_formatter)
    {
        xt::xarray<int> e = {{1, 2, 3, 4}, {100, 200, 1000, 10000000}};

        std::stringstream out;
        pretty_print(e, [](const int& val) {
            std::stringstream buf;
            buf << "0x" << std::hex << val;
            return buf.str();
        }, out);

        EXPECT_EQ(custom_formatter_result, out.str());
    }

    TEST(xio, view_on_broadcast)
    {
        auto on = xt::ones<int>({5, 5});
        auto von = xt::view(on, 1);
        std::stringstream out;
        out << von;
        std::string exp = "{1, 1, 1, 1, 1}";
        EXPECT_EQ(exp, out.str());
    }

    TEST(xio, flags_reset)
    {
        xt::xarray<double> aod = {123400000., 123400000.};
        std::stringstream out;
        out << aod;
        double d = 2.119;
        out << '\n' << d;
        std::string exp = "{ 1.234000e+08,  1.234000e+08}\n2.119";
        EXPECT_EQ(exp, out.str());
    }
}
