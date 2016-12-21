/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include <vector>
#include <algorithm>
#include <sstream>

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xio.hpp"


namespace xt
{

    TEST(xio, one_d)
    {
        xarray<double> e{1, 2, 3, 4, 5};
        std::stringstream out;
        out << e;
        EXPECT_EQ("{1, 2, 3, 4, 5}", out.str());
    }

    TEST(xio, two_d)
    {
        xarray<double> e{{1, 2, 3, 4},
                         {5, 6, 7, 8},
                         {9, 10, 11, 12}};
        std::stringstream out;
        out << e;
        EXPECT_EQ(R"xio({{1, 2, 3, 4},
 {5, 6, 7, 8},
 {9, 10, 11, 12}})xio", out.str());
    }

    TEST(xio, stacked)
    {
        xarray<double> e = {1, 2, 3, 4, 5};
        auto stacked = vstack(e, e);
        std::stringstream v_out;
        v_out << stacked;
        EXPECT_EQ(R"xio({{1, 2, 3, 4, 5},
 {1, 2, 3, 4, 5}})xio", v_out.str());

        auto h_stacked = hstack(e, e);
        std::stringstream h_out;
        h_out << h_stacked;
        EXPECT_EQ(R"xio({1, 2, 3, 4, 5, 1, 2, 3, 4, 5})xio", h_out.str());
    }

    TEST(xio, view)
    {
        xarray<double> e{{1, 2, 3, 4},
                         {5, 6, 7, 8},
                         {9, 10, 11, 12}};

        auto v_1 = make_xview(e, 1, xt::all());
        auto v_2 = make_xview(e, xt::all(), 1);

        std::stringstream out_1;
        out_1 << v_1;
        EXPECT_EQ("{5, 6, 7, 8}", out_1.str());

        std::stringstream out_2;
        out_2 << v_2;
        EXPECT_EQ("{2, 6, 10}", out_2.str());
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
        EXPECT_EQ(R"xio({{{1, 2},
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
  {4, 3}}})xio", out.str());
    }
}

