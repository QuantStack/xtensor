/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xaxis_iterator.hpp"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

namespace xt
{
    using std::size_t;

    xarray<int> get_test_array()
    {
        xarray<int> res = {{{1, 2, 3, 4},
                            {5, 6, 7, 8},
                            {9, 10, 11, 12}},
                           {{13, 14, 15, 16},
                            {17, 18, 19, 20},
                            {21, 22, 23, 24}}};
        return res;
    }

    TEST(xaxis_iterator, begin)
    {
        xarray<int> a = get_test_array();
        auto iter_begin = axis_begin(a);
        EXPECT_EQ(size_t(2), iter_begin->dimension());
        EXPECT_EQ(a.shape()[1], iter_begin->shape()[0]);
        EXPECT_EQ(a.shape()[2], iter_begin->shape()[1]);

        EXPECT_EQ(a(0, 0, 0), (*iter_begin)(0, 0));
        EXPECT_EQ(a(0, 1, 1), (*iter_begin)(1, 1));
        EXPECT_EQ(a(0, 2, 3), (*iter_begin)(2, 3));
    }

    TEST(xaxis_iterator, increment)
    {
        xarray<int> a = get_test_array();
        auto iter = axis_begin(a);
        ++iter;

        EXPECT_EQ(size_t(2), iter->dimension());
        EXPECT_EQ(a.shape()[1], iter->shape()[0]);
        EXPECT_EQ(a.shape()[2], iter->shape()[1]);

        EXPECT_EQ(a(1, 0, 0), (*iter)(0, 0));
        EXPECT_EQ(a(1, 1, 1), (*iter)(1, 1));
        EXPECT_EQ(a(1, 2, 3), (*iter)(2, 3));
    }

    TEST(xaxis_iterator, end)
    {
        xarray<int> a = get_test_array();
        auto iter_begin = axis_begin(a);
        auto iter_end = axis_end(a);
        auto dist = std::distance(iter_begin, iter_end);
        EXPECT_EQ(2, dist);
    }

    TEST(xaxis_iterator, nested)
    {
        xarray<int> a = get_test_array();
        auto iter = axis_begin(a);
        ++iter;
        auto niter = axis_begin(*iter);
        ++niter;
        EXPECT_EQ(size_t(1), niter->dimension());
        EXPECT_EQ(a.shape()[2], niter->shape()[0]);
        EXPECT_EQ(a(1, 1, 0), (*niter)(0));
        EXPECT_EQ(a(1, 1, 1), (*niter)(1));
        EXPECT_EQ(a(1, 1, 2), (*niter)(2));
        EXPECT_EQ(a(1, 1, 3), (*niter)(3));
    }

    TEST(xaxis_iterator, const_array)
    {
        const xarray<int> a = get_test_array();
        auto iter = axis_begin(a);
        ++iter;

        EXPECT_EQ(size_t(2), iter->dimension());
        EXPECT_EQ(a.shape()[1], iter->shape()[0]);
        EXPECT_EQ(a.shape()[2], iter->shape()[1]);

        EXPECT_EQ(a(1, 0, 0), (*iter)(0, 0));
        EXPECT_EQ(a(1, 1, 1), (*iter)(1, 1));
        EXPECT_EQ(a(1, 2, 3), (*iter)(2, 3));
    }
}
