/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xscalar.hpp"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "test_common_macros.hpp"

namespace xt
{
    TEST(xscalar, size)
    {
        // The shape of a 0-D xarray is ().  The size of the buffer is 1.
        xscalar<int> x(1);
        EXPECT_EQ(x.size(), size_t(1));
    }

    TEST(xscalar, construct_convert)
    {
        xscalar<int> a, b, c;
        b = 123;
        c = 3;

        int bc = b;
        int ac = a;
        int abc = a + b;
        EXPECT_EQ(b(), 123);
        EXPECT_EQ(bc, 123);
        EXPECT_EQ(ac, 0);
        EXPECT_EQ(abc, 123);
    }

    TEST(xscalar, access)
    {
        // Calling operator() with no argument returns the wrapped value.
        xscalar<int> x(2);
        EXPECT_EQ(x(), 2);

        x() = 4;
        EXPECT_EQ(4, x());
    }

    TEST(xscalar, indexed_access)
    {
        // Calling operator() with no argument returns the wrapped value.
        xscalar<int> x(2);
        EXPECT_EQ((x[{0}]), 2);

        x[{0}] = 4;
        EXPECT_EQ(4, (x[{0}]));
    }

    TEST(xscalar, at)
    {
        xscalar<int> x(2);
        EXPECT_ANY_THROW(x.at(0));
    }

    TEST(xscalar, dimension)
    {
        // The dimension of a xscalar is 0
        xscalar<int> x(2);
        EXPECT_EQ(x.dimension(), size_t(0));
        EXPECT_EQ(x.layout(), layout_type::any);
    }

    TEST(xscalar, iterator)
    {
        xscalar<int> x(2);
        auto xiter = x.begin();
        *xiter = 4;
        EXPECT_EQ(4, x());
        auto xiter_end = x.end();
        ++xiter;
        EXPECT_EQ(xiter, xiter_end);
        auto iter = x.begin();
        *iter = 3;
        EXPECT_EQ(3, x());
        auto iter_end = x.end();
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }


    TEST(xscalar, stepper)
    {
        xarray<double> ref = zeros<double>({4, 6});
        xarray<int> a{0, 1};
        ref[a + 1] = 1;
        EXPECT_EQ(ref(1, 2), 1);
    }

    TEST(xscalar, dummy_iterator)
    {
        xscalar<int> x(2);
        auto iter = x.dummy_begin();
        auto iter_end = x.dummy_end();
        EXPECT_EQ(iter, iter_end);
        ++iter;
        EXPECT_EQ(2, *iter);
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xscalar, xref)
    {
        int ref = 4;
        int x = 2;
        auto s = xref(x);
        s() = ref;
        EXPECT_EQ(ref, x);
    }

    TEST(xscalar, broadcasting)
    {
        xarray<int> a = {1, 2, 3, 4};
        xarray<int> ref = {6, 7, 8, 9};
        xarray<int> b = a + 5;
        EXPECT_EQ(ref, b);
    }

    TEST(xscalar, all_scalar)
    {
        auto tr = all_xscalar<xscalar<double>, xscalar<int>>::value;
        EXPECT_TRUE(tr);
        auto fa = all_xscalar<xscalar<double>, xarray<int>>::value;
        EXPECT_FALSE(fa);
    }
}
