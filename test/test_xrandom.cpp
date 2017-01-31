/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xrandom.hpp"
#include "xtensor/xarray.hpp"

namespace xt
{
    TEST(xrandom, random)
    {
        auto r = random::rand<double>({3, 3});
        xarray<double> a = r;
        xarray<double> b = r;
        xarray<double> c = r;

        ASSERT_NE(a(0, 0), a(0, 1));
        ASSERT_NE(a, b);
        ASSERT_NE(a, c);

        xarray<double> other_rand = random::rand<double>({3, 3});
        ASSERT_NE(a, other_rand);

        random::seed(0);
        auto same_d_a = random::rand<double>({3, 3});
        xarray<double> same_a = same_d_a;

        random::seed(0);
        auto same_d_b = random::rand<double>({3, 3});
        xarray<double> same_b = same_d_b;

        ASSERT_EQ(same_a, same_b);

        // check that it compiles
        xarray<int> q = random::randint<int>({3, 3});

        // checking if internal state needs reset
        auto n_dist = random::randn<double>({3, 3});
        xarray<double> p1 = n_dist;
        xarray<double> p2 = n_dist;
        xarray<double> p3 = n_dist;
        ASSERT_NE(p1, p2);
        ASSERT_NE(p1, p3);
    }
}
