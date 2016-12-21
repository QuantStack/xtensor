/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xarray.hpp"

namespace xt
{
    using std::size_t;

    TEST(xbuilder, ones)
    {
        auto m = ones<double>({1, 2});
        ASSERT_EQ(2, m.dimension());
        ASSERT_EQ(1.0, m(0, 1));
        xarray<double> m_assigned = m;
        ASSERT_EQ(1.0, m_assigned(0, 1));
    }

    TEST(xarray, random)
    {
        xarray<double> a = random::rand<double>({20, 20});
        EXPECT_GE(a(0, 0), 0);
        EXPECT_LE(a(0, 0), 1);
        EXPECT_GE(a(19, 19), 0);
        EXPECT_LE(a(19, 19), 1);
    }
}
