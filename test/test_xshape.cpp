/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xshape.hpp"

#include "gtest/gtest.h"
#include "xtensor/xbroadcast.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"

namespace xt
{
    TEST(xshape, initializer_dimension)
    {
        size_t d0 = initializer_dimension<double>::value;
        size_t d1 = initializer_dimension<std::initializer_list<double>>::value;
        size_t d2 = initializer_dimension<std::initializer_list<std::initializer_list<double>>>::value;
        EXPECT_EQ(size_t(0), d0);
        EXPECT_EQ(size_t(1), d1);
        EXPECT_EQ(size_t(2), d2);
    }

    TEST(xshape, shape)
    {
        auto s0 = shape<std::vector<size_t>>(3);
        auto s1 = shape<std::vector<size_t>>(std::initializer_list<size_t>{1, 2});
        auto s2 = shape<std::vector<size_t>>(std::initializer_list<std::initializer_list<size_t>>{{1, 2, 4}, {1, 3, 5}});

        std::vector<size_t> e0 = {};
        std::vector<size_t> e1 = {2};
        std::vector<size_t> e2 = {2, 3};

        EXPECT_EQ(e0, s0);
        EXPECT_EQ(e1, s1);
        EXPECT_EQ(e2, s2);
    }

    TEST(xshape, promote_shape)
    {
        bool expect_v = std::is_same<
            dynamic_shape<size_t>,
            promote_shape_t<dynamic_shape<size_t>, std::array<size_t, 3>, std::array<size_t, 2>>
        >::value;

        bool expect_a = std::is_same<
            std::array<size_t, 3>,
            promote_shape_t<std::array<size_t, 2>, std::array<size_t, 3>, std::array<size_t, 2>>
        >::value;

        ASSERT_TRUE(expect_v);
        ASSERT_TRUE(expect_a);
    }
}
