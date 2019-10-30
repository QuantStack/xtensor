/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xhistogram.hpp"

#include <complex>
#include <limits>

#include "gtest/gtest.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"

namespace xt
{
    TEST(xhistogram, histogram)
    {
        xt::xtensor<double,1> data = {1., 1., 2., 2.};

        {
            xt::xtensor<double,1> count = xt::histogram(data, std::size_t(2));

            EXPECT_EQ(count.size(), std::size_t(2) );
            EXPECT_EQ(count[0]    , 2.);
            EXPECT_EQ(count[1]    , 2.);
        }

        {
            xt::xtensor<double,1> count = xt::histogram(data,
                xt::histogram_bin_edges(data, std::size_t(2), xt::histogram_algorithm::uniform)
            );

            EXPECT_EQ(count.size(), std::size_t(2));
            EXPECT_EQ(count[0]    , 2.);
            EXPECT_EQ(count[1]    , 2.);
        }
    }

    TEST(xhistogram, bincount)
    {
        xtensor<int, 1> data = {1,2,3,1,1,1,1,2,3,2,3,3,3,3};
        xtensor<int, 1> weights = xt::ones<int>(data.shape()) * 3;
        xtensor<int, 1> expc = {0, 5, 3, 6};
        auto bc = bincount(data);
        EXPECT_EQ(bc, expc);

        auto bc2 = bincount(data, weights);
        EXPECT_EQ(bc2, expc * 3);

        auto bc3 = bincount(data, 10);
        EXPECT_EQ(bc3.size(), std::size_t(10));
        EXPECT_EQ(bc3(3), expc(3));
    }
}
