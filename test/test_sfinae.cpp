/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <complex>
#include <limits>

#include "gtest/gtest.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xview.hpp"

namespace xt
{
    template <class E, std::enable_if_t<!xt::has_rank_t<E, 2>::value, int> = 0>
    inline E sfinae_rank_basic_func(E&& a)
    {
        E b = a;
        b.fill(0);
        return b;
    }

    template <class E, std::enable_if_t<xt::has_rank_t<E, 2>::value, int> = 0>
    inline E sfinae_rank_basic_func(E&& a)
    {
        E b = a;
        b.fill(2);
        return b;
    }

    TEST(sfinae, rank_basic)
    {
        xt::xarray<size_t> a = {{9, 9, 9}, {9, 9, 9}};
        xt::xtensor<size_t, 1> b = {9, 9};
        xt::xtensor<size_t, 2> c = {{9, 9}, {9, 9}};
        xt::xtensor_fixed<size_t, xt::xshape<2, 2>> d = {{9, 9}, {9, 9}};
        auto v = xt::view(c, 0, xt::all());

        EXPECT_TRUE(xt::all(xt::equal(sfinae_rank_basic_func(a), 0ul)));
        EXPECT_TRUE(xt::all(xt::equal(sfinae_rank_basic_func(b), 0ul)));
        EXPECT_TRUE(xt::all(xt::equal(sfinae_rank_basic_func(c), 2ul)));
        EXPECT_TRUE(xt::all(xt::equal(sfinae_rank_basic_func(d), 2ul)));
        EXPECT_TRUE(xt::all(xt::equal(sfinae_rank_basic_func(v), 0ul)));
    }

    template <class E, std::enable_if_t<xt::has_rank_t<E, SIZE_MAX>::value, int> = 0>
    inline E sfinae_rank_func(E&& a)
    {
        E b = a;
        b.fill(0);
        return b;
    }

    template <class E, std::enable_if_t<xt::has_rank_t<E, 1>::value, int> = 0>
    inline E sfinae_rank_func(E&& a)
    {
        E b = a;
        b.fill(1);
        return b;
    }

    template <class E, std::enable_if_t<xt::has_rank_t<E, 2>::value, int> = 0>
    inline E sfinae_rank_func(E&& a)
    {
        E b = a;
        b.fill(2);
        return b;
    }

    TEST(sfinae, rank)
    {
        xt::xarray<size_t> a = {{9, 9, 9}, {9, 9, 9}};
        xt::xtensor<size_t, 1> b = {9, 9};
        xt::xtensor<size_t, 2> c = {{9, 9}, {9, 9}};
        xt::xtensor_fixed<size_t, xt::xshape<2, 2>> d = {{9, 9}, {9, 9}};
        auto v = xt::view(c, 0, xt::all());

        EXPECT_TRUE(xt::all(xt::equal(sfinae_rank_func(a), 0ul)));
        EXPECT_TRUE(xt::all(xt::equal(sfinae_rank_func(b), 1ul)));
        EXPECT_TRUE(xt::all(xt::equal(sfinae_rank_func(c), 2ul)));
        EXPECT_TRUE(xt::all(xt::equal(sfinae_rank_func(d), 2ul)));
        EXPECT_TRUE(xt::all(xt::equal(sfinae_rank_func(v), 0ul)));
    }
}
