/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xindex_view.hpp"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xbroadcast.hpp"
#include "xtensor/xview.hpp"
#include "test_common.hpp"

namespace xt
{
    using std::size_t;

    TEST(xindex_view, indices)
    {
        xarray<double> e = xt::random::rand<double>({3, 3});
        xarray<double> e_copy = e;
        auto v = index_view(e, {{1ul, 1ul}, {1ul, 2ul}, {2ul, 2ul}});
        EXPECT_EQ(v.layout(), layout_type::dynamic);

        using shape_type = typename decltype(v)::shape_type;
        EXPECT_EQ(shape_type{3}, v.shape());

        EXPECT_EQ(e(1, 1), v(0));
        EXPECT_EQ(e(1, 2), v[{1ul}]);

        std::vector<size_t> idx = {2ul};
        EXPECT_EQ(e(2, 2), v.element(idx.begin(), idx.end()));

        v += 3;
        auto expected = e_copy(1, 1) + 3;
        EXPECT_EQ(expected, e(1, 1));

        auto t = v + 3;
		EXPECT_DOUBLE_EQ((e_copy(1, 1) + 6), t(0));
        EXPECT_EQ((e(1, 1) + 3), t(0));

        v = broadcast(123, v.shape());
        EXPECT_EQ(123, e(1, 1));
        EXPECT_EQ(123, e(1, 2));
        EXPECT_EQ(123, e(2, 2));

        xarray<double> as = {3, 3, 3};
        v = as;
        EXPECT_TRUE(all(equal(v, as)));
        EXPECT_EQ(3, e(2, 2));
    }

    TEST(xindex_view, boolean)
    {
        xarray<double> e = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        auto v = filter(e, e > 0);
        EXPECT_EQ(1, v(0));

        v += 2;
        EXPECT_EQ(3, e(1, 1));
        EXPECT_EQ(3, v(1));

        v += xarray<double>{1, 2, 3};
        EXPECT_EQ(5, e(1, 1));
        EXPECT_EQ(6, e(2, 2));

        xarray<double> e2 = random::rand<double>({3, 3, 3, 3});
        auto v2 = filter(e2, e2 > 0.5);
        v2 *= 0;
        EXPECT_TRUE(!any(e2 > 0.5));
    }

    TEST(xindex_view, access)
    {
        xarray<double> e = {{ 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }};
        auto v = filter(e, e > 0);
        EXPECT_EQ(v(), v(0));
        EXPECT_EQ(v(1, 2, 1), v(1));
    }

    TEST(xindex_view, fill)
    {
        xarray<double> e = { { 1, 0, 0 },{ 0, 1, 0 },{ 0, 0, 1 } };
        xarray<double> res = { {1, 2, 2}, {2, 1, 2}, {2, 2, 1} };
        auto v = filter(e, e < 1);
        v.fill(2);
        EXPECT_EQ(e, res);
    }

    TEST(xindex_view, unchecked)
    {
        xarray<double> e = { { 1, 0, 0 },{ 0, 1, 0 },{ 0, 0, 1 } };
        auto v = filter(e, e > 0);
        EXPECT_EQ(v.unchecked(1), v(1));
    }

    TEST(xindex_view, indices_on_function)
    {
        xarray<double> e = xt::random::rand<double>({3, 3});
        auto fn = e * 3 - 120;
        auto v = index_view(fn, {{1ul, 1ul}, {1ul, 2ul}, {2ul, 2ul}});
        EXPECT_EQ(fn(1, 1), v(0));
        EXPECT_EQ(fn(1, 2), v[{1ul}]);

        std::vector<size_t> idx = {2};
        EXPECT_EQ(fn(2, 2), v.element(idx.begin(), idx.end()));

        auto it = v.begin();
        EXPECT_EQ(fn(1, 1), *it);

        EXPECT_EQ(fn(1, 2), *(++it));
        EXPECT_EQ(fn(2, 2), *(++it));
        EXPECT_EQ(++it, v.end());
    }

    TEST(xindex_view, view_on_view)
    {
        xarray<double> e = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        auto v = filter(e, e > 0);
        auto v_on_v = view(v, 1);
        v_on_v(0) = 10;
        EXPECT_EQ(10, e(1, 1));
    }

    TEST(xindex_view, assign_scalar)
    {
        xarray<double> a = {{1, 5, 3}, {4, 5, 6}};
        auto v = filter(a, a >= 5);
        v = 100;
        EXPECT_EQ(100, v(0));
        EXPECT_EQ(100, v(1));
        EXPECT_EQ(100, v(2));
    }

    TEST(xindex_view, filtration)
    {
        xarray<double> a = {{1, 5, 3}, {4, 5, 6}};
        filtration(a, a >= 5) += 2;
        xarray<double> expected = {{1, 7, 3}, {4, 7, 8}};
        EXPECT_EQ(expected, a);
    }

    TEST(xindex_view, filter)
    {
        xarray<double> a = {{ 1, 5, 3 },{ 4, 5, 6 }};
        const xarray<double> b = {{ 1, 5, 3 },{ 4, 5, 6 }};
        filter(a, a > 3) += filter(b, b > 3);
        xarray<double> expected = {{ 1, 10, 3}, {8, 10, 12}};
        EXPECT_EQ(expected, a);
    }

    TEST(xindex_view, filter_column_major)
    {
        xarray<int> a = {{{1, 3}, {2, 4}}, {{5, 7}, {6, 8}}};
        xarray<bool> cond = {{{true, true}, {false, false}}, {{true, true}, {false, false}}};
        
        xarray<int> resc = xt::filter<xt::layout_type::column_major>(a, cond);
        xarray<int> expc = {1, 5 ,3, 7};
        EXPECT_EQ(resc, expc);

        xarray<int> resr = xt::filter(a, cond);
        xarray<int> expr = {1, 3 ,5, 7};
        EXPECT_EQ(resr, expr);
    }

    TEST(xindex_view, const_adapt_filter)
    {
        const std::vector<double> av({1,2,3,4,5,6});
        auto a = xt::adapt(av, std::array<std::size_t, 2>({3, 2}));
        xt::xarray<double> b = {{1, 2, 3}, {4, 5, 6}};
        xt::filter(b, b > 3) += xt::filter(a, a < 4);
        xarray<double> expected = {{1, 2, 3}, {5, 7, 9}};
        EXPECT_EQ(expected, b);
    }
}
