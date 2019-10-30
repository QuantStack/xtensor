/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xmanipulation.hpp"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"
#include "test_common_macros.hpp"

namespace xt
{
    TEST(xmanipulation, transpose_assignment)
    {
        xarray<double> e = xt::arange<double>(24);
        e.resize({2, 2, 6});
        auto vt = transpose(e);

        vt(0, 0, 1) = 123;
        EXPECT_EQ(123, e(1, 0, 0));
        auto val = vt[{1, 0, 1}];
        EXPECT_EQ(e(1, 0, 1), val);
        EXPECT_ANY_THROW(vt.at(10, 10, 10));
        EXPECT_ANY_THROW(vt.at(0, 0, 0, 0));
    }

    TEST(xmanipulation, transpose_layout_swap)
    {
        xarray<double, layout_type::row_major> a = xt::ones<double>({5, 5});

        auto tv = transpose(a);
        EXPECT_EQ(tv.layout(), layout_type::column_major);

        auto tvt = transpose(tv);
        EXPECT_EQ(tvt.layout(), layout_type::row_major);

        xarray<double, layout_type::column_major> b = xt::ones<double>({5, 5, 5});
        auto cbt = transpose(b);
        EXPECT_EQ(cbt.layout(), layout_type::row_major);

        auto cbw1 = transpose(b, {0, 1 ,2});
        auto cbw2 = transpose(b, {2, 1, 0});
        auto cbw3 = transpose(b, {2, 0, 1});
        EXPECT_EQ(cbw1.layout(), layout_type::column_major);
        EXPECT_EQ(cbw2.layout(), layout_type::row_major);
        EXPECT_EQ(cbw3.layout(), layout_type::dynamic);
    }

    TEST(xmanipulation, transpose_function)
    {
        xarray<int, layout_type::row_major> a = { { 0, 1, 2 }, { 3, 4, 5 } };
        xarray<int, layout_type::row_major> b = { { 0, 1, 2 }, { 3, 4, 5 } };
        auto fun = a + b;
        auto tr = transpose(fun);
        EXPECT_EQ(fun(0, 0), tr(0, 0));
        EXPECT_EQ(fun(0, 1), tr(1, 0));
        EXPECT_EQ(fun(0, 2), tr(2, 0));
        EXPECT_EQ(fun(1, 0), tr(0, 1));
        EXPECT_EQ(fun(1, 1), tr(1, 1));
        EXPECT_EQ(fun(1, 2), tr(2, 1));

        xarray<int, layout_type::column_major> a2 = { { 0, 1, 2 }, { 3, 4, 5 } };
        xarray<int, layout_type::column_major> b2 = { { 0, 1, 2 }, { 3, 4, 5 } };
        auto fun2 = a2 + b2;
        auto tr2 = transpose(fun2);
        EXPECT_EQ(fun2(0, 0), tr2(0, 0));
        EXPECT_EQ(fun2(0, 1), tr2(1, 0));
        EXPECT_EQ(fun2(0, 2), tr2(2, 0));
        EXPECT_EQ(fun2(1, 0), tr2(0, 1));
        EXPECT_EQ(fun2(1, 1), tr2(1, 1));
        EXPECT_EQ(fun2(1, 2), tr2(2, 1));
    }

    TEST(xmanipulation, ravel)
    {
        xarray<int, layout_type::row_major> a = { { 0, 1, 2 }, { 3, 4, 5 } };

        auto flat = ravel<layout_type::row_major>(a);
        EXPECT_EQ(flat(0), a(0, 0));
        EXPECT_EQ(flat(1), a(0, 1));
        EXPECT_EQ(flat(2), a(0, 2));
        EXPECT_EQ(flat(3), a(1, 0));
        EXPECT_EQ(flat(4), a(1, 1));
        EXPECT_EQ(flat(5), a(1, 2));

        auto flat_c = ravel<layout_type::column_major>(a);
        EXPECT_EQ(flat_c(0), a(0, 0));
        EXPECT_EQ(flat_c(1), a(1, 0));
        EXPECT_EQ(flat_c(2), a(0, 1));
        EXPECT_EQ(flat_c(3), a(1, 1));
        EXPECT_EQ(flat_c(4), a(0, 2));
        EXPECT_EQ(flat_c(5), a(1, 2));

        auto flat2 = flatten(a);
        EXPECT_EQ(flat, flat2);

        auto flat3 = ravel(a);
        EXPECT_EQ(flat, flat3);
    }
    
    TEST(xmanipulation, flatten)
    {
        xtensor<double, 3> a = linspace<double>(1., 100., 100).reshape({2, 5, 10});
        auto v = view(a, range(0, 2), range(0, 3), range(0, 3));
        xtensor<double, 1> fl = flatten<XTENSOR_DEFAULT_TRAVERSAL>(v);
        xtensor<double, 1> expected_rm = {  1.,  2., 3., 11., 12., 13., 21., 22., 23.,
                                           51., 52., 53, 61., 62., 63., 71., 72., 73. };
        xtensor<double, 1> expected_cm = { 1.,  2.,  3.,  4.,  5.,  6.,
                                          11., 12., 13., 14., 15., 16.,
                                          21., 22., 23., 24., 25., 26.};
        xtensor<double, 1> expected = XTENSOR_DEFAULT_TRAVERSAL==layout_type::row_major ? expected_rm : expected_cm;
        EXPECT_EQ(fl, expected);
        auto v2 = strided_view(a, {range(0, 2), range(0, 3), range(0, 3)});
        xtensor<double, 1> fl2 = flatten<XTENSOR_DEFAULT_TRAVERSAL>(v2);
        EXPECT_EQ(fl2, expected);
    }

    TEST(xmanipulation, split)
    {
        auto b = xt::xarray<double>::from_shape({3, 3, 3});
        using ds = xt::dynamic_shape<std::size_t>;
        std::iota(b.begin(), b.end(), 0);
        auto s1 = split(b, 3);
        EXPECT_EQ(s1.size(), 3u);
        EXPECT_EQ(s1[0].shape(), ds({1, 3, 3}));
        EXPECT_EQ(s1[0](0, 0), b(0, 0, 0));
        EXPECT_EQ(s1[1](0, 0), b(1, 0, 0));
        EXPECT_EQ(s1[2](0, 0), b(2, 0, 0));

        EXPECT_THROW(split(b, 4), std::runtime_error);
        EXPECT_THROW(split(b, 2), std::runtime_error);

        auto s2 = split(b, 1);
        EXPECT_EQ(s2.size(), 1u);
        EXPECT_EQ(s2[0].shape(), ds({3, 3, 3}));

        auto s3 = split(b, 3, 1);
        EXPECT_EQ(s3.size(), std::size_t(3));
        EXPECT_EQ(s3[0].shape(), ds({3, 1, 3}));

        EXPECT_EQ(s3[0](0, 1), b(0, 0, 1));
        EXPECT_EQ(s3[1](0, 1), b(0, 1, 1));
        EXPECT_EQ(s3[2](0, 1), b(0, 2, 1));
    }

    TEST(xmanipulation, squeeze)
    {
        auto b = xt::xarray<double>::from_shape({3, 3, 1, 1, 2, 1, 3});
        std::iota(b.begin(), b.end(), 0);
        using ds = xt::dynamic_shape<std::size_t>;
        auto sq = squeeze(b);

        EXPECT_EQ(sq.shape(), ds({3, 3, 2, 3}));
        EXPECT_EQ(sq(1, 1, 1, 1), b(1, 1, 0, 0, 1, 0, 1));
        EXPECT_THROW(squeeze(b, 1, check_policy::full()), std::runtime_error);
        EXPECT_THROW(squeeze(b, 10, check_policy::full()), std::runtime_error);

        auto sq2 = squeeze(b, {2, 3}, check_policy::full());
        EXPECT_EQ(sq2.shape(), ds({3, 3, 2, 1, 3}));
        EXPECT_EQ(sq2(1, 1, 1, 0, 1), b(1, 1, 0, 0, 1, 0, 1));

        auto sq3 = squeeze(b, 2);
        EXPECT_EQ(sq3.shape(), ds({3, 3, 1, 2, 1, 3}));
        EXPECT_EQ(sq3(2, 2, 0, 1, 0, 2), b(2, 2, 0, 0, 1, 0, 2));
    }

    TEST(xmanipulation, expand_dims)
    {
        auto b = xt::xarray<double>::from_shape({3, 3});
        std::iota(b.begin(), b.end(), 0);
        using ds = xt::dynamic_shape<std::size_t>;
        auto ex = expand_dims(b, 0);
        EXPECT_EQ(ex.shape(), ds({1, 3, 3}));
        auto ex1 = expand_dims(b, 1);
        EXPECT_EQ(ex1.shape(), ds({3, 1, 3}));
        auto ex2 = expand_dims(b, 2);
        EXPECT_EQ(ex2.shape(), ds({3, 3, 1}));

        EXPECT_EQ(ex1(0, 0, 1), b(0, 1));
        EXPECT_EQ(ex1(2, 0, 1), b(2, 1));
    }

    TEST(xmanipulation, atleast_nd)
    {
        xt::xarray<char> d0 = 123;
        auto d1 = xt::xarray<char>::from_shape({3});
        auto d2 = xt::xarray<char>::from_shape({3, 3});
        auto d3 = xt::xarray<char>::from_shape({3, 3, 3});
        auto d5 = xt::xarray<char>::from_shape({3, 3, 3, 3, 3});
        std::iota(d1.begin(), d1.end(), 0);
        std::iota(d2.begin(), d2.end(), 0);
        std::iota(d3.begin(), d3.end(), 0);
        std::iota(d5.begin(), d5.end(), 0);
        using ds = xt::dynamic_shape<std::size_t>;

        auto d3d1 = atleast_3d(d1);
        EXPECT_EQ(d3d1.shape(), ds({1, 3, 1}));
        auto d3d2 = atleast_3d(d2);
        EXPECT_EQ(d3d2.shape(), ds({3, 3, 1}));
        auto d3d3 = atleast_3d(d3);
        EXPECT_EQ(d3d3.shape(), ds({3, 3, 3}));
        auto d3d5 = atleast_3d(d5);
        EXPECT_EQ(d3d5.shape(), ds({3, 3, 3, 3, 3}));
        auto d3d0 = atleast_3d(d0);
        EXPECT_EQ(d3d0.shape(), ds({1, 1, 1}));
        EXPECT_EQ(d3d0(0, 0, 0), 123);
        auto d4d1 = atleast_Nd<4>(d1);
        EXPECT_EQ(d4d1.shape(), ds({1, 3, 1, 1}));
        auto d2d1 = atleast_2d(d1);
        EXPECT_EQ(d2d1.shape(), ds({1, 3}));
    }

    TEST(xmanipulation, trim_zeros)
    {
        using arr_t = xarray<int>;
        arr_t a = {0, 0, 0, 1, 3, 0};
        arr_t b = {0, 0, 0, 0};
        arr_t c = {0, 0, 0, 1};
        arr_t d = {1, 0, 0, 1};

        arr_t ea = {1, 3};
        arr_t ec = {1};
        arr_t ed = {1, 0, 0, 1};

        arr_t eaf = {1, 3, 0};
        arr_t ecf = {1};
        arr_t edf = {1, 0, 0, 1};

        arr_t eab = {0, 0, 0, 1, 3};
        arr_t ecb = {0, 0, 0, 1};
        arr_t edb = {1, 0, 0, 1};

        EXPECT_EQ(trim_zeros(a), ea);
        EXPECT_EQ(trim_zeros(b).size(), std::size_t(0));
        EXPECT_EQ(trim_zeros(c), ec);
        EXPECT_EQ(trim_zeros(d), ed);

        EXPECT_EQ(trim_zeros(a, "f"), eaf);
        EXPECT_EQ(trim_zeros(b, "f").size(), std::size_t(0));
        EXPECT_EQ(trim_zeros(c, "f"), ecf);
        EXPECT_EQ(trim_zeros(d, "f"), edf);

        EXPECT_EQ(trim_zeros(a, "b"), eab);
        EXPECT_EQ(trim_zeros(b, "b").size(), std::size_t(0));
        EXPECT_EQ(trim_zeros(c, "b"), ecb);
        EXPECT_EQ(trim_zeros(d, "b"), edb);
    }

    TEST(xmanipulation, flipud)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> t = xt::flip(e, 0);
        xarray<double> expected = {{7, 8, 9}, {4, 5, 6}, {1, 2, 3}};
        ASSERT_EQ(expected, t);

        xindex idx = {0, 0};
        ASSERT_EQ(7, t[idx]);
        ASSERT_EQ(2, t(2, 1));
        ASSERT_EQ(7, t.element(idx.begin(), idx.end()));

        xarray<double> f = {{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}};

        xarray<double> ft = xt::flip(f, 0);
        xarray<double> expected_2 = {{{6, 7, 8},
        {9, 10, 11}},
        {{0, 1, 2},
        {3, 4, 5}}};
        ASSERT_EQ(expected_2, ft);
    }

    TEST(xmanipulation, fliplr)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> t = xt::flip(e, 1);
        xarray<double> expected = {{3, 2, 1}, {6, 5, 4}, {9, 8, 7}};
        ASSERT_EQ(expected, t);

        xindex idx = {0, 0};
        ASSERT_EQ(3, t[idx]);
        ASSERT_EQ(8, t(2, 1));
        ASSERT_EQ(3, t.element(idx.begin(), idx.end()));

        xarray<double> f = {{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}};

        xarray<double> ft = xt::flip(f, 1);
        xarray<double> expected_2 = {
            {{3, 4, 5},
            {0, 1, 2}},
            {{9, 10, 11},
            {6, 7, 8}}};

        ASSERT_EQ(expected_2, ft);
    }

    TEST(xmanipulation, rot90)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> e2 = {{1, 2}, {3, 4}};
        xarray<double> e3 = {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};

        std::array<std::ptrdiff_t, 2> axes = {0, 0};
        ASSERT_ANY_THROW(xt::rot90(e, axes));
        axes = {1, 1};
        ASSERT_ANY_THROW(xt::rot90(e, axes));
        axes = {0, 2};
        ASSERT_ANY_THROW(xt::rot90(e, axes));
        axes = {56, 58};
        ASSERT_ANY_THROW(xt::rot90(e, axes));

        ASSERT_EQ(e, xt::rot90<0>(e));
        ASSERT_EQ(e, xt::rot90<4>(e));
        ASSERT_EQ(e, xt::rot90<-4>(e));

        xarray<double> expected2 = {{2, 4}, {1, 3}};
        ASSERT_EQ(expected2, xt::rot90(e2));
        ASSERT_EQ(expected2, xt::rot90(e2, {-2, -1}));

        xarray<double> expected3 = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        ASSERT_EQ(expected3, xt::rot90<2>(e));

        xarray<double> expected4 = {{3, 1}, {4, 2}};
        ASSERT_EQ(expected4, xt::rot90<3>(e2));
        ASSERT_EQ(expected4, xt::rot90<-1>(e2));

        xarray<double> expected5 = {{{1, 3}, {0, 2}}, {{5, 7}, {4, 6}}};
        ASSERT_EQ(expected5, xt::rot90(e3, {1, 2}));
    }
}
