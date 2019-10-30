/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xsort.hpp"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xinfo.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xslice.hpp"

namespace xt
{

    TEST(xsort, argsort)
    {
        xarray<double> a1 = {2, 3, 1};
        xarray<double, xt::layout_type::column_major> a2_c = {{2, 3, 1}, {4, 6, 5}};
        xarray<double, xt::layout_type::row_major> a2_r = {{2, 3, 1}, {4, 6, 5}};
        xarray<float> a3 = {{{1,3,2},
                             {4,2,1}},
                            {{5,1,3},
                             {4,2,6}}};

        xarray<std::size_t> ex = {2, 0, 1};
        EXPECT_EQ(ex, argsort(a1, 0));
        EXPECT_EQ(ex, argsort(a1));

        xarray<std::size_t> ex2_0 = {{0, 0, 0}, {1, 1, 1}};
        xarray<std::size_t> ex2_1 = {{2, 0, 1}, {0, 2, 1}};

        EXPECT_EQ(ex2_1, argsort(a2_c));
        EXPECT_EQ(ex2_1, argsort(a2_r));
        EXPECT_EQ(ex2_0, argsort(a2_c, 0));
        EXPECT_EQ(ex2_0, argsort(a2_r, 0));
        EXPECT_EQ(ex2_1, argsort(a2_c, 1));
        EXPECT_EQ(ex2_1, argsort(a2_r, 1));

        xarray<std::size_t> ex3_0 = {{{0,1,0},{0,0,0}},{{1,0,1},{1,1,1}}};
        xarray<std::size_t> ex3_1 = {{{0,1,1},{1,0,0}},{{1,0,0},{0,1,1}}};
        xarray<std::size_t> ex3_2 = {{{0,2,1},{2,1,0}},{{1,2,0},{1,0,2}}};

        EXPECT_EQ(ex3_2, argsort(a3));
        EXPECT_EQ(ex3_0, argsort(a3, 0));
        EXPECT_EQ(ex3_1, argsort(a3, 1));
        EXPECT_EQ(ex3_2, argsort(a3, 2));

        xtensor<double, 1> t1 = {2, 3, 1};
        xtensor<std::size_t, 1> ex4 = {2, 0, 1};
        EXPECT_EQ(ex4, argsort(t1, 0));
        EXPECT_EQ(ex4, argsort(t1));

        if (XTENSOR_DEFAULT_TRAVERSAL == layout_type::row_major)
        {
            xtensor_fixed<double, xt::xshape<2, 3>> tf1 = a2_r;
            EXPECT_EQ(ex2_1, argsort(tf1));
        }
        else
        {
            xtensor_fixed<double, xt::xshape<2, 3>> tf1 = a2_c;
            EXPECT_EQ(ex2_1, argsort(tf1));
        }
    }

    TEST(xsort, flatten_argsort)
    {
        {
            xarray<int, layout_type::row_major> a = {{1, 2, 3}, {4, 5, 6}};
            xarray<std::size_t, layout_type::row_major> res = argsort(a, placeholders::xtuph());
            xarray<std::size_t, layout_type::row_major> expected = {0, 1, 2, 3, 4, 5};
            EXPECT_EQ(res, expected);
        }
        {
            xarray<int, layout_type::column_major> a = {{1, 2, 3}, {4, 5, 6}};
            xarray<std::size_t, layout_type::column_major> res = argsort(a, placeholders::xtuph());
            xarray<std::size_t, layout_type::column_major> expected = {0, 1, 2, 3, 4, 5};
            EXPECT_EQ(res, expected);
        }
    }

    TEST(xsort, sort_easy)
    {
        xarray<double> a = {{5, 3, 1}, {4, 4, 4}};

        xarray<double> ex = {1, 3, 4, 4, 4, 5};
        EXPECT_EQ(ex, sort(a, xnone()));

        auto n_ex = sort(ex);
        EXPECT_EQ(n_ex, ex);

        auto n_ex_0 = sort(ex, 0);
        EXPECT_EQ(n_ex, ex);

        xarray<double> ex_2 = {{4, 3, 1}, {5, 4, 4}};
        EXPECT_EQ(ex_2, sort(a, 0));

        xarray<double> ex_3 = {{1, 3, 5}, {4, 4, 4}};
        EXPECT_EQ(ex_3, sort(a, 1));
    }

    TEST(xsort, fixed)
    {
        xtensor_fixed<double, xshape<4, 3>> a = {{5, 3, 1}, {4, 4, 4}, {5, 9, 1}, {2, 4, 2}};
        xarray<double> b = {{5, 3, 1}, {4, 4, 4}, {5, 9, 1}, {2, 4, 2}};

        EXPECT_EQ(sort(a, xnone()), sort(b, xnone()));
        EXPECT_EQ(sort(a), sort(b));
        EXPECT_EQ(sort(a, 1), sort(b, 1));
        EXPECT_EQ(sort(a, 0), sort(b, 0));
    }

    TEST(xsort, argmin)
    {
        xarray<double> a = {{5, 3, 1}, {4, 4, 4}};
        xarray<double> b = {1,3,4,-100};
        xarray<double, layout_type(int(XTENSOR_DEFAULT_LAYOUT) & 0x03)> ar = {{5, 3, 1}, {4, 4, 4}};

        xarray<std::size_t> ex = 2ul;

        EXPECT_EQ(ex, argmin(a));

        EXPECT_EQ(size_t(3), argmin(b)());
        EXPECT_EQ(size_t(3), argmin(b, 0)());

        xarray<std::size_t> ex_2 = {1, 0, 0};
        EXPECT_EQ(ex_2, argmin(a, 0));

        xarray<std::size_t> ex_3 = {2, 0};
        EXPECT_EQ(ex_3, argmin(a, 1));

        xtensor<double, 2> xa = {{5,3,1}, {4,4,4}};
        EXPECT_EQ(ex, argmin(xa));
        EXPECT_EQ(ex_2, argmin(xa, 0));
        EXPECT_EQ(ex_3, argmin(xa, 1));
    }

    TEST(xsort, argmax)
    {
        xarray<double> a = {{5, 3, 1}, {4, 4, 4}};

        EXPECT_EQ(0ul, argmax(a)());

        xarray<std::size_t> ex_2 = {0, 1, 1};
        EXPECT_EQ(ex_2, argmax(a, 0));

        xarray<std::size_t> ex_3 = {0, 0};
        EXPECT_EQ(ex_3, argmax(a, 1));

        xtensor<int, 2> b = {{ 1,2 }};
        auto res = xt::eval(xt::argmax(b, 1));
        EXPECT_EQ(res(), 1u);

        xtensor<int, 3> c = {{{1, 2, 3, 4}},
                             {{4, 3, 2, 1}}};
        xtensor<std::size_t, 2> ex_4 = {{3}, {0}};
        EXPECT_EQ(ex_4, argmax(c, 2));

        xtensor<std::size_t, 2> ex_5 = {{1, 1, 0, 0}};
        EXPECT_EQ(ex_5, argmax(c, 0));

        xtensor<std::size_t, 2> ex_6 = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        EXPECT_EQ(ex_6, argmax(c, 1));
    }

    TEST(xsort, sort_large_prob)
    {
        for (std::size_t i = 0; i < 20; ++i)
        {
            xarray<double> a = xt::random::rand<double>({5, 5, 100, 10});

            auto a_s0 = sort(a, 0);
            auto va_s0 = view(a_s0, xt::all(), 3, 45, 3);
            EXPECT_TRUE(std::is_sorted(va_s0.begin(), va_s0.end()));

            auto a_s1 = sort(a, 1);
            auto va_s1 = view(a_s1, 0, xt::all(), 0, 0);
            EXPECT_TRUE(std::is_sorted(va_s1.begin(), va_s1.end()));

            auto a_s2 = sort(a, 2);
            auto va_s2 = view(a_s2, 3, 1, xt::all(), 3);
            EXPECT_TRUE(std::is_sorted(va_s2.begin(), va_s2.end()));

            auto a_s3 = sort(a, 3);
            auto va_s3 = view(a_s3, 4, 3, 45, xt::all());
            EXPECT_TRUE(std::is_sorted(va_s3.begin(), va_s3.end()));
        }
    }

    TEST(xsort, argmax_prob)
    {
        for (std::size_t i = 0; i < 20; ++i)
        {
            xarray<double> a = xt::random::rand<double>({5, 4, 6, 7});

            auto a_s0 = argmin<XTENSOR_DEFAULT_LAYOUT>(a, 0);
            auto va_s0 = view(a, xt::all(), 3, 2, 3);
            auto m0_idx = a_s0(3, 2, 3);
            auto it0 = std::min_element(va_s0.begin(), va_s0.end());
            auto c0_idx = std::distance(va_s0.begin(), it0);
            EXPECT_EQ(static_cast<size_t>(c0_idx), m0_idx);

            auto a_s1 = argmin<XTENSOR_DEFAULT_LAYOUT>(a, 1);
            auto va_s1 = view(a, 3, xt::all(), 2, 3);
            auto m1_idx = a_s1(3, 2, 3);
            auto it1 = std::min_element(va_s1.begin(), va_s1.end());
            auto c1_idx = std::distance(va_s1.begin(), it1);
            EXPECT_EQ(static_cast<size_t>(c1_idx), m1_idx);

            auto a_s2 = argmin<XTENSOR_DEFAULT_LAYOUT>(a, 2);
            auto va_s2 = view(a, 3, 2, xt::all(), 3);
            auto m2_idx = a_s2(3, 2, 3);
            auto it2 = std::min_element(va_s2.begin(), va_s2.end());
            auto c2_idx = std::distance(va_s2.begin(), it2);
            EXPECT_EQ(static_cast<size_t>(c2_idx), m2_idx);

            auto a_s3 = argmin<XTENSOR_DEFAULT_LAYOUT>(a, 3);
            auto va_s3 = view(a, 3, 2, 3, xt::all());
            auto m3_idx = a_s3(3, 2, 3);
            auto it3 = std::min_element(va_s3.begin(), va_s3.end());
            auto c3_idx = std::distance(va_s3.begin(), it3);
            EXPECT_EQ(static_cast<size_t>(c3_idx), m3_idx);
        }
    }

    TEST(xsort, unique)
    {
        xarray<double> a = {1,2,3, 5,3,2,1,2,2,2,2,2,2, 45};
        xarray<double> ax = {1, 2, 3, 5, 45};
        EXPECT_EQ(unique(a), ax);

        xarray<double> b = {{1,2,3}, {4,5,6}, {7,8,9}};
        xarray<double> bx = {1,2,3,4,5,6,7,8,9};
        EXPECT_EQ(unique(b), bx);

        xarray<double> bb = {{1,2,3}, {7,8,9}, {4,5,6}, {7,8,9}};
        xarray<double> bbx = {1,2,3,4,5,6,7,8,9};
        EXPECT_EQ(unique(bb), bbx);

        auto c = view(b, range(0, 3), range(0, 3));
        auto d = unique(c);
        EXPECT_EQ(d, bx);

        auto e = xt::unique(xt::where(xt::greater(b,2), 1, 0)); 
        xarray<double> ex = {0, 1};
        EXPECT_EQ(e, ex);
    }

    TEST(xsort, setdiff1d)
    {
        {
            xarray<size_t> ar1 = {1,2,3,4,4,4,5};
            xarray<size_t> ar2 = {4,5};
            xarray<size_t> out = {1,2,3};
            EXPECT_EQ(setdiff1d(ar1, ar2), out);
        }

        {
            xarray<size_t> ar1 = {{5,6,7},{4,4,4},{1,2,3}};
            xarray<size_t> ar2 = {4,1};
            xarray<size_t> out = {2,3,5,6,7};
            EXPECT_EQ(setdiff1d(ar1, ar2), out);
        }
    }

    template <class T>
    bool check_partition(T& arr, std::size_t pos)
    {
        bool res = true;
        for (std::size_t i = 0; i < pos; ++i)
        {
            res = res && arr(i) < arr(pos);
        }
        for (std::size_t i = pos; i < arr.size(); ++i)
        {
            res = res && arr(i) >= arr(pos);
        }
        return res;
    }

    TEST(xsort, partition)
    {
        xt::xarray<int> a = {3, 4, 2, 1};
        auto r1 = xt::partition(a, 2);
        EXPECT_TRUE(check_partition(r1, 2));
    }

    template <class T, class U>
    bool check_argpartition(T& arr, U& idxs, std::size_t pos)
    {
        bool res = true;
        for (std::size_t i = 0; i < pos; ++i)
        {
            res = res && arr(idxs[i]) < arr(idxs[pos]);
        }
        for (std::size_t i = pos; i < arr.size(); ++i)
        {
            res = res && arr(idxs[i]) >= arr(idxs[pos]);
        }
        return res;
    }

    TEST(xsort, argpartition)
    {
        xt::xarray<int> a = {3, 4, 2, 1};
        auto r1 = xt::argpartition(a, 2);
        EXPECT_TRUE(check_argpartition(a, r1, 2));
    }

    TEST(xsort, median)
    {
        xt::xtensor<float, 2> a = {{ 3, 4, 2, 1},
                                   { 1, 1, 3, 2},
                                   { 9, 9, 9, 9},
                                   {12,12,12,12},
                                   { 5, 5, 5, 5}};
        auto mall = median(a);
        auto ma0 = median(a, 0);
        auto ma1 = median(a, 1);

        EXPECT_EQ(mall, 5.f);

        xt::xtensor<float, 1> ma0_exp = {5, 5, 5, 5};
        xt::xtensor<float, 1> ma1_exp = {2.5, 1.5, 9., 12., 5.};
        EXPECT_EQ(ma0, ma0_exp);
        EXPECT_EQ(ma1, ma1_exp);
    }
}
