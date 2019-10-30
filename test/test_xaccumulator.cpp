/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xaccumulator.hpp"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xfixed.hpp"

namespace xt
{
    TEST(xaccumulator, one_d)
    {
        xt::xarray<short> a = { short(1), short(2), short(3), short(4)};
        xt::xarray<long> expected = { 1, 3, 6, 10};
        auto no_axis = cumsum(a);
        auto with_axis = cumsum(a, 0);
        bool promotion_works = std::is_same<decltype(no_axis)::value_type, long long>::value;
        EXPECT_TRUE(promotion_works);
        EXPECT_TRUE(all(equal(no_axis, expected)));

        EXPECT_TRUE(all(equal(with_axis, expected)));
    }

    TEST(xaccumulator, dim_one)
    {
        xt::xarray<double> arr = {{ 5., 6., 7. }};
        xt::xarray<double> res = xt::cumsum(arr, 0);
        EXPECT_EQ(res, arr);

        xt::xarray<double> arr2 = xt::transpose(arr);
        xt::xarray<double> res2 = xt::cumsum(arr2, 1);
        EXPECT_EQ(res2, arr2);
    }

    TEST(xaccumulator, four_d)
    {
        xarray<double> arg_0 = {{{{ 0., 1., 2.},
                                  { 3., 4., 5.}},

                                 {{ 6., 7., 8.},
                                  { 9.,10.,11.}},

                                 {{12.,13.,14.},
                                  {15.,16.,17.}}},


                                {{{18.,19.,20.},
                                  {21.,22.,23.}},

                                 {{24.,25.,26.},
                                  {27.,28.,29.}},

                                 {{30.,31.,32.},
                                  {33.,34.,35.}}}};
        auto res = cumsum(arg_0);
        xarray<double> expected = {  0.,  1.,  3.,  6., 10., 15., 21., 28., 36., 45., 55., 66., 78., 91.,
                                   105.,120.,136.,153.,171.,190.,210.,231.,253.,276.,300.,325.,351.,378.,
                                   406.,435.,465.,496.,528.,561.,595.,630.};

        xarray<double> expected_col = {   0.,   18.,   24.,   48.,   60.,   90.,   93.,  114.,  123.,  150.,  165.,  198.,
                                        199.,  218.,  225.,  250.,  263.,  294.,  298.,  320.,  330.,  358.,  374.,  408.,
                                        410.,  430.,  438.,  464.,  478.,  510.,  515.,  538.,  549.,  578.,  595.,  630.};
        if (XTENSOR_DEFAULT_TRAVERSAL == layout_type::row_major)
        {
            EXPECT_TRUE(allclose(expected, res));
        }
        else
        {
            EXPECT_TRUE(allclose(expected_col, res));
        }

        auto res_0 = cumsum(arg_0, 0);
        xarray<double> expected_0 = {{{{ 0., 1., 2.},
                                       { 3., 4., 5.}},

                                      {{ 6., 7., 8.},
                                       { 9.,10.,11.}},

                                      {{12.,13.,14.},
                                       {15.,16.,17.}}},


                                     {{{18.,20.,22.},
                                       {24.,26.,28.}},

                                      {{30.,32.,34.},
                                       {36.,38.,40.}},

                                      {{42.,44.,46.},
                                       {48.,50.,52.}}}};

        EXPECT_TRUE(all(equal(expected_0, res_0)));

        auto res_1 = cumsum(arg_0, 1);
        xarray<double> expected_1 = {{{{ 0., 1., 2.},
                                       { 3., 4., 5.}},

                                      {{ 6., 8.,10.},
                                       {12.,14.,16.}},

                                      {{18.,21.,24.},
                                       {27.,30.,33.}}},


                                     {{{18.,19.,20.},
                                       {21.,22.,23.}},

                                      {{42.,44.,46.},
                                       {48.,50.,52.}},

                                      {{72.,75.,78.},
                                       {81.,84.,87.}}}};
        EXPECT_TRUE(all(equal(res_1, expected_1)));

        auto res_2 = cumsum(arg_0, 2);
        xarray<double> expected_2 = {{{{ 0., 1., 2.},
                                      { 3., 5., 7.}},

                                     {{ 6., 7., 8.},
                                      {15.,17.,19.}},

                                     {{12.,13.,14.},
                                      {27.,29.,31.}}},


                                    {{{18.,19.,20.},
                                      {39.,41.,43.}},

                                     {{24.,25.,26.},
                                      {51.,53.,55.}},

                                     {{30.,31.,32.},
                                      {63.,65.,67.}}}};

        EXPECT_TRUE(all(equal(res_2, expected_2)));

        auto res_3 = cumsum(arg_0, 3);
        auto res_m1 = cumsum(arg_0, -1);
        xarray<double> expected_3 = {{{{  0.,  1.,  3.},
                                       {  3.,  7., 12.}},

                                      {{  6., 13., 21.},
                                       {  9., 19., 30.}},

                                      {{ 12., 25., 39.},
                                       { 15., 31., 48.}}},


                                     {{{ 18., 37., 57.},
                                       { 21., 43., 66.}},

                                      {{ 24., 49., 75.},
                                       { 27., 55., 84.}},

                                      {{ 30., 61., 93.},
                                       { 33., 67.,102.}}}};
        EXPECT_TRUE(allclose(expected_3, res_3));
        EXPECT_TRUE(allclose(expected_3, res_m1));
    }

    TEST(xaccumulator, xtensor)
    {
        xtensor<double, 2> arr = {{1, 2, 3}, {4, 5, 6}};
        auto res = xt::cumsum(arr, 0);
        bool type_eq = std::is_same<xtensor<double, 2>, decltype(res)>::value;
        EXPECT_TRUE(type_eq);
        xtensor<double, 2> expected = {{1, 2, 3}, {5, 7, 9}};
        EXPECT_EQ(expected, res);
    }

    TEST(xaccumulator, cumprod)
    {
        xarray<long> arg_0 = {{ 0, 1, 2},
                              { 3, 4, 5},
                              { 6, 7, 8},
                              { 9,10,11}};
        auto res = cumprod(arg_0);

        xarray<long> expected = {0,0,0,0,0,0,0,0,0,0,0,0};
        EXPECT_TRUE(allclose(expected, res));

        auto res_0 = cumprod(arg_0, 0);
        xarray<long> expected_0 = {{  0,   1,   2},
                                   {  0,   4,  10},
                                   {  0,  28,  80},
                                   {  0, 280, 880}};
        EXPECT_TRUE(allclose(expected_0, res_0));

        auto res_1 = cumprod(arg_0, 1);
        xarray<long> expected_1 = {{  0,  0,   0},
                                   {  3, 12,  60},
                                   {  6, 42, 336},
                                   {  9, 90, 990}};
        EXPECT_TRUE(allclose(expected_1, res_1));
    }

    TEST(xaccumulator, xfixed)
    {
        xtensor_fixed<float, xshape<2, 4, 3>> a = xt::random::rand<float>({2, 4, 3});
        auto res = cumsum(a, 1);

        bool truth = std::is_same<decltype(res), xtensor_fixed<double, xshape<2, 4, 3>>>::value;
        EXPECT_TRUE(truth);
        xtensor_fixed<long, xshape<4, 3>> arg_0({{ 0, 1, 2},
                                                 { 3, 4, 5},
                                                 { 6, 7, 8},
                                                 { 9,10,11}});
        auto res_0 = cumprod(arg_0, 0);
        xarray<long> expected_0 = {{  0,   1,   2},
                                   {  0,   4,  10},
                                   {  0,  28,  80},
                                   {  0, 280, 880}};
        EXPECT_TRUE(expected_0 == res_0);
        truth = std::is_same<typename decltype(res_0)::shape_type, xshape<4, 3>>::value;
        EXPECT_TRUE(truth);
    }

    TEST(xaccumulator, empty_array)
    {
        xt::xarray<double> a = xt::ones<double>({ 3, 4, 0 });

        auto result0 = xt::cumsum(a);
        auto expected0 = xt::xarray<double>::from_shape({ 0 });
        EXPECT_EQ(result0, expected0);

        auto result1 = xt::cumsum(a, 2);
        auto expected = xt::xarray<double>::from_shape({ 3, 4, 0 });
        EXPECT_EQ(result1, expected);

        auto result2 = xt::cumsum(a, 1);
        EXPECT_EQ(result2, expected);
    }
}
