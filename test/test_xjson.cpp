/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xjson.hpp"

#include <string>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

namespace xt
{
    TEST(xjson, xarray_to_json)
    {
        xt::xarray<double> t =
          {{{1, 2},
            {3, 4}},
           {{1, 2},
            {3, 4}}};

        nlohmann::json jl = t;
        std::string s = jl.dump();
        EXPECT_EQ(s, "[[[1.0,2.0],[3.0,4.0]],[[1.0,2.0],[3.0,4.0]]]");
    }

    TEST(xjson, xarray_from_json)
    {
        nlohmann::json j = "[[[1.0,2.0],[3.0,4.0]],[[1.0,2.0],[3.0,4.0]]]"_json;
        auto arr = j.get<xt::xarray<double>>();
        auto ref = xt::xarray<double>(
          {{{1, 2},
            {3, 4}},
           {{1, 2},
            {3, 4}}});
        EXPECT_TRUE(all(equal(arr, ref)));
    }

    TEST(xjson, xview_from_json)
    {
        xt::xarray<double> arr =
          {{{1, 2},
            {3, 4}},
           {{1, 2},
            {3, 4}}};

        auto v = xt::view(arr, 0);
        auto j = "[[10.0,10.0],[10.0,10.0]]"_json;
        from_json(j, v);

        auto ref = xt::xarray<double>(
          {{{10, 10},
            {10, 10}},
           {{1, 2},
            {3, 4}}});
        EXPECT_TRUE(all(equal(arr, ref)));
    }
}
