/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xexpression_holder.hpp"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

namespace xt
{
    TEST(xexpression_holder, ctor)
    {
        xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        xarray<double> b = {{3,2,1}, {5,6,7}};
        xarray<double> c = {{56,5,2}, {4,2,6}};

        xexpression_holder holder_a = xexpression_holder(a);
        xexpression_holder holder_b(b);
        xexpression_holder holder_c(std::move(xexpression_holder(c)));
    }

    TEST(xexpression_holder, assign)
    {
        xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        xarray<double> b = {{3,2,1}, {5,6,7}};
        xarray<double> c = {{56,5,2}, {4,2,6}};

        xexpression_holder holder_a = xexpression_holder(a);
        xexpression_holder holder_b(b);

        holder_a = holder_b;
        holder_b = xexpression_holder(c);
    }

    TEST(xexpression_holder, to_json)
    {
        xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        xexpression_holder holder_a = xexpression_holder(a);

        nlohmann::json json_out;
        to_json(json_out, holder_a);

        ASSERT_EQ(json_out[0][0], 1);
        ASSERT_EQ(json_out[0][1], 2);
        ASSERT_EQ(json_out[0][2], 3);
        ASSERT_EQ(json_out[1][1], 6);
    }

    TEST(xexpression_holder, from_json)
    {
        xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        xarray<double> b = {{5,6,7,8}, {1,2,3,4}};

        xexpression_holder holder_a = xexpression_holder(a);

        nlohmann::json json_b;
        to_json(json_b, b);
        from_json(json_b, holder_a);

        ASSERT_EQ(a, b);
    }
}
