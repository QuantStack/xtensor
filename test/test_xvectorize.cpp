/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xvectorize.hpp"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

namespace xt
{
    double f1(double d1, double d2)
    {
        return d1 + d2;
    }

    using shape_type = dynamic_shape<size_t>;

    TEST(xvectorize, function)
    {
        auto vecf1 = vectorize(f1);
        shape_type shape = {3, 2};
        xarray<double> a(shape, 1.5);
        xarray<double> b(shape, 2.3);
        xarray<double> c = vecf1(a, b);
        EXPECT_EQ(a(0, 0) + b(0, 0), c(0, 0));
    }

    TEST(xvectorize, lambda)
    {
        auto lambda = [](double d1, double d2) { return d1 + d2; };
        auto vec_lambda = vectorize(lambda);
        shape_type shape = {3, 2};
        xarray<double> a(shape, 1.5);
        xarray<double> b(shape, 2.3);
        xarray<double> c = vec_lambda(a, b);
        EXPECT_EQ(a(0, 0) + b(0, 0), c(0, 0));
    }

    struct func
    {
        inline double operator()(double d1, double d2) const
        {
            return d1 + d2;
        }
    };

    TEST(xvectorize, functor)
    {
        func f;
        auto vecfunc = vectorize(f);
        shape_type shape = {3, 2};
        xarray<double> a(shape, 1.5);
        xarray<double> b(shape, 2.3);
        xarray<double> c = vecfunc(a, b);
        EXPECT_EQ(a(0, 0) + b(0, 0), c(0, 0));
    }

    TEST(xvectorize, noargument)
    {
        auto vecfunc = vectorize([] { return double(1.); });
        shape_type shape = {3, 2};
        xarray<double> a(shape, 1.5);
        a = vecfunc();
        EXPECT_EQ(size_t(0), a.dimension());
    }
}
