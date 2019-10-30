/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xnoalias.hpp"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "test_xsemantic.hpp"

namespace xt
{
    template <class C>
    class xnoalias : public ::testing::Test
    {
    public:

        using storage_type = C;
    };

    using testing_types = ::testing::Types<xarray_dynamic, xtensor_dynamic>;
    TYPED_TEST_CASE(xnoalias, testing_types);

    TYPED_TEST(xnoalias, a_plus_b)
    {
        operation_tester<std::plus<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major + row_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a + tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major + column_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a + tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major + central_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a + tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major + unit_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a + tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(xnoalias, a_minus_b)
    {
        operation_tester<std::minus<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major - row_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a - tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major - column_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a - tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major - central_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a - tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major - unit_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a - tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(xnoalias, a_multiplies_b)
    {
        operation_tester<std::multiplies<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major * row_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a * tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major * column_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a * tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major * central_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a * tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major * unit_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a * tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(xnoalias, a_divides_by_b)
    {
        operation_tester<std::divides<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major / row_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a / tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major / column_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a / tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major / central_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a / tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major / unit_major");
            TypeParam b(tester.ca.shape(), 0);
            noalias(b) = tester.a / tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(xnoalias, a_plus_equal_b)
    {
        operation_tester<std::plus<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major += row_major");
            TypeParam b = tester.a;
            noalias(b) += tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major += column_major");
            TypeParam b = tester.a;
            noalias(b) += tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major += central_major");
            TypeParam b = tester.a;
            noalias(b) += tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major += unit_major");
            TypeParam b = tester.a;
            noalias(b) += tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(xnoalias, a_minus_equal_b)
    {
        operation_tester<std::minus<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major -= row_major");
            TypeParam b = tester.a;
            noalias(b) -= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major -= column_major");
            TypeParam b = tester.a;
            noalias(b) -= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major -= central_major");
            TypeParam b = tester.a;
            noalias(b) -= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major -= unit_major");
            TypeParam b = tester.a;
            noalias(b) -= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(xnoalias, a_times_equal_b)
    {
        operation_tester<std::multiplies<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major *= row_major");
            TypeParam b = tester.a;
            noalias(b) *= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major *= column_major");
            TypeParam b = tester.a;
            noalias(b) *= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major *= central_major");
            TypeParam b = tester.a;
            noalias(b) *= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major *= unit_major");
            TypeParam b = tester.a;
            noalias(b) *= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TYPED_TEST(xnoalias, a_divide_by_equal_b)
    {
        operation_tester<std::divides<>, TypeParam> tester;

        {
            SCOPED_TRACE("row_major /= row_major");
            TypeParam b = tester.a;
            noalias(b) /= tester.ra;
            EXPECT_EQ(tester.res_rr, b);
        }

        {
            SCOPED_TRACE("row_major /= column_major");
            TypeParam b = tester.a;
            noalias(b) /= tester.ca;
            EXPECT_EQ(tester.res_rc, b);
        }

        {
            SCOPED_TRACE("row_major /= central_major");
            TypeParam b = tester.a;
            noalias(b) /= tester.cta;
            EXPECT_EQ(tester.res_rct, b);
        }

        {
            SCOPED_TRACE("row_major /= unit_major");
            TypeParam b = tester.a;
            noalias(b) /= tester.ua;
            EXPECT_EQ(tester.res_ru, b);
        }
    }

    TEST(xnoalias, scalar_ops)
    {
        xarray<int> a = {{1,2,3}, {4,5,6}, {7,8,9}};
        xarray<int> b = {{1,2,3}, {4,5,6}, {7,8,9}};
        xarray<int> c = {{1,2,3}, {4,5,6}, {7,8,9}};

        xscalar<int> one(1), five(5), twelve(12), seven(7), bhalf(0b00001111), bxor(0b01001111);

        xt::noalias(a) += 1;
        b += 1;
        xt::noalias(c) += one;
        EXPECT_EQ(a, b);
        EXPECT_EQ(a, c);

        xt::noalias(a) -= 5;
        b -= 5;
        xt::noalias(c) -= five;
        EXPECT_EQ(a, b);
        EXPECT_EQ(a, c);

        xt::noalias(a) *= 12;
        b *= 12;
        xt::noalias(c) *= twelve;
        EXPECT_EQ(a, b);
        EXPECT_EQ(a, c);

        xt::noalias(a) /= 7;
        b /= 7;
        xt::noalias(c) /= seven;
        EXPECT_EQ(a, b);
        EXPECT_EQ(a, c);

        xt::noalias(a) %= 7;
        b %= 7;
        xt::noalias(c) %= seven;
        EXPECT_EQ(a, b);
        EXPECT_EQ(a, c);

        xt::noalias(a) &= 0b00001111;
        b &= 0b00001111;
        xt::noalias(c) &= bhalf;
        EXPECT_EQ(a, b);
        EXPECT_EQ(a, c);

        xt::noalias(a) |= 0b00001111;
        b |= 0b00001111;
        xt::noalias(c) |= bhalf;
        EXPECT_EQ(a, b);
        EXPECT_EQ(a, c);

        xt::noalias(a) ^= 0b01001111;
        b ^= 0b01001111;
        xt::noalias(c) ^= bxor;
        EXPECT_EQ(a, b);
        EXPECT_EQ(a, c);

        xt::noalias(a) = 123;
        b = 123;

        EXPECT_EQ(a, b);
    }

    TEST(xnoalias, rvalue)
    {
        xarray<int> a = {{1,2,3}, {4,5,6}, {7,8,9}};
        xarray<int> b = {{1,2,3}, {4,5,6}, {7,8,9}};

        xt::noalias(xt::view(a, 1)) += 10;
        xt::view(b, 1) += 10;

        EXPECT_EQ(a, b);

        xt::noalias(xt::view(a, 1)) = 10;
        xt::view(b, 1) = 10;
        EXPECT_EQ(a, b);
    }
}
