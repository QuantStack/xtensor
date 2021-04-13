/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#if defined(_MSC_VER) && !defined(__clang__)
#define VS_SKIP_XFIXED 1
#endif

// xfixed leads to ICE in debug mode, this provides
// an easy way to prevent compilation
#ifndef VS_SKIP_XFIXED

#include "gtest/gtest.h"
#include "test_common_macros.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xmanipulation.hpp"

// On VS2015, when compiling in x86 mode, alignas(T) leads to C2718
// when used for a function parameter, even indirectly. This means that
// we cannot pass parameters whose class is declared with alignas specifier
// or any type wrapping or inheriting from such a type.
// The xtensor_fixed class internally uses aligned_array which is declared as
// alignas(something_different_from_0), hence the workaround.
#if _MSC_VER < 1910 && !_WIN64
#define VS_X86_WORKAROUND 1
#endif

// test_fixed removed from MSVC x86 because of recurring ICE.
// Will be enabled again when the compiler is fixed

#if (_MSC_VER < 1910 && _WIN64) || (_MSC_VER >= 1910 && !defined(DISABLE_VS2017)) || !defined(_MSC_VER)

namespace xt
{
    using xtensorf3x4 = xtensor_fixed<double, xt::xshape<3, 4>>;
    using xtensorf4 = xtensor_fixed<double, xt::xshape<4>>;

    TEST(xtensor_fixed, basic)
    {
        xtensorf3x4 a({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
        xtensorf3x4 b({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});

        xtensorf3x4 res1 = a + b;
        xtensorf3x4 res2 = 2.0 * a;

#ifndef VS_X86_WORKAROUND
        EXPECT_EQ(res1, res2);
#else
        bool res = std::equal(res1.cbegin(), res1.cend(), res2.cbegin());
        EXPECT_TRUE(res);
#endif
    }

    TEST(xtensor_fixed, broadcast)
    {
        xtensorf3x4 a({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
        xtensorf4 b({4, 5, 6, 7});

        xtensorf3x4 res = a * b;
        xtensorf3x4 resb = b * a;

        xarray<double> ax = a;
        xarray<double> bx = b;
        xarray<double> arx = a * b;
        xarray<double> brx = b * a;

#ifndef VS_X86_WORKAROUND
        EXPECT_EQ(res, arx);
        EXPECT_EQ(resb, brx);
#else
        bool bresa = std::equal(res.cbegin(), res.cend(), arx.cbegin());
        EXPECT_TRUE(bresa);
        bool bresb = std::equal(resb.cbegin(), resb.cend(), brx.cbegin());
        EXPECT_TRUE(bresb);
#endif

#ifdef XTENSOR_ENABLE_ASSERT
        XT_EXPECT_THROW(a.resize({2, 2}), std::runtime_error);
#endif
        // reshaping fixed container
        XT_EXPECT_THROW(a.reshape({{1, 9}}), std::runtime_error);
        XT_EXPECT_NO_THROW(a.reshape({3, 4}));
        XT_EXPECT_NO_THROW(a.reshape({3, 4}, XTENSOR_DEFAULT_LAYOUT));
        XT_EXPECT_THROW(a.reshape({3, 4}, layout_type::any), std::runtime_error);
    }

    TEST(xtensor_fixed, strides)
    {
        xtensor_fixed<double, xshape<3, 7, 2, 5, 3>, layout_type::row_major> arm;
        xtensor<double, 5, layout_type::row_major> brm = xtensor<double, 5, layout_type::row_major>::from_shape({3, 7, 2, 5, 3});

        EXPECT_TRUE(std::equal(arm.strides().begin(), arm.strides().end(), brm.strides().begin()));
        EXPECT_EQ(arm.strides().size(), brm.strides().size());
        EXPECT_TRUE(std::equal(arm.backstrides().begin(), arm.backstrides().end(), brm.backstrides().begin()));
        EXPECT_EQ(arm.backstrides().size(), brm.backstrides().size());
        EXPECT_EQ(arm.size(), std::size_t(3 * 7 * 2 * 5 * 3));

        xtensor_fixed<double, xshape<3, 7, 2, 5, 3>, layout_type::column_major> acm;
        xtensor<double, 5, layout_type::column_major> bcm = xtensor<double, 5, layout_type::column_major>::from_shape({3, 7, 2, 5, 3});

        EXPECT_TRUE(std::equal(acm.strides().begin(), acm.strides().end(), bcm.strides().begin()));
        EXPECT_EQ(acm.strides().size(), bcm.strides().size());
        EXPECT_TRUE(std::equal(acm.backstrides().begin(), acm.backstrides().end(), bcm.backstrides().begin()));
        EXPECT_EQ(acm.backstrides().size(), bcm.backstrides().size());
        EXPECT_EQ(acm.size(), std::size_t(3 * 7 * 2 * 5 * 3));

        auto s = get_strides<layout_type::row_major, const_array<ptrdiff_t, 3>>(xshape<3, 4, 5>());
        EXPECT_EQ(s[0], 20u);
        EXPECT_EQ(s[1], 5u);
        EXPECT_EQ(s[2], 1u);

        auto sc = get_strides<layout_type::column_major, const_array<ptrdiff_t, 3>>(xshape<3, 4, 5>());
        EXPECT_EQ(sc[0], 1u);
        EXPECT_EQ(sc[1], 3u);
        EXPECT_EQ(sc[2], 12u);

        std::array<std::ptrdiff_t, 3> ts1 = {1, 5, 3}, tt1;

        auto sc2 = get_strides<layout_type::column_major, const_array<ptrdiff_t, 3>>(xshape<1, 5, 3>());
        compute_strides(ts1, layout_type::column_major, tt1);
        EXPECT_EQ(tt1[0], sc2[0]);
        EXPECT_EQ(tt1[1], sc2[1]);
        EXPECT_EQ(tt1[2], sc2[2]);

        auto sc3c = get_strides<layout_type::column_major, const_array<ptrdiff_t, 6>>(xshape<3, 1, 3, 2, 1, 3>());
        auto sc3r = get_strides<layout_type::row_major, const_array<ptrdiff_t, 6>>(xshape<3, 1, 3, 2, 1, 3>());
        std::vector<std::size_t> ts2({3, 1, 3, 2, 1, 3}), tt2(6);

        compute_strides(ts2, layout_type::column_major, tt2);
        EXPECT_TRUE(std::equal(tt2.begin(), tt2.end(), sc3c.begin()) && ts2.size() == sc3c.size());
        compute_strides(ts2, layout_type::row_major, tt2);
        EXPECT_TRUE(std::equal(tt2.begin(), tt2.end(), sc3r.begin()) && ts2.size() == sc3r.size());

        xtensor_fixed<double, xshape<3, 1, 3, 2, 1, 3>> saxa;
        xtensor<double, 6> saxt(std::array<std::size_t, 6>{3, 1, 3, 2, 1, 3});
        EXPECT_TRUE(std::equal(saxa.backstrides().begin(), saxa.backstrides().end(), saxt.backstrides().begin()));
    }

    TEST(xtensor_fixed, adapt)
    {
        std::vector<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::vector<double> b = a;

        xfixed_adaptor<std::vector<double>&, xt::xshape<3, 4>> ad(a);
        auto bd = adapt(b, std::array<std::size_t, 2>{3, 4});

        EXPECT_EQ(ad.layout(), XTENSOR_DEFAULT_LAYOUT);

        EXPECT_EQ(ad(1, 1), bd(1, 1));
        auto expr = ad + bd;
        EXPECT_EQ(expr(1, 1), bd(1, 1) * 2);
        ad = bd * 2;
        EXPECT_EQ(bd(1, 1) * 2, ad(1, 1));
        EXPECT_EQ(a[0], 2);
    }

    TEST(xtensor_fixed, buffer_adaptor)
    {
        xtensor_fixed<double, xshape<3>> a;
        xtensor<double, 2> b = zeros<double>({3, 3});
        auto c = adapt(&b(0, 0), xshape<3>());
        c *= a;
    }

    TEST(xtensor_fixed, layout)
    {
        xtensor_fixed<double, xshape<2, 2>, layout_type::row_major> a;
        EXPECT_EQ(a.layout(), layout_type::row_major);
        xtensor_fixed<double, xshape<2, 2>, layout_type::column_major> b;
        EXPECT_EQ(b.layout(), layout_type::column_major);
    }

    TEST(xtensor_fixed, nulld)
    {
        xtensor_fixed<double, xshape<>> a = 123;
        xtensor_fixed<double, xshape<>> b(4);
        xtensor_fixed<double, xshape<>> c = 123;

        EXPECT_EQ(a(), 123);
        b += 432;
        EXPECT_EQ(b(), 432 + 4);
        EXPECT_TRUE(c == a);
    }

    auto check_shape_a()
    {
        xtensor_fixed<double, xshape<3, 4>> a = {{1,2,3,4}, {5,6,7,8}, {9,10,11,12}, {9,10,11,12}};
        return a;
    }

    auto check_shape_b()
    {
        xtensor_fixed<double, xshape<3, 4>> a({{1,2}, {5,6,7,8}, {9,10,11,12}});
        return a;
    }

    auto check_shape_c()
    {
        xtensor_fixed<double, xshape<3, 4>> a = {{1,2,3}, {5,6,7}, {9,10,11}};
        return a;
    }

    TEST(xtensor_fixed, initializer_list_constructor)
    {
        using T = xtensor_fixed<double, xshape<3, 4>>;
        T a = {{1,2,3,4}, {5,6,7,8}, {9,10,11,12}};

    #ifdef XTENSOR_ENABLE_ASSERT
        XT_EXPECT_THROW(T{{1}}, std::runtime_error);
        XT_EXPECT_THROW(check_shape_a(), std::runtime_error);
        XT_EXPECT_THROW(check_shape_b(), std::runtime_error);
        XT_EXPECT_THROW(check_shape_c(), std::runtime_error);
    #endif
    }

    TEST(xtensor_fixed, transpose)
    {
        xtensorf3x4 a = {{1,2,3,4}, {5,6,7,8}, {9,10,11,12}};
        xtensor_fixed<double, xshape<4, 3>> ta = xt::transpose(a);
        EXPECT_EQ(a(1, 1), ta(1, 1));
        EXPECT_EQ(a(2, 1), ta(1, 2));
    }

    TEST(xtensor_fixed, xfunction_eval)
    {
        xtensorf3x4 a({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
        xtensorf4 b({4, 5, 6, 7});

        auto f1 = a + b;
        bool truth = std::is_same<typename decltype(f1)::shape_type, xshape<3, 4>>::value;
        EXPECT_TRUE(truth);
        auto f2 = a + b + 5.0;
        auto f3 = 5 + a + b;
        auto f4 = a / 5 + b;

        truth = std::is_same<typename decltype(f2)::shape_type, xshape<3, 4>>::value;
        EXPECT_TRUE(truth);
        truth = std::is_same<typename decltype(f3)::shape_type, xshape<3, 4>>::value;
        EXPECT_TRUE(truth);
        truth = std::is_same<typename decltype(f4)::shape_type, xshape<3, 4>>::value;
        EXPECT_TRUE(truth);

        auto e1 = xt::eval(f1);
        auto e2 = xt::eval(f3);
        truth = std::is_same<decltype(e1), xtensor_fixed<double, xshape<3, 4>>>::value;
        EXPECT_TRUE(truth);
        truth = std::is_same<decltype(e2), xtensor_fixed<double, xshape<3, 4>>>::value;
        EXPECT_TRUE(truth);

        xtensor_fixed<char, xshape<   2, 1, 10, 5>> xa;
        xtensor_fixed<char, xshape<3, 2, 4, 10, 1>> xb;

        auto fx1 = xa * xb;
        auto fx2 = 5 + xb * xa;
        truth = std::is_same<typename decltype(fx1)::shape_type, xshape<3, 2, 4, 10, 5>>::value;
        EXPECT_TRUE(truth);
        truth = std::is_same<typename decltype(fx2)::shape_type, xshape<3, 2, 4, 10, 5>>::value;
        EXPECT_TRUE(truth);

        xtensor_fixed<char, xshape<   2, 1, 10, 5>> xc;
        auto fx3 = xa * xc;
        truth = std::is_same<typename decltype(fx3)::shape_type, xshape<2, 1, 10, 5>>::value;
        EXPECT_TRUE(truth);
    }

    TEST(xtensor_fixed, adaptor_function_assignment)
    {
        xt::xtensor<double, 4> a_Eps  = xt::zeros<double>({2, 2, 2, 2});
        xt::xtensor<double, 4> a_Epsd = xt::zeros<double>({2, 2, 2, 2});
        std::size_t e = 0, k = 1;
        auto Eps  = xt::adapt(&a_Eps (e, k, 0, 0), xt::xshape<2, 2>());
        auto Epsd = xt::adapt(&a_Epsd(e, k, 0, 0), xt::xshape<2, 2>());

        xt::noalias(Eps) = Epsd * 123;
        // Eps = Epsd * 123; <-- Enable after XTL release!
    }

    TEST(xtensor_fixed, print)
    {
        xtensor_fixed<char, xshape<2>> a = {0, 1};
        xtensor_fixed<char, xshape<2>> b = {1, 1};

        std::stringstream out;
        out << a + b;
        EXPECT_EQ("{1, 2}", out.str());
    }

    TEST(xtensor_fixed, nonsharable)
    {
        using fixed_tensor = xtensor_fixed<double, xshape<2>, layout_type::row_major, true>;
        using tiny_tensor = xtensor_fixed<double, xshape<2>, layout_type::row_major, false>;
        EXPECT_GT(sizeof(fixed_tensor), sizeof(tiny_tensor));
    }

    class FixedMember
    {
    public:
        FixedMember(size_t bar, double foo)
        {
            m_bar = bar;
            m_foo = foo;
            m_I = xt::xtensor_fixed<double, xt::xshape<2, 2>>({{1.0, 0.0},
                                                               {0.0, 1.0}});
        }

        xt::xtensor_fixed<double, xt::xshape<2, 2>> get() const
        {
            xt::xtensor_fixed<double, xt::xshape<2, 2>> ret = 2.0 * m_I;
            return ret;
        }

    private:
        size_t m_bar;
        double m_foo;
        xt::xtensor_fixed<double, xt::xshape<2, 2>> m_I;
    };

    TEST(xtensor_fixed, member)
    {
        FixedMember c(20, 1.2);
        auto m = c.get();
        auto r = xt::xtensor_fixed<double, xt::xshape<2, 2>>({{2.0, 0.0},
                                                              {0.0, 2.0}});
        EXPECT_TRUE(xt::all(xt::equal(m, r)));
    }

}

#endif

#endif // VS_SKIP_XFIXED
