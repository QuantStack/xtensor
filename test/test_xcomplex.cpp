/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xcomplex.hpp"

#include <complex>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xnorm.hpp"

namespace xt
{
    using namespace std::complex_literals;

    TEST(xcomplex, expression)
    {
        xarray<std::complex<double>> e =
            {{1.0       , 1.0 + 1.0i},
             {1.0 - 1.0i, 1.0       }};

        // Test real expression
        auto r = real(e);
        auto i = imag(e);

        ASSERT_EQ(r.dimension(), size_t(2));
        ASSERT_EQ(i.dimension(), size_t(2));

        ASSERT_EQ(r.shape()[0], size_t(2));
        ASSERT_EQ(r.shape()[1], size_t(2));
        ASSERT_EQ(i.shape()[0], size_t(2));
        ASSERT_EQ(i.shape()[1], size_t(2));

        ASSERT_EQ(i(0, 0), 0);
        ASSERT_EQ(i(0, 1), 1);
        ASSERT_EQ(i(1, 0), -1);
        ASSERT_EQ(i(1, 1), 0);

        // Test assignment to an array
        xarray<double> ar = r;
        EXPECT_TRUE(all(equal(ar, ones<double>({2, 2}))));
    }

    TEST(xcomplex, lvalue)
    {
        xarray<std::complex<double>> e =
            {{1.0       , 1.0 + 1.0i},
             {1.0 - 1.0i, 1.0       }};

        // Test assigning an expression to the complex view
        real(e) = zeros<double>({2, 2});
        xarray<std::complex<double>> expect1 =
            {{0.0       , 0.0 + 1.0i},
             {0.0 - 1.0i, 0.0       }};
        EXPECT_TRUE(all(equal(e, expect1)));

        imag(e) = zeros<double>({2, 2});
        EXPECT_TRUE(all(equal(e, zeros<std::complex<double>>({2, 2}))));
    }

    TEST(xcomplex, scalar_assignmnent)
    {
        xarray<std::complex<double>> e =
            {{1.0       , 1.0 + 1.0i},
             {1.0 - 1.0i, 1.0       }};

        // Test assigning an expression to the complex view
        real(e) = 0.0;
        xarray<std::complex<double>> expect1 =
            {{0.0       , 0.0 + 1.0i},
             {0.0 - 1.0i, 0.0       }};
        EXPECT_TRUE(all(equal(e, expect1)));
    }

    TEST(xcomplex, noncomplex)
    {
        xarray<double> e = ones<double>({2, 2});
        auto r = real(e);
        auto i = imag(e);
        EXPECT_TRUE(all(equal(r, e)));
        EXPECT_TRUE(all(equal(i, zeros<double>({2, 2}))));
    }

    TEST(xcomplex, scalar)
    {
        double d = 1.0;
        ASSERT_EQ(1.0, real(d));
        ASSERT_EQ(0.0, imag(d));
        real(d) = 2.0;
        ASSERT_EQ(2.0, d);
    }

    TEST(xcomplex, pointer)
    {
        xarray<std::complex<double>> e =
            {{1.0       , 1.0 + 1.0i},
             {1.0 - 1.0i, 1.0       }};
        auto r = real(e);
        auto it = r.begin();
        EXPECT_EQ(*(it.operator->()), 1.0);
    }

    TEST(xcomplex, abs_angle_conj)
    {
        xarray<std::complex<double>> cmplarg_0 = {{0.40101756 + 0.71233018i, 0.62731701 + 0.42786349i, 0.32415089 + 0.2977805i},
                                                  {0.24475928 + 0.49208478i, 0.69475518 + 0.74029639i, 0.59390240 + 0.35772892i},
                                                  {0.63179202 + 0.41720995i, 0.44025718 + 0.65472131i, 0.08372648 + 0.37380143i}};
        auto cmplres = xt::abs(cmplarg_0);
        xarray<double> cmplexpected = {{0.81745298, 0.75933774, 0.44016704},
                                       {0.54959488, 1.01524554, 0.69331814},
                                       {0.75711643, 0.78897806, 0.38306348}};

        EXPECT_TRUE(allclose(cmplexpected, cmplres));

        auto cmplres_angle = xt::angle(cmplarg_0);
        xarray<double> cmplexpected_angle = {{1.05805307, 0.59857922, 0.74302273},
                                             {1.10923689, 0.81712241, 0.54213553},
                                             {0.58362348, 0.97881125, 1.35044673}};
        EXPECT_TRUE(allclose(cmplexpected_angle, cmplres_angle));

        using assign_t_angle = xassign_traits<xarray<double>, decltype(cmplres_angle)>;

#if XTENSOR_USE_XSIMD
        EXPECT_TRUE(assign_t_angle::simd_linear_assign());
#endif

        auto cmplres_conj = xt::conj(cmplarg_0);
        xarray<std::complex<double>> cmplexpected_conj = {{0.40101756 - 0.71233018i, 0.62731701 - 0.42786349i, 0.32415089 - 0.2977805i},
                                                          {0.24475928 - 0.49208478i, 0.69475518 - 0.74029639i, 0.59390240 - 0.35772892i},
                                                          {0.63179202 - 0.41720995i, 0.44025718 - 0.65472131i, 0.08372648 - 0.37380143i}};
        EXPECT_TRUE(allclose(cmplexpected_conj, cmplres_conj));

        using assign_t_conj = xassign_traits<xarray<std::complex<double>>, decltype(cmplres_conj)>;

#if XTENSOR_USE_XSIMD
        auto b1 = cmplres_angle.template load_simd<xsimd::aligned_mode>(0);
        auto b2 = cmplres_conj.template load_simd<xsimd::aligned_mode>(0);
        static_cast<void>(b1);
        static_cast<void>(b2);
        EXPECT_TRUE(assign_t_conj::simd_linear_assign());
#endif

        auto cmplres_norm = xt::norm(cmplarg_0);
        xarray<double> fieldnorm = {{0.66822937, 0.5765938, 0.19374703},
                                    {0.30205453, 1.0307235, 0.48069004},
                                    {0.57322529, 0.62248637, 0.14673763}};

        using assign_t_norm = xassign_traits<xarray<double>, decltype(cmplres_norm)>;

#if XTENSOR_USE_XSIMD
        EXPECT_TRUE(assign_t_norm::simd_linear_assign());
#endif

        EXPECT_TRUE(allclose(fieldnorm, cmplres_norm));
    }

    TEST(xcomplex, arg)
    {
        xarray<std::complex<double>> cmplarg_0 = {{0.40101756 + 0.71233018i, 0.62731701 + 0.42786349i, 0.32415089 + 0.2977805i},
                                                  {0.24475928 + 0.49208478i, 0.69475518 + 0.74029639i, 0.59390240 + 0.35772892i},
                                                  {0.63179202 + 0.41720995i, 0.44025718 + 0.65472131i, 0.08372648 + 0.37380143i}};
        auto cmplres = xt::arg(cmplarg_0);

        auto evc = xt::eval(cmplres);
        auto it = cmplarg_0.begin();

        for (auto el : evc)
        {
            auto exp = std::arg(*it);
            EXPECT_DOUBLE_EQ(el, exp);
            ++it;
        }

        using assign_t_arg = xassign_traits<xarray<double>, decltype(cmplres)>;

#if XTENSOR_USE_XSIMD
        EXPECT_TRUE(assign_t_arg::simd_linear_assign());
#endif

    }

    TEST(xcomplex, conj_real)
    {
        xarray<double> A = {{0.81745298, 0.75933774, 0.44016704},
                            {0.54959488, 1.01524554, 0.69331814},
                            {0.75711643, 0.78897806, 0.38306348}};
        xarray<double> B = xt::conj(A);
        EXPECT_EQ(A, B);
    }

    TEST(xcomplex, isnan)
    {
        using c_t = std::complex<double>;
        double nan = std::numeric_limits<double>::quiet_NaN();

        xarray<std::complex<double>> e = {c_t(0, 1), c_t(0, nan), c_t(-nan, 2), c_t(nan, -nan)};
        xarray<bool> expected = {false, true, true, true};
        // Full qualification required by Windows
        EXPECT_TRUE(all(equal(expected, xt::isnan(e))));
    }

    TEST(xcomplex, isinf)
    {
        using c_t = std::complex<double>;
        double inf = std::numeric_limits<double>::infinity();

        xarray<std::complex<double>> e = {c_t(0, 1), c_t(0, inf), c_t(-inf, 2), c_t(inf, -inf), c_t(0, -inf)};
        xarray<bool> expected = {false, true, true, true, true};

        EXPECT_TRUE(all(equal(expected, xt::isinf(e))));
    }

    TEST(xcomplex, isclose)
    {
        xarray<std::complex<double>> arg = {{0.40101756 + 0.71233018i, 0.62731701 + 0.42786349i, 0.32415089 + 0.2977805i},
                                            {0.24475928 + 0.49208478i, 0.69475518 + 0.74029639i, 0.59390240 + 0.35772892i},
                                            {0.63179202 + 0.41720995i, 0.44025718 + 0.65472131i, 0.08372648 + 0.37380143i}};

        xarray<std::complex<double>> compare = {{0.401 + 0.712i, 0.627 + 0.427i, 0.324 + 0.297i},
                                                {0.244 + 0.492i, 0.694 + 0.740i, 0.593 + 0.357i},
                                                {0.631 + 0.417i, 0.440 + 0.654i, 0.083 + 0.373i}};

        auto veryclose = isclose(arg, compare, 1e-5);
        auto looselyclose = isclose(arg, compare, 1e-1);

        EXPECT_TRUE(all(equal(false, veryclose)));
        EXPECT_TRUE(all(equal(true, looselyclose)));

        double inf = std::numeric_limits<double>::infinity();
        double nan = std::numeric_limits<double>::quiet_NaN();
        using c_t = std::complex<double>;

        EXPECT_TRUE(isclose(c_t(0, nan), c_t(0, nan))() == false);
        EXPECT_TRUE(isclose(c_t(0, nan), c_t(0, nan), 1e-5, 1e-3, true)() == true);
        EXPECT_TRUE(isclose(c_t(0, inf), c_t(0, inf))() == true);
        EXPECT_TRUE(isclose(c_t(0, -inf), c_t(0, inf))() == false);

        EXPECT_TRUE(isclose(c_t(inf, -inf), c_t(0, inf))() == false);
        EXPECT_TRUE(isclose(c_t(5, 5), c_t(5, -5))() == false);

    }

    TEST(xcomplex, real_expression)
    {
        using cpx = std::complex<double>;
        xtensor<cpx, 2> a = {{ cpx(1, 1), cpx(-1, 1), cpx(-2, -2) },
                             { cpx(-1, 0), cpx(0, 1), cpx(2, 2) }};

        xtensor<double, 2> exp = {{2, -2, -4},
                                  {-2, 0, 4}};
        xtensor<double, 2> res = real(a + a);
        EXPECT_EQ(res, exp);
    }

    TEST(xcomplex, conj)
    {
        using cpx = std::complex<double>;
        xtensor<cpx, 2> a = {{ cpx(1, 1), cpx(-1, 1), cpx(-2, -2) },
                             { cpx(-1, 0), cpx(0, 1), cpx(2, 2) }};
        xtensor<cpx, 2> res = conj(a);
        xtensor<cpx, 2> exp = {{ cpx(1, -1), cpx(-1, -1), cpx(-2, 2) },
                             { cpx(-1, 0), cpx(0, -1), cpx(2, -2) }};

        EXPECT_EQ(res, exp);
    }

    TEST(xcomplex, exp)
    {
        xt::xarray<float> ph = {274.7323f, 276.3974f, 274.7323f, 276.3974f, 274.7323f, 276.3974f, 274.7323f, 276.3974f};
        xt::xarray<std::complex<float>> input = ph * std::complex<float>(0, 1.f);
        xt::xarray<std::complex<float>> res = xt::exp(input);
        auto expected = xt::xarray<std::complex<float>>::from_shape({ size_t(8) });
        std::transform(input.cbegin(), input.cend(), expected.begin(), [](const std::complex<float>& arg) {
            return std::exp(arg);
        });
        EXPECT_EQ(expected, res);
    }

    TEST(xcomplex, longdouble)
    {
        using cmplx = std::complex<long double>;
        xt::xtensor<cmplx, 2> a = xt::empty<cmplx>({5, 5});
        xt::real(a) = 123.321;
        xt::imag(a) = -123.321;

        EXPECT_EQ(a(4, 4), cmplx(123.321, -123.321));

        xt::real(a) = xt::imag(a);

        EXPECT_EQ(a(0, 0), cmplx(-123.321, -123.321));
        EXPECT_EQ(a(4, 4), cmplx(-123.321, -123.321));
    }
}
