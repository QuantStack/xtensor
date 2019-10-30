/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xmath.hpp"

#include <complex>
#include <limits>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xoptional_assembly.hpp"
#include "xtensor/xrandom.hpp"

namespace xt
{
    using std::size_t;
    using shape_type = dynamic_shape<size_t>;

    /********************
     * Basic operations *
     ********************/

    TEST(xmath, abs)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, -4.5);
        EXPECT_EQ(abs(a)(0, 0), std::abs(a(0, 0)));

        // check SIMD type deduction
        xarray<double> res = xt::abs(a);

        xarray<std::complex<double>> b(shape, std::complex<double>(1.2, 2.3));
        EXPECT_EQ(abs(b)(0, 0), std::abs(b(0, 0)));

        // check SIMD type deduction
        xarray<double> res2 = xt::abs(b);

        auto f = abs(b);
        using assign_traits = xassign_traits<xarray<double>, decltype(f)>;

#if XTENSOR_USE_XSIMD
        EXPECT_TRUE(assign_traits::simd_linear_assign());
#endif
    }

    TEST(xmath, fabs)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        EXPECT_EQ(fabs(a)(0, 0), std::fabs(a(0, 0)));
    }

    TEST(xmath, fmod)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(fmod(a, b)(0, 0), std::fmod(a(0, 0), b(0, 0)));

        double sb = 1.2;
        EXPECT_EQ(fmod(a, sb)(0, 0), std::fmod(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(fmod(sa, b)(0, 0), std::fmod(sa, b(0, 0)));
    }

    TEST(xmath, remainder)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(remainder(a, b)(0, 0), std::remainder(a(0, 0), b(0, 0)));

        double sb = 1.2;
        EXPECT_EQ(remainder(a, sb)(0, 0), std::remainder(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(remainder(sa, b)(0, 0), std::remainder(sa, b(0, 0)));
    }

    TEST(xmath, fma)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        xarray<double> c(shape, 2.6);
        EXPECT_EQ(xt::fma(a, b, c)(0, 0), std::fma(a(0, 0), b(0, 0), c(0, 0)));

        double sb = 1.2;
        EXPECT_EQ(xt::fma(a, sb, c)(0, 0), std::fma(a(0, 0), sb, c(0, 0)));

        double sa = 4.6;
        EXPECT_EQ(xt::fma(sa, b, c)(0, 0), std::fma(sa, b(0, 0), c(0, 0)));

        EXPECT_EQ(xt::fma(sa, sb, c)(0, 0), std::fma(sa, sb, c(0, 0)));
    }

    TEST(xmath, fmax)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(fmax(a, b)(0, 0), std::fmax(a(0, 0), b(0, 0)));

        double sb = 1.2;
        EXPECT_EQ(fmax(a, sb)(0, 0), std::fmax(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(fmax(sa, b)(0, 0), std::fmax(sa, b(0, 0)));
    }

    TEST(xmath, fmin)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(fmin(a, b)(0, 0), std::fmin(a(0, 0), b(0, 0)));

        double sb = 1.2;
        EXPECT_EQ(fmin(a, sb)(0, 0), std::fmin(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(fmin(sa, b)(0, 0), std::fmin(sa, b(0, 0)));
    }

    TEST(xmath, fdim)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(fdim(a, b)(0, 0), std::fdim(a(0, 0), b(0, 0)));

        double sb = 1.2;
        EXPECT_EQ(fdim(a, sb)(0, 0), std::fdim(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(fdim(sa, b)(0, 0), std::fdim(sa, b(0, 0)));
    }

    TEST(xmath, amin_amax)
    {
        xarray<double> a{-10.0};
        EXPECT_EQ(amin(a)[0], -10.0);
        EXPECT_EQ(amax(a)[0], -10.0);

        xarray<double> b{-10.0, -20.0};
        EXPECT_EQ(amin(b)[0], -20.0);
        EXPECT_EQ(amax(b)[0], -10.0);

        xarray<double> c{-10.0, +20.0};
        EXPECT_EQ(amin(c)[0], -10.0);
        EXPECT_EQ(amax(c)[0], +20.0);

        xarray<double> d{+10.0, +20.0};
        EXPECT_EQ(amin(d)[0], +10.0);
        EXPECT_EQ(amax(d)[0], +20.0);

        xarray<double> e{+10.0};
        EXPECT_EQ(amin(e)[0], +10.0);
        EXPECT_EQ(amax(e)[0], +10.0);
    }

    TEST(xmath, minimum)
    {
        using opt_type = xoptional_assembly<xarray<double>, xarray<bool>>;
        auto missing = xtl::missing<double>();

        xarray<double> a = {1, 2, 3, 4, 5, 6};
        xarray<double> b = {6, 5, 4, 3, 2, 1};
        opt_type opt_a = {1, missing, 3, 4,       5, missing};
        opt_type opt_b = {6,       5, 4, 3, missing,       1};

        xarray<double> res = {1, 2, 3, 3, 2, 1};
        EXPECT_EQ(res, minimum(a, b));

        opt_type res1 = {1, missing, 3, 3, 2, missing};
        EXPECT_EQ(res1, minimum(opt_a, b));

        opt_type res2 = {1, missing, 3, 3, missing, missing};
        EXPECT_EQ(res2, minimum(opt_a, opt_b));
    }

    TEST(xmath, maximum)
    {
        using opt_type = xoptional_assembly<xarray<double>, xarray<bool>>;
        auto missing = xtl::missing<double>();

        xarray<double> a = {1, 2, 3, 4, 5, 6};
        xarray<double> b = {6, 5, 4, 3, 2, 1};
        opt_type opt_a = {1, missing, 3, 4,       5, missing};
        opt_type opt_b = {6,       5, 4, 3, missing,       1};

        xarray<double> res = {6, 5, 4, 4, 5, 6};
        EXPECT_EQ(res, maximum(a, b));

        opt_type res1 = {6, missing, 4, 4, 5, missing};
        EXPECT_EQ(res1, maximum(opt_a, b));

        opt_type res2 = {6, missing, 4, 4, missing, missing};
        EXPECT_EQ(res2, maximum(opt_a, opt_b));
    }

    TEST(xmath, clip)
    {
        using opt_type = xoptional_assembly<xarray<double>, xarray<bool>>;
        auto missing = xtl::missing<double>();

        xarray<double> a = {1, 2, 3, 4, 5, 6};
        opt_type opt_a = {1, missing, 3, 4, 5, missing};

        xarray<double> res = {2, 2, 3, 4, 4, 4};
        EXPECT_EQ(res, clip(a, 2.0, 4.0));

        opt_type res1 = {2, missing, 3, 4, 4, missing};
        EXPECT_EQ(res1, clip(opt_a, 2.0, 4.0));
    }

    TEST(xmath, sign)
    {
        shape_type shape = {3, 2};
        xarray<float> a(shape, 1);
        a(0, 1) = -1;
        a(1, 1) = 0;
        a(2, 1) = -0;

        auto signs = sign(a);
        EXPECT_EQ(1.f, signs(0, 0));
        EXPECT_EQ(-1.f, signs(0, 1));
        EXPECT_EQ(0.f, signs(1, 1));
        EXPECT_EQ(0.f, signs(2, 1));

        xarray<unsigned int> b(shape, 1);
        b(1, 1) = static_cast<unsigned int>(-1);
        b(2, 1) = 0;

        auto signs_b = sign(b);
        EXPECT_EQ(1u, signs_b(0, 0));
        // sign from overflow
        EXPECT_EQ(1u, signs_b(1, 1));
        EXPECT_EQ(0u, signs_b(2, 1));

        xarray<double> c(shape, 1);
        c(0, 0) = std::numeric_limits<double>::infinity();
        c(0, 1) = -std::numeric_limits<double>::infinity();
        c(1, 0) = std::numeric_limits<double>::quiet_NaN();
        c(1, 1) = -std::numeric_limits<double>::quiet_NaN();

        auto signs_c = sign(c);
        EXPECT_EQ(1, signs_c(0, 0));
        EXPECT_EQ(-1, signs_c(0, 1));
        EXPECT_TRUE(std::isnan(signs_c(1, 0)));
        EXPECT_TRUE(std::isnan(signs_c(1, 1)));

        using ctype = std::complex<double>;
        xarray<ctype> d(shape, ctype(3, 2));
        d(0, 0) = ctype(1, 1);
        d(0, 1) = ctype(-1, 1);
        d(1, 0) = ctype(0, -1);
        d(1, 1) = ctype(-0, 1);

        auto signs_d = sign(d);
        EXPECT_EQ(ctype(1, 0), signs_d(0, 0));
        EXPECT_EQ(ctype(-1, 0), signs_d(0, 1));
        EXPECT_EQ(ctype(-1, 0), signs_d(1, 0));
        EXPECT_EQ(ctype(1, 0), signs_d(1, 1));
    }

    TEST(xmath, isnan)
    {
        xarray<double> arr
           {{1.0, std::numeric_limits<double>::quiet_NaN()},
            {std::numeric_limits<double>::quiet_NaN(), 0.0}};
        xarray<bool> expected
           {{false, true}, {true, false}};
        EXPECT_TRUE(all(equal(expected, xt::isnan(arr))));
    }

    TEST(xmath, deg2rad)
    {
        xarray<double> arr
            {-180, -135, -90, -45, 0, 45, 90, 135, 180};
        xarray<double> expected
            {-3.141593, -2.356194, -1.570796, -0.785398,  0.,
              0.785398,  1.570796,  2.356194,  3.141593};
        EXPECT_TRUE(all(isclose(expected, xt::deg2rad(arr))));
    }

    TEST(xmath, radians)
    {
        xarray<double> arr
            {-180, -135, -90, -45, 0, 45, 90, 135, 180};
        xarray<double> expected
            {-3.141593, -2.356194, -1.570796, -0.785398,  0.,
             0.785398,  1.570796,  2.356194,  3.141593};
        EXPECT_TRUE(all(isclose(expected, xt::radians(arr))));
    }

    TEST(xmath, rad2deg)
    {
        xarray<double> arr
            {-3.141593, -2.356194, -1.570796, -0.785398,  0.,
             0.785398,  1.570796,  2.356194,  3.141593};
        xarray<double> expected
            {-180, -135, -90, -45, 0, 45, 90, 135, 180};
        EXPECT_TRUE(all(isclose(expected, xt::rad2deg(arr))));
    }

    TEST(xmath, degrees)
    {
        xarray<double> arr
            {-3.141593, -2.356194, -1.570796, -0.785398,  0.,
             0.785398,  1.570796,  2.356194,  3.141593};
        xarray<double> expected
            {-180, -135, -90, -45, 0, 45, 90, 135, 180};
        EXPECT_TRUE(all(isclose(expected, xt::degrees(arr))));
    }

    /*************************
     * Exponential functions *
     *************************/

    TEST(xmath, assign_traits)
    {
        using array_type = xarray<double>;
        array_type a = { {1.2, 2.3}, {3.4, 4.5} };

        {
            SCOPED_TRACE("unary function");
            auto fexp = exp(a);
            using assign_traits = xassign_traits<array_type, decltype(fexp)>;
#if XTENSOR_USE_XSIMD
            EXPECT_TRUE(assign_traits::simd_linear_assign());
#else
            // SFINAE on load_simd is broken on mingw when xsimd is disabled. This using
            // triggers the same error as the one caught by mingw.
            using return_type = decltype(fexp.template load_simd<aligned_mode>(std::size_t(0)));
            EXPECT_FALSE(assign_traits::simd_linear_assign());
            EXPECT_TRUE((std::is_same<return_type, double>::value));
#endif
        }

        {
            SCOPED_TRACE("binary function");
            auto fpow = pow(a, a);
            using assign_traits = xassign_traits<array_type, decltype(fpow)>;
#if XTENSOR_USE_XSIMD
            EXPECT_TRUE(assign_traits::simd_linear_assign());
#else
            // SFINAE on load_simd is broken on mingw when xsimd is disabled. This using
            // triggers the same error as the one caught by mingw.
            using return_type = decltype(fpow.template load_simd<aligned_mode>(std::size_t(0)));
            EXPECT_FALSE(assign_traits::simd_linear_assign());
            EXPECT_TRUE((std::is_same<return_type, double>::value));
#endif
        }

        {
            SCOPED_TRACE("ternary function");
            auto ffma = xt::fma(a, a, a);
            using assign_traits = xassign_traits<array_type, decltype(ffma)>;
#if XTENSOR_USE_XSIMD
            EXPECT_TRUE(assign_traits::simd_linear_assign());
#else
            // SFINAE on load_simd is broken on mingw when xsimd is disabled. This using
            // triggers the same error as the one caught by mingw.
            using return_type = decltype(ffma.template load_simd<aligned_mode>(std::size_t(0)));
            EXPECT_FALSE(assign_traits::simd_linear_assign());
            EXPECT_TRUE((std::is_same<return_type, double>::value));
#endif
        }
    }

    TEST(xmath, exp)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(exp(a)(0, 0), std::exp(a(0, 0)));
    }

    TEST(xmath, exp2)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(exp2(a)(0, 0), std::exp2(a(0, 0)));
    }

    TEST(xmath, expm1)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(expm1(a)(0, 0), std::expm1(a(0, 0)));
    }

    TEST(xmath, log)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(log(a)(0, 0), std::log(a(0, 0)));
    }

    TEST(xmath, log2)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(log2(a)(0, 0), std::log2(a(0, 0)));
    }

    TEST(xmath, log1p)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(log1p(a)(0, 0), std::log1p(a(0, 0)));
    }

    /*******************
     * Power functions *
     *******************/

    TEST(xmath, pow)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(pow(a, b)(0, 0), std::pow(a(0, 0), b(0, 0)));

        double sb = 1.2;
        EXPECT_EQ(pow(a, sb)(0, 0), std::pow(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(pow(sa, b)(0, 0), std::pow(sa, b(0, 0)));
    }

    TEST(xmath, sqrt)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(sqrt(a)(0, 0), std::sqrt(a(0, 0)));
    }

    TEST(xmath, square)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(square(a)(0, 0), (a(0, 0) * a(0, 0)));
        xarray<double> b = square(a);
        xarray<double> exp = a * a;
        EXPECT_EQ(b, exp);

        auto f = square(a);

#if XTENSOR_USE_XSIMD
        using assign_traits = xassign_traits<xarray<double>, decltype(f)>;
        EXPECT_TRUE(assign_traits::simd_linear_assign());
#endif
    }

    TEST(xmath, integer_pow)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3);

        xarray<double> b = pow<16>(a);
        xarray<double> exp = pow(a, 16);
        EXPECT_TRUE(allclose(exp, b));

        b = pow<1>(a);
        exp = pow(a, 1);
        EXPECT_TRUE(allclose(exp, b));
        b = pow<2>(a);
        exp = pow(a, 2);
        EXPECT_TRUE(allclose(exp, b));
        b = pow<3>(a);
        exp = pow(a, 3);
        EXPECT_TRUE(allclose(exp, b));
        b = pow<4>(a);
        exp = pow(a, 4);
        EXPECT_TRUE(allclose(exp, b));
        b = pow<5>(a);
        exp = pow(a, 5);
        EXPECT_TRUE(allclose(exp, b));
        b = pow<13>(a);
        exp = pow(a, 13);
        EXPECT_TRUE(allclose(exp, b));

        auto f = pow<13>(a);

#if XTENSOR_USE_XSIMD
        using assign_traits = xassign_traits<xarray<double>, decltype(f)>;
        EXPECT_TRUE(assign_traits::simd_linear_assign());
#endif

    }

    TEST(xmath, cube)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(cube(a)(0, 0), (a(0, 0) * a(0, 0) * a(0, 0)));
        xarray<double> b = cube(a);
        xarray<double> exp = a * a * a;
        EXPECT_EQ(b, exp);
    }

    TEST(xmath, cbrt)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(cbrt(a)(0, 0), std::cbrt(a(0, 0)));
    }

    TEST(xmath, hypot)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(hypot(a, b)(0, 0), std::hypot(a(0, 0), b(0, 0)));

        double sb = 1.2;
        EXPECT_EQ(hypot(a, sb)(0, 0), std::hypot(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(hypot(sa, b)(0, 0), std::hypot(sa, b(0, 0)));
    }

    /***************************
     * Trigonometric functions *
     ***************************/

    TEST(xmath, sin)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(sin(a)(0, 0), std::sin(a(0, 0)));
    }

    TEST(xmath, cos)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(cos(a)(0, 0), std::cos(a(0, 0)));
    }

    TEST(xmath, tan)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(tan(a)(0, 0), std::tan(a(0, 0)));
    }

    TEST(xmath, asin)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(asin(a)(0, 0), std::asin(a(0, 0)));
    }

    TEST(xmath, acos)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(acos(a)(0, 0), std::acos(a(0, 0)));
    }

    TEST(xmath, atan)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(atan(a)(0, 0), std::atan(a(0, 0)));
    }

    TEST(xmath, atan2)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(atan2(a, b)(0, 0), std::atan2(a(0, 0), b(0, 0)));

        double sb = 1.2;
        EXPECT_EQ(atan2(a, sb)(0, 0), std::atan2(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(atan2(sa, b)(0, 0), std::atan2(sa, b(0, 0)));
    }


    /************************
     * Hyperbolic functions *
     ************************/

    TEST(xmath, sinh)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(sinh(a)(0, 0), std::sinh(a(0, 0)));
    }

    TEST(xmath, cosh)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(cosh(a)(0, 0), std::cosh(a(0, 0)));
    }

    TEST(xmath, tanh)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(tanh(a)(0, 0), std::tanh(a(0, 0)));
    }

    TEST(xmath, asinh)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(asinh(a)(0, 0), std::asinh(a(0, 0)));
    }

    TEST(xmath, acosh)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 1.7);
        EXPECT_EQ(acosh(a)(0, 0), std::acosh(a(0, 0)));
    }

    TEST(xmath, atanh)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(atanh(a)(0, 0), std::atanh(a(0, 0)));
    }


    /*****************************
     * Error and gamma functions *
     *****************************/

    TEST(xmath, erf)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(erf(a)(0, 0), std::erf(a(0, 0)));
    }

    TEST(xmath, erfc)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(erfc(a)(0, 0), std::erfc(a(0, 0)));
    }

    TEST(xmath, tgamma)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(tgamma(a)(0, 0), std::tgamma(a(0, 0)));
    }

    TEST(xmath, lgamma)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(lgamma(a)(0, 0), std::lgamma(a(0, 0)));
    }

    TEST(xmath, ceil)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 2.3);
        EXPECT_EQ(ceil(a)(0, 0), std::ceil(a(0, 0)));
    }

    TEST(xmath, floor)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 2.3);
        EXPECT_EQ(floor(a)(0, 0), std::floor(a(0, 0)));
    }

    TEST(xmath, trunc)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 2.3);
        EXPECT_EQ(trunc(a)(0, 0), std::trunc(a(0, 0)));
    }

    TEST(xmath, round)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 2.3);
        EXPECT_EQ(round(a)(0, 0), std::round(a(0, 0)));
    }

    TEST(xmath, nearbyint)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 2.3);
        EXPECT_EQ(nearbyint(a)(0, 0), std::nearbyint(a(0, 0)));
    }

    TEST(xmath, rint)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 2.3);
        EXPECT_EQ(rint(a)(0, 0), std::rint(a(0, 0)));
    }

    TEST(xmath, isclose)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        xarray<double> b(shape, 0.70000000001);
        xarray<double> c(shape, 0.80000000001);
        EXPECT_TRUE(allclose(a, b));
        EXPECT_FALSE(allclose(a, c));
        b(1, 1) = 1;
        EXPECT_FALSE(allclose(a, b));
        EXPECT_TRUE(allclose(a, b, 10, 10));

        b = a;
        b(0, 0) = nan("n");
        EXPECT_FALSE(isclose(a, b)(0, 0));
        EXPECT_FALSE(isclose(a, b)(0, 0));
        a(0, 0) = nan("n");
        EXPECT_FALSE(isclose(a, b)(0, 0));
        EXPECT_TRUE(isclose(a, b, 1, 1, true)(0, 0));
    }

    TEST(xmath, isclose_int)
    {
        EXPECT_FALSE(isclose(1, 2)());
        EXPECT_TRUE(isclose(1, 2, 1)());
        EXPECT_TRUE(isclose(1u, 2u, 1u)());
        EXPECT_TRUE(isclose(1ul, 2ul, 1ul)());
        EXPECT_TRUE(isclose(1ul, 10ul, 1ul)());
        EXPECT_TRUE(isclose(1ul, 10ul, 1ul, 100ul)());
    }

    TEST(xmath, integer_abs)
    {
        EXPECT_EQ(math::abs(1ul), 1ul);
        EXPECT_EQ(math::abs(1u), 1u);
        EXPECT_EQ(math::abs(static_cast<unsigned char>(1)), static_cast<unsigned char>(1));
        EXPECT_EQ(math::abs(char(1)), 1);
        EXPECT_EQ(math::abs(int(1)), 1);
        EXPECT_EQ(math::abs(int(-1)), 1);
        EXPECT_EQ(math::abs(long(-1)), long(1));
        EXPECT_EQ(math::abs(-1.5), 1.5);
    }

    TEST(xmath, isinf_nan_int)
    {
        EXPECT_FALSE(math::isinf(132));
        EXPECT_FALSE(math::isinf(-123123));
        EXPECT_FALSE(math::isnan(132));
        EXPECT_FALSE(math::isnan(-123123));
        EXPECT_FALSE(math::isinf(132ul));
        EXPECT_FALSE(math::isnan(123123ul));
    }

    TEST(xmath, scalar_cast)
    {
        double arg = 1.0;
        double res = ::xt::atan(xscalar<double>(arg));
        EXPECT_EQ(res, std::atan(arg));
        bool close = ::xt::isclose(1.0, 1.0);
        EXPECT_EQ(close, true);

        auto p = ::xt::numeric_constants<>::PI;
        EXPECT_EQ(p, 3.141592653589793238463);
    }

    TEST(xmath, count_nonzero)
    {
        xarray<double> a = {{1, 2, 3, 4}, {0, 0, 0, 0}, {3, 0, 1, 0}};
        std::size_t as = count_nonzero(a)();
        std::size_t ase = count_nonzero(a, evaluation_strategy::immediate)();
        EXPECT_EQ(as, 6u);
        EXPECT_EQ(ase, 6u);

        xarray<std::size_t> ea0 = {2, 1, 2, 1};
        xarray<std::size_t> ea1 = {4, 0, 2};

        EXPECT_EQ(count_nonzero(a, {0}), ea0);
        EXPECT_EQ(count_nonzero(a, {1}), ea1);

        EXPECT_EQ(count_nonzero(a, {0}, evaluation_strategy::immediate), ea0);
        EXPECT_EQ(count_nonzero(a, {1}, evaluation_strategy::immediate), ea1);

        a = random::randint<int>({5, 5, 5, 5, 5}, 10);
        auto lm = count_nonzero(a, {0, 1, 3}, evaluation_strategy::immediate);
        auto lz = count_nonzero(a, {0, 1, 3});
        EXPECT_EQ(lm, lz);
    }

    TEST(xmath, diff)
    {
        xt::xarray<int> a = {1, 2, 4, 7, 0};
        xt::xarray<int> expected1 = {1,  2,  3, -7};
        EXPECT_EQ(xt::diff(a), expected1);
        xt::xarray<int> expected2 = {1, 1, -10};
        EXPECT_EQ(xt::diff(a, 2), expected2);

        xt::xarray<int> b = {{1, 3, 6, 10}, {0, 5, 6, 8}};
        xt::xarray<int> expected3 = {{2, 3, 4}, {5, 1, 2}};
        EXPECT_EQ(xt::diff(b), expected3);
        xt::xarray<int> expected4 = {{-1, 2, 0, -2}};
        EXPECT_EQ(xt::diff(b, 1, 0), expected4);

        xt::xarray<bool> c = {{true, false, true}, {true, true, true}};
        xt::xarray<bool> expected6 = {{true, true}, {false, false}};
        EXPECT_EQ(xt::diff(c, 1), expected6);
        xt::xarray<bool> expected7({2, 1}, false);
        EXPECT_EQ(xt::diff(c, 2), expected7);

        std::vector<int> d = { 1, 2, 4, 7, 0 };
        xt::xarray<int> orig = { 1, 2, 4, 7, 0 };
        auto ad = xt::adapt(d);
        EXPECT_EQ(xt::diff(ad), expected1);
        EXPECT_EQ(ad, orig);
    }

    TEST(xmath, trapz)
    {
        xt::xarray<int> a = {{0, 1, 2},
                             {3, 4, 5}};
        xt::xarray<double> expected1 = {1.5, 2.5, 3.5};
        EXPECT_EQ(trapz(a, 1.0, 0), expected1);

        xt::xarray<double> expected2 = {2.0, 8.0};
        EXPECT_EQ(trapz(a, 1.0, -1), expected2);

        xt::xarray<int> b = {1, 2, 3};
        auto res3 = trapz(b);
        EXPECT_EQ(res3[0], 4.0);

        xt::xarray<int> c = {1, 2, 3};
        auto res4 = trapz(c, 2.0);
        EXPECT_EQ(res4[0], 8.0);

        xt::xarray<int> d = {1, 2, 3};
        xt::xarray<int> d_x = {4, 6, 8};
        auto res5 = trapz(d, d_x);
        EXPECT_EQ(res5[0], 8.0);
    }

    /************************
     * Linear interpolation *
     ************************/

    TEST(xmath, interp)
    {
        xt::xtensor<double,1> xp = {0.0, 1.0, 3.0};
        xt::xtensor<double,1> fp = {0.0, 1.0, 3.0};
        xt::xtensor<double,1> x  = {0.0, .5, 1.0, 1.5, 2.0, 2.5, 3.0};

        auto f = xt::interp(x,xp,fp);

        for ( std::size_t i = 0 ; i < x.size() ; ++i )
        {
            EXPECT_EQ(f[i], x[i]);
        }
    }
}
