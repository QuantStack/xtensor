/****************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

/**
 * @brief standard mathematical functions for xexpressions
 */

#ifndef XTENSOR_MATH_HPP
#define XTENSOR_MATH_HPP

#include <cmath>
#include <algorithm>
#include <array>
#include <complex>
#include <type_traits>

#include <xtl/xcomplex.hpp>
#include <xtl/xtype_traits.hpp>

#include "xaccumulator.hpp"
#include "xoperation.hpp"
#include "xreducer.hpp"
#include "xslice.hpp"
#include "xstrided_view.hpp"
#include "xeval.hpp"

namespace xt
{
    template <class T = double>
    struct numeric_constants
    {
        static constexpr T PI = 3.141592653589793238463;
        static constexpr T PI_2 = 1.57079632679489661923;
        static constexpr T PI_4 = 0.785398163397448309616;
        static constexpr T D_1_PI = 0.318309886183790671538;
        static constexpr T D_2_PI = 0.636619772367581343076;
        static constexpr T D_2_SQRTPI = 1.12837916709551257390;
        static constexpr T SQRT2 = 1.41421356237309504880;
        static constexpr T SQRT1_2 = 0.707106781186547524401;
        static constexpr T E = 2.71828182845904523536;
        static constexpr T LOG2E = 1.44269504088896340736;
        static constexpr T LOG10E = 0.434294481903251827651;
        static constexpr T LN2 = 0.693147180559945309417;
    };

    /***********
     * Helpers *
     ***********/

#define XTENSOR_UNSIGNED_ABS_FUNC(T)                                                               \
    constexpr inline T abs(const T& x)                                                             \
    {                                                                                              \
        return x;                                                                                  \
    }

#define XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, T)                                  \
    constexpr inline bool FUNC_NAME(const T& /*x*/) noexcept                                       \
    {                                                                                              \
        return RETURN_VAL;                                                                         \
    }

#define XTENSOR_INT_SPECIALIZATION(FUNC_NAME, RETURN_VAL)                                          \
    XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, char);                                  \
    XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, short);                                 \
    XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, int);                                   \
    XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, long);                                  \
    XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, long long);                             \
    XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned char);                         \
    XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned short);                        \
    XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned int);                          \
    XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned long);                         \
    XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned long long);


#define XTENSOR_UNARY_MATH_FUNCTOR_IMPL(NAME, R)                                                   \
    struct NAME##_fun                                                                              \
    {                                                                                              \
        template <class T>                                                                         \
        constexpr auto operator()(const T& arg) const                                              \
        {                                                                                          \
            using math::NAME;                                                                      \
            return NAME(arg);                                                                      \
        }                                                                                          \
        template <class B>                                                                         \
        constexpr auto simd_apply(const B& arg) const                                              \
        {                                                                                          \
            using math::NAME;                                                                      \
            return NAME(arg);                                                                      \
        }                                                                                          \
    }

#define XTENSOR_UNARY_MATH_FUNCTOR(NAME) XTENSOR_UNARY_MATH_FUNCTOR_IMPL(NAME, T)
#define XTENSOR_UNARY_BOOL_FUNCTOR(NAME) XTENSOR_UNARY_MATH_FUNCTOR_IMPL(NAME, bool)

#define XTENSOR_UNARY_MATH_FUNCTOR_COMPLEX_REDUCING(NAME)                                          \
    struct NAME##_fun                                                                              \
    {                                                                                              \
        template <class T>                                                                         \
        constexpr auto operator()(const T& arg) const                                              \
        {                                                                                          \
            using math::NAME;                                                                      \
            return NAME(arg);                                                                      \
        }                                                                                          \
        template <class B>                                                                         \
        constexpr auto simd_apply(const B& arg) const                                              \
        {                                                                                          \
            using math::NAME;                                                                      \
            return NAME(arg);                                                                      \
        }                                                                                          \
    }

#define XTENSOR_BINARY_MATH_FUNCTOR_IMPL(NAME, R)                                                  \
    struct NAME##_fun                                                                              \
    {                                                                                              \
        template <class T1, class T2>                                                              \
        constexpr auto operator()(const T1& arg1, const T2& arg2) const                            \
        {                                                                                          \
            using math::NAME;                                                                      \
            return NAME(arg1, arg2);                                                               \
        }                                                                                          \
        template <class B>                                                                         \
        constexpr auto simd_apply(const B& arg1, const B& arg2) const                              \
        {                                                                                          \
            using math::NAME;                                                                      \
            return NAME(arg1, arg2);                                                               \
        }                                                                                          \
    }

#define XTENSOR_BINARY_MATH_FUNCTOR(NAME) XTENSOR_BINARY_MATH_FUNCTOR_IMPL(NAME, T)
#define XTENSOR_BINARY_BOOL_FUNCTOR(NAME) XTENSOR_BINARY_MATH_FUNCTOR_IMPL(NAME, bool)

#define XTENSOR_TERNARY_MATH_FUNCTOR_IMPL(NAME, R)                                                 \
    struct NAME##_fun                                                                              \
    {                                                                                              \
        template <class T1, class T2, class T3>                                                    \
        constexpr auto operator()(const T1& arg1, const T2& arg2, const T3& arg3) const            \
        {                                                                                          \
            using math::NAME;                                                                      \
            return NAME(arg1, arg2, arg3);                                                         \
        }                                                                                          \
        template <class B>                                                                         \
        auto simd_apply(const B& arg1, const B& arg2, const B& arg3) const                         \
        {                                                                                          \
            using math::NAME;                                                                      \
            return NAME(arg1, arg2, arg3);                                                         \
        }                                                                                          \
    }

#define XTENSOR_TERNARY_MATH_FUNCTOR(NAME) XTENSOR_TERNARY_MATH_FUNCTOR_IMPL(NAME, T)
#define XTENSOR_TERNARY_BOOL_FUNCTOR(NAME) XTENSOR_TERNARY_MATH_FUNCTOR_IMPL(NAME, bool)

    namespace math
    {
        using std::abs;
        using std::fabs;

        using std::acos;
        using std::asin;
        using std::atan;
        using std::cos;
        using std::sin;
        using std::tan;

        using std::acosh;
        using std::asinh;
        using std::atanh;
        using std::cosh;
        using std::sinh;
        using std::tanh;

        using std::cbrt;
        using std::sqrt;

        using std::exp;
        using std::exp2;
        using std::expm1;
        using std::ilogb;
        using std::log;
        using std::log10;
        using std::log1p;
        using std::log2;
        using std::logb;

        using std::ceil;
        using std::floor;
        using std::llround;
        using std::lround;
        using std::nearbyint;
        using std::remainder;
        using std::rint;
        using std::round;
        using std::trunc;

        using std::erf;
        using std::erfc;
        using std::erfc;
        using std::lgamma;
        using std::tgamma;

        using std::arg;
        using std::conj;
        using std::imag;
        using std::real;

        using std::atan2;

// copysign is not in the std namespace for MSVC
#if !defined(_MSC_VER)
        using std::copysign;
#endif
        using std::fdim;
        using std::fmax;
        using std::fmin;
        using std::fmod;
        using std::hypot;
        using std::pow;

        using std::fma;

        using std::fpclassify;
        using std::isfinite;
        using std::isinf;
        using std::isnan;

        // Overload isinf, isnan and isfinite for complex datatypes,
        // following the Python specification:
        template <class T>
        inline bool isinf(const std::complex<T>& c)
        {
            return std::isinf(std::real(c)) || std::isinf(std::imag(c));
        }

        template <class T>
        inline bool isnan(const std::complex<T>& c)
        {
            return std::isnan(std::real(c)) || std::isnan(std::imag(c));
        }

        template <class T>
        inline bool isfinite(const std::complex<T>& c)
        {
            return !isinf(c) && !isnan(c);
        }

        // The following specializations are needed to avoid 'ambiguous overload' errors,
        // whereas 'unsigned char' and 'unsigned short' are automatically converted to 'int'.
        // we're still adding those functions to silence warnings
        XTENSOR_UNSIGNED_ABS_FUNC(unsigned char)
        XTENSOR_UNSIGNED_ABS_FUNC(unsigned short)
        XTENSOR_UNSIGNED_ABS_FUNC(unsigned int)
        XTENSOR_UNSIGNED_ABS_FUNC(unsigned long)
        XTENSOR_UNSIGNED_ABS_FUNC(unsigned long long)

#ifdef _WIN32
        XTENSOR_INT_SPECIALIZATION(isinf, false);
        XTENSOR_INT_SPECIALIZATION(isnan, false);
        XTENSOR_INT_SPECIALIZATION(isfinite, true);
#endif

        XTENSOR_UNARY_MATH_FUNCTOR_COMPLEX_REDUCING(abs);

        XTENSOR_UNARY_MATH_FUNCTOR(fabs);
        XTENSOR_BINARY_MATH_FUNCTOR(fmod);
        XTENSOR_BINARY_MATH_FUNCTOR(remainder);
        XTENSOR_TERNARY_MATH_FUNCTOR(fma);
        XTENSOR_BINARY_MATH_FUNCTOR(fmax);
        XTENSOR_BINARY_MATH_FUNCTOR(fmin);
        XTENSOR_BINARY_MATH_FUNCTOR(fdim);
        XTENSOR_UNARY_MATH_FUNCTOR(exp);
        XTENSOR_UNARY_MATH_FUNCTOR(exp2);
        XTENSOR_UNARY_MATH_FUNCTOR(expm1);
        XTENSOR_UNARY_MATH_FUNCTOR(log);
        XTENSOR_UNARY_MATH_FUNCTOR(log10);
        XTENSOR_UNARY_MATH_FUNCTOR(log2);
        XTENSOR_UNARY_MATH_FUNCTOR(log1p);
        XTENSOR_BINARY_MATH_FUNCTOR(pow);
        XTENSOR_UNARY_MATH_FUNCTOR(sqrt);
        XTENSOR_UNARY_MATH_FUNCTOR(cbrt);
        XTENSOR_BINARY_MATH_FUNCTOR(hypot);
        XTENSOR_UNARY_MATH_FUNCTOR(sin);
        XTENSOR_UNARY_MATH_FUNCTOR(cos);
        XTENSOR_UNARY_MATH_FUNCTOR(tan);
        XTENSOR_UNARY_MATH_FUNCTOR(asin);
        XTENSOR_UNARY_MATH_FUNCTOR(acos);
        XTENSOR_UNARY_MATH_FUNCTOR(atan);
        XTENSOR_BINARY_MATH_FUNCTOR(atan2);
        XTENSOR_UNARY_MATH_FUNCTOR(sinh);
        XTENSOR_UNARY_MATH_FUNCTOR(cosh);
        XTENSOR_UNARY_MATH_FUNCTOR(tanh);
        XTENSOR_UNARY_MATH_FUNCTOR(asinh);
        XTENSOR_UNARY_MATH_FUNCTOR(acosh);
        XTENSOR_UNARY_MATH_FUNCTOR(atanh);
        XTENSOR_UNARY_MATH_FUNCTOR(erf);
        XTENSOR_UNARY_MATH_FUNCTOR(erfc);
        XTENSOR_UNARY_MATH_FUNCTOR(tgamma);
        XTENSOR_UNARY_MATH_FUNCTOR(lgamma);
        XTENSOR_UNARY_MATH_FUNCTOR(ceil);
        XTENSOR_UNARY_MATH_FUNCTOR(floor);
        XTENSOR_UNARY_MATH_FUNCTOR(trunc);
        XTENSOR_UNARY_MATH_FUNCTOR(round);
        XTENSOR_UNARY_MATH_FUNCTOR(nearbyint);
        XTENSOR_UNARY_MATH_FUNCTOR(rint);
        XTENSOR_UNARY_BOOL_FUNCTOR(isfinite);
        XTENSOR_UNARY_BOOL_FUNCTOR(isinf);
        XTENSOR_UNARY_BOOL_FUNCTOR(isnan);
    }

#undef XTENSOR_UNARY_MATH_FUNCTOR
#undef XTENSOR_UNARY_BOOL_FUNCTOR
#undef XTENSOR_UNARY_MATH_FUNCTOR_IMPL
#undef XTENSOR_BINARY_MATH_FUNCTOR
#undef XTENSOR_BINARY_BOOL_FUNCTOR
#undef XTENSOR_BINARY_MATH_FUNCTOR_IMPL
#undef XTENSOR_TERNARY_MATH_FUNCTOR
#undef XTENSOR_TERNARY_BOOL_FUNCTOR
#undef XTENSOR_TERNARY_MATH_FUNCTOR_IMPL
#undef XTENSOR_UNARY_MATH_FUNCTOR_COMPLEX_REDUCING
#undef XTENSOR_UNSIGNED_ABS_FUNC

#define XTENSOR_REDUCER_FUNCTION(NAME, FUNCTOR, RESULT_TYPE, INIT)                                 \
    template <class T = void,                                                                      \
              class E,                                                                             \
              class X,                                                                             \
              class EVS = DEFAULT_STRATEGY_REDUCERS,                                               \
              XTL_REQUIRES(xtl::negation<is_reducer_options<X>>,                                   \
                           xtl::negation<std::is_integral<X>>)>                                    \
    inline auto NAME(E&& e, X&& axes, EVS es = EVS())                                              \
    {                                                                                              \
        using result_type = std::conditional_t<std::is_same<T, void>::value, RESULT_TYPE, T>;      \
        using functor_type = FUNCTOR<result_type>;                                                 \
        using init_value_fct = xt::const_value<result_type /*, INIT*/>;                            \
        return xt::reduce(make_xreducer_functor(functor_type(), init_value_fct(INIT)),             \
                          std::forward<E>(e),                                                      \
                          std::forward<X>(axes),                                                   \
                          es);                                                                     \
    }                                                                                              \
                                                                                                   \
    template <class T = void,                                                                      \
              class E,                                                                             \
              class X,                                                                             \
              class EVS = DEFAULT_STRATEGY_REDUCERS,                                               \
              XTL_REQUIRES(xtl::negation<is_reducer_options<X>>, std::is_integral<X>)>             \
    inline auto NAME(E&& e, X axis, EVS es = EVS())                                                \
    {                                                                                              \
        return NAME(std::forward<E>(e), { axis }, es);                                             \
    }                                                                                              \
                                                                                                   \
    template <class T = void,                                                                      \
              class E,                                                                             \
              class EVS = DEFAULT_STRATEGY_REDUCERS,                                               \
              XTL_REQUIRES(is_reducer_options<EVS>)>                                               \
    inline auto NAME(E&& e, EVS es = EVS())                                                        \
    {                                                                                              \
        using result_type = std::conditional_t<std::is_same<T, void>::value, RESULT_TYPE, T>;      \
        using functor_type = FUNCTOR<result_type>;                                                 \
        using init_value_fct = xt::const_value<result_type /*, INIT*/>;                            \
        return xt::reduce(                                                                         \
            make_xreducer_functor(functor_type(), init_value_fct(INIT)), std::forward<E>(e), es);  \
    }

#define XTENSOR_OLD_CLANG_REDUCER(NAME, FUNCTOR, RESULT_TYPE, INIT)                                \
    template <class T = void, class E, class I, class EVS = DEFAULT_STRATEGY_REDUCERS>             \
    inline auto NAME(E&& e, std::initializer_list<I> axes, EVS es = EVS())                         \
    {                                                                                              \
        using result_type = std::conditional_t<std::is_same<T, void>::value, RESULT_TYPE, T>;      \
        using functor_type = FUNCTOR<result_type>;                                                 \
        using init_value_fct = xt::const_value<result_type /*, INIT*/>;                            \
        return xt::reduce(make_xreducer_functor(functor_type(), init_value_fct(INIT)),             \
                          std::forward<E>(e),                                                      \
                          axes,                                                                    \
                          es);                                                                     \
    }

#define XTENSOR_MODERN_CLANG_REDUCER(NAME, FUNCTOR, RESULT_TYPE, INIT)                             \
    template <class T = void,                                                                      \
              class E,                                                                             \
              class I,                                                                             \
              std::size_t N,                                                                       \
              class EVS = DEFAULT_STRATEGY_REDUCERS>                                               \
    inline auto NAME(E&& e, const I(&axes)[N], EVS es = EVS())                                     \
    {                                                                                              \
        using result_type = std::conditional_t<std::is_same<T, void>::value, RESULT_TYPE, T>;      \
        using functor_type = FUNCTOR<result_type>;                                                 \
        using init_value_fct = xt::const_value<result_type /*, INIT*/>;                            \
        return xt::reduce(make_xreducer_functor(functor_type(), init_value_fct(INIT)),             \
                          std::forward<E>(e),                                                      \
                          axes,                                                                    \
                          es);                                                                     \
    }

    /*******************
     * basic functions *
     *******************/

    /**
     * @defgroup basic_functions Basic functions
     */

    /**
     * @ingroup basic_functions
     * @brief Absolute value function.
     *
     * Returns an \ref xfunction for the element-wise absolute value
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto abs(E&& e) noexcept -> detail::xfunction_type_t<math::abs_fun, E>
    {
        return detail::make_xfunction<math::abs_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup basic_functions
     * @brief Absolute value function.
     *
     * Returns an \ref xfunction for the element-wise absolute value
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto fabs(E&& e) noexcept -> detail::xfunction_type_t<math::fabs_fun, E>
    {
        return detail::make_xfunction<math::fabs_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup basic_functions
     * @brief Remainder of the floating point division operation.
     *
     * Returns an \ref xfunction for the element-wise remainder of
     * the floating point division operation <em>e1 / e2</em>.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fmod(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<math::fmod_fun, E1, E2>
    {
        return detail::make_xfunction<math::fmod_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Signed remainder of the division operation.
     *
     * Returns an \ref xfunction for the element-wise signed remainder
     * of the floating point division operation <em>e1 / e2</em>.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto remainder(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::remainder_fun, E1, E2>
    {
        return detail::make_xfunction<math::remainder_fun>(std::forward<E1>(e1),
                                                           std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Fused multiply-add operation.
     *
     * Returns an \ref xfunction for <em>e1 * e2 + e3</em> as if
     * to infinite precision and rounded only once to fit the result type.
     * @param e1 an \ref xfunction or a scalar
     * @param e2 an \ref xfunction or a scalar
     * @param e3 an \ref xfunction or a scalar
     * @return an \ref xfunction
     * @note e1, e2 and e3 can't be scalars every three.
     */
    template <class E1, class E2, class E3>
    inline auto fma(E1&& e1, E2&& e2, E3&& e3) noexcept
        -> detail::xfunction_type_t<math::fma_fun, E1, E2, E3>
    {
        return detail::make_xfunction<math::fma_fun>(
            std::forward<E1>(e1), std::forward<E2>(e2), std::forward<E3>(e3));
    }

    /**
     * @ingroup basic_functions
     * @brief Maximum function.
     *
     * Returns an \ref xfunction for the element-wise maximum
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fmax(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<math::fmax_fun, E1, E2>
    {
        return detail::make_xfunction<math::fmax_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Minimum function.
     *
     * Returns an \ref xfunction for the element-wise minimum
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fmin(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<math::fmin_fun, E1, E2>
    {
        return detail::make_xfunction<math::fmin_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Positive difference function.
     *
     * Returns an \ref xfunction for the element-wise positive
     * difference of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fdim(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<math::fdim_fun, E1, E2>
    {
        return detail::make_xfunction<math::fdim_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    namespace math
    {
        template <class T = void>
        struct minimum
        {
            template <class A1, class A2>
            constexpr auto operator()(const A1& t1, const A2& t2) const noexcept
            {
                return xtl::select(t1 < t2, t1, t2);
            }

            template <class A1, class A2>
            constexpr auto simd_apply(const A1& t1, const A2& t2) const noexcept
            {
                return xt_simd::select(t1 < t2, t1, t2);
            }
        };

        template <class T = void>
        struct maximum
        {
            template <class A1, class A2>
            constexpr auto operator()(const A1& t1, const A2& t2) const noexcept
            {
                return xtl::select(t1 > t2, t1, t2);
            }

            template <class A1, class A2>
            constexpr auto simd_apply(const A1& t1, const A2& t2) const noexcept
            {
                return xt_simd::select(t1 > t2, t1, t2);
            }
        };

        struct clamp_fun
        {
            template <class A1, class A2, class A3>
            constexpr auto operator()(const A1& v, const A2& lo, const A3& hi) const
            {
                return xtl::select(v < lo, lo, xtl::select(hi < v, hi, v));
            }

            template <class A1, class A2, class A3>
            constexpr auto simd_apply(const A1& v, const A2& lo, const A3& hi) const
            {
                return xt_simd::select(v < lo, lo, xt_simd::select(hi < v, hi, v));
            }
        };
    }

    /**
     * @ingroup basic_functions
     * @brief Elementwise maximum
     *
     * Returns an \ref xfunction for the element-wise
     * maximum between e1 and e2.
     * @param e1 an \ref xexpression
     * @param e2 an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto maximum(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::maximum<void>, E1, E2>
    {
        return detail::make_xfunction<math::maximum<void>>(std::forward<E1>(e1),
                                                           std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Elementwise minimum
     *
     * Returns an \ref xfunction for the element-wise
     * minimum between e1 and e2.
     * @param e1 an \ref xexpression
     * @param e2 an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto minimum(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::minimum<void>, E1, E2>
    {
        return detail::make_xfunction<math::minimum<void>>(std::forward<E1>(e1),
                                                           std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Maximum element along given axis.
     *
     * Returns an \ref xreducer for the maximum of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the maximum is found (optional)
     * @param es evaluation strategy of the reducer
     * @return an \ref xreducer
     */
    XTENSOR_REDUCER_FUNCTION(amax,
                             math::maximum,
                             typename std::decay_t<E>::value_type,
                             std::numeric_limits<xvalue_type_t<std::decay_t<E>>>::lowest())
#ifdef X_OLD_CLANG
    XTENSOR_OLD_CLANG_REDUCER(amax,
                              math::maximum,
                              typename std::decay_t<E>::value_type,
                              std::numeric_limits<xvalue_type_t<std::decay_t<E>>>::lowest())
#else
    XTENSOR_MODERN_CLANG_REDUCER(amax,
                                 math::maximum,
                                 typename std::decay_t<E>::value_type,
                                 std::numeric_limits<xvalue_type_t<std::decay_t<E>>>::lowest())
#endif

    /**
     * @ingroup basic_functions
     * @brief Minimum element along given axis.
     *
     * Returns an \ref xreducer for the minimum of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the minimum is found (optional)
     * @param es evaluation strategy of the reducer
     * @return an \ref xreducer
     */
    XTENSOR_REDUCER_FUNCTION(amin,
                             math::minimum,
                             typename std::decay_t<E>::value_type,
                             std::numeric_limits<xvalue_type_t<std::decay_t<E>>>::max())
#ifdef X_OLD_CLANG
    XTENSOR_OLD_CLANG_REDUCER(amin,
                              math::minimum,
                              typename std::decay_t<E>::value_type,
                              std::numeric_limits<xvalue_type_t<std::decay_t<E>>>::max())
#else
    XTENSOR_MODERN_CLANG_REDUCER(amin,
                                 math::minimum,
                                 typename std::decay_t<E>::value_type,
                                 std::numeric_limits<xvalue_type_t<std::decay_t<E>>>::max())
#endif

    /**
     * @ingroup basic_functions
     * @brief Clip values between hi and lo
     *
     * Returns an \ref xfunction for the element-wise clipped
     * values between lo and hi
     * @param e1 an \ref xexpression or a scalar
     * @param lo a scalar
     * @param hi a scalar
     *
     * @return a \ref xfunction
     */
    template <class E1, class E2, class E3>
    inline auto clip(E1&& e1, E2&& lo, E3&& hi) noexcept
        -> detail::xfunction_type_t<math::clamp_fun, E1, E2, E3>
    {
        return detail::make_xfunction<math::clamp_fun>(
            std::forward<E1>(e1), std::forward<E2>(lo), std::forward<E3>(hi));
    }

    namespace math
    {
        template <class T>
        struct sign_impl
        {
            template <class XT = T>
            static constexpr std::enable_if_t<std::is_signed<XT>::value, T> run(T x)
            {
                return std::isnan(x) ? std::numeric_limits<T>::quiet_NaN()
                                     : x == 0 ? T(copysign(T(0), x)) : T(copysign(T(1), x));
            }

            template <class XT = T>
            static constexpr std::enable_if_t<xtl::is_complex<XT>::value, T> run(T x)
            {
                return T(sign_impl<typename T::value_type>::run(
                             (x.real() != typename T::value_type(0)) ? x.real() : x.imag()),
                         0);
            }

            template <class XT = T>
            static constexpr std::enable_if_t<std::is_unsigned<XT>::value, T> run(T x)
            {
                return T(x > T(0));
            }
        };

        struct sign_fun
        {
            template <class T>
            constexpr auto operator()(const T& x) const
            {
                return sign_impl<T>::run(x);
            }
        };
    }

    /**
     * @ingroup basic_functions
     * @brief Returns an element-wise indication of the sign of a number
     *
     * If the number is positive, returns +1. If negative, -1. If the number
     * is zero, returns 0.
     *
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sign(E&& e) noexcept -> detail::xfunction_type_t<math::sign_fun, E>
    {
        return detail::make_xfunction<math::sign_fun>(std::forward<E>(e));
    }

    /*************************
     * exponential functions *
     *************************/

    /**
     * @defgroup exp_functions Exponential functions
     */

    /**
     * @ingroup exp_functions
     * @brief Natural exponential function.
     *
     * Returns an \ref xfunction for the element-wise natural
     * exponential of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto exp(E&& e) noexcept -> detail::xfunction_type_t<math::exp_fun, E>
    {
        return detail::make_xfunction<math::exp_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Base 2 exponential function.
     *
     * Returns an \ref xfunction for the element-wise base 2
     * exponential of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto exp2(E&& e) noexcept -> detail::xfunction_type_t<math::exp2_fun, E>
    {
        return detail::make_xfunction<math::exp2_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Natural exponential minus one function.
     *
     * Returns an \ref xfunction for the element-wise natural
     * exponential of \em e, minus 1.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto expm1(E&& e) noexcept -> detail::xfunction_type_t<math::expm1_fun, E>
    {
        return detail::make_xfunction<math::expm1_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Natural logarithm function.
     *
     * Returns an \ref xfunction for the element-wise natural
     * logarithm of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log(E&& e) noexcept -> detail::xfunction_type_t<math::log_fun, E>
    {
        return detail::make_xfunction<math::log_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Base 10 logarithm function.
     *
     * Returns an \ref xfunction for the element-wise base 10
     * logarithm of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log10(E&& e) noexcept -> detail::xfunction_type_t<math::log10_fun, E>
    {
        return detail::make_xfunction<math::log10_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Base 2 logarithm function.
     *
     * Returns an \ref xfunction for the element-wise base 2
     * logarithm of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log2(E&& e) noexcept -> detail::xfunction_type_t<math::log2_fun, E>
    {
        return detail::make_xfunction<math::log2_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Natural logarithm of one plus function.
     *
     * Returns an \ref xfunction for the element-wise natural
     * logarithm of \em e, plus 1.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log1p(E&& e) noexcept -> detail::xfunction_type_t<math::log1p_fun, E>
    {
        return detail::make_xfunction<math::log1p_fun>(std::forward<E>(e));
    }

    /*******************
     * power functions *
     *******************/

    /**
     * @defgroup pow_functions Power functions
     */

    /**
     * @ingroup pow_functions
     * @brief Power function.
     *
     * Returns an \ref xfunction for the element-wise value of
     * of \em e1 raised to the power \em e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto pow(E1&& e1, E2&& e2) noexcept -> detail::xfunction_type_t<math::pow_fun, E1, E2>
    {
        return detail::make_xfunction<math::pow_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    namespace detail
    {
        template <class F, class... T, typename = decltype(std::declval<F>()(std::declval<T>()...))>
        std::true_type supports_test(const F&, const T&...);
        std::false_type supports_test(...);

        template <class... T>
        struct supports;

        template <class F, class... T>
        struct supports<F(T...)> : decltype(supports_test(std::declval<F>(), std::declval<T>()...))
        {
        };

        template <class F>
        struct lambda_adapt
        {
            explicit lambda_adapt(F&& lmbd)
                : m_lambda(std::move(lmbd))
            {
            }

            template <class... T>
            auto operator()(T... args) const
            {
                return m_lambda(args...);
            }

            template <class... T, XTL_REQUIRES(detail::supports<F(T...)>)>
            auto simd_apply(T... args) const
            {
                return m_lambda(args...);
            }

            F m_lambda;
        };
    }

    /**
     * Create a xfunction from a lambda
     *
     * This function can be used to easily create performant xfunctions from lambdas:
     *
     * \code{cpp}
     * template <class E1>
     * inline auto square(E1&& e1) noexcept
     * {
     *     auto fnct = [](auto x) -> decltype(x * x) {
     *         return x * x;
     *     };
     *     return make_lambda_xfunction(std::move(fnct), std::forward<E1>(e1));
     * }
     * \endcode
     *
     * Lambda function allow the reusal of a single arguments in multiple places (otherwise
     * only correctly possible when using xshared_expressions). ``auto`` lambda functions are
     * automatically vectorized with ``xsimd`` if possible (note that the trailing
     * ``-> decltype(...)`` is mandatory for the feature detection to work).
     *
     * @param lambda the lambda to be vectorized
     * @param args forwarded arguments
     *
     * @return lazy xfunction
     */
    template <class F, class... E>
    inline auto make_lambda_xfunction(F&& lambda, E&&... args)
    {
        using xfunction_type = typename detail::xfunction_type<detail::lambda_adapt<F>, E...>::type;
        return xfunction_type(detail::lambda_adapt<F>(std::forward<F>(lambda)),
                              std::forward<E>(args)...);
    }


#define XTENSOR_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

// Workaround for MSVC 2015 & GCC 4.9
#if (defined(_MSC_VER) && _MSC_VER < 1910) || (defined(__GNUC__) && GCC_VERSION < 49999)
#define XTENSOR_DISABLE_LAMBDA_FCT
#endif

#ifdef XTENSOR_DISABLE_LAMBDA_FCT
    struct square_fct
    {
        template <class T>
        auto operator()(T x) const -> decltype(x * x)
        {
            return x * x;
        }
    };

    struct cube_fct
    {
        template <class T>
        auto operator()(T x) const -> decltype(x * x * x)
        {
            return x * x * x;
        }
    };
#endif

    /**
     * @ingroup pow_functions
     * @brief Square power function, equivalent to e1 * e1.
     *
     * Returns an \ref xfunction for the element-wise value of
     * of \em e1 * \em e1.
     * @param e1 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1>
    inline auto square(E1&& e1) noexcept
    {
#ifdef XTENSOR_DISABLE_LAMBDA_FCT
        return make_lambda_xfunction(square_fct{}, std::forward<E1>(e1));
#else
        auto fnct = [](auto x) -> decltype(x * x) { return x * x; };
        return make_lambda_xfunction(std::move(fnct), std::forward<E1>(e1));
#endif
    }

    /**
     * @ingroup pow_functions
     * @brief Cube power function, equivalent to e1 * e1 * e1.
     *
     * Returns an \ref xfunction for the element-wise value of
     * of \em e1 * \em e1.
     * @param e1 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1>
    inline auto cube(E1&& e1) noexcept
    {
#ifdef XTENSOR_DISABLE_LAMBDA_FCT
        return make_lambda_xfunction(cube_fct{}, std::forward<E1>(e1));
#else
        auto fnct = [](auto x) -> decltype(x * x * x) { return x * x * x; };
        return make_lambda_xfunction(std::move(fnct), std::forward<E1>(e1));
#endif
    }

#undef XTENSOR_GCC_VERSION
#undef XTENSOR_DISABLE_LAMBDA_FCT

    namespace detail
    {
        // Thanks to Matt Pharr in http://pbrt.org/hair.pdf
        template <std::size_t N>
        struct pow_impl;

        template <std::size_t N>
        struct pow_impl
        {
            template <class T>
            auto operator()(T v) const -> decltype(v * v)
            {
                T temp = pow_impl<N / 2>{}(v);
                return temp * temp * pow_impl<N & 1>{}(v);
            }
        };

        template <>
        struct pow_impl<1>
        {
            template <class T>
            auto operator()(T v) const -> T
            {
                return v;
            }
        };

        template <>
        struct pow_impl<0>
        {
            template <class T>
            auto operator()(T /*v*/) const -> T
            {
                return T(1);
            }
        };
    }

    /**
     * @ingroup pow_functions
     * @brief Integer power function.
     *
     * Returns an \ref xfunction for the element-wise power of e1 to
     * an integral constant.
     *
     * Instead of computing the power by using the (expensive) logarithm, this function
     * computes the power in a number of straight-forward multiplication steps. This function
     * is therefore much faster (even for high N) than the generic pow-function.
     *
     * For example, `e1^20` can be expressed as `(((e1^2)^2)^2)^2*(e1^2)^2`, which is just 5
     * multiplications.
     *
     * @param e an \ref xexpression
     * @tparam N the exponent (has to be positive integer)
     * @return an \ref xfunction
     */
    template <std::size_t N, class E>
    inline auto pow(E&& e) noexcept
    {
        static_assert(N > 0, "integer power cannot be negative");
        return make_lambda_xfunction(detail::pow_impl<N>{}, std::forward<E>(e));
    }

    /**
     * @ingroup pow_functions
     * @brief Square root function.
     *
     * Returns an \ref xfunction for the element-wise square
     * root of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sqrt(E&& e) noexcept -> detail::xfunction_type_t<math::sqrt_fun, E>
    {
        return detail::make_xfunction<math::sqrt_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup pow_functions
     * @brief Cubic root function.
     *
     * Returns an \ref xfunction for the element-wise cubic
     * root of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto cbrt(E&& e) noexcept -> detail::xfunction_type_t<math::cbrt_fun, E>
    {
        return detail::make_xfunction<math::cbrt_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup pow_functions
     * @brief Hypotenuse function.
     *
     * Returns an \ref xfunction for the element-wise square
     * root of the sum of the square of \em e1 and \em e2, avoiding
     * overflow and underflow at intermediate stages of computation.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto hypot(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::hypot_fun, E1, E2>
    {
        return detail::make_xfunction<math::hypot_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /***************************
     * trigonometric functions *
     ***************************/

    /**
     * @defgroup trigo_functions Trigonometric function
     */

    /**
     * @ingroup trigo_functions
     * @brief Sine function.
     *
     * Returns an \ref xfunction for the element-wise sine
     * of \em e (measured in radians).
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sin(E&& e) noexcept -> detail::xfunction_type_t<math::sin_fun, E>
    {
        return detail::make_xfunction<math::sin_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Cosine function.
     *
     * Returns an \ref xfunction for the element-wise cosine
     * of \em e (measured in radians).
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto cos(E&& e) noexcept -> detail::xfunction_type_t<math::cos_fun, E>
    {
        return detail::make_xfunction<math::cos_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Tangent function.
     *
     * Returns an \ref xfunction for the element-wise tangent
     * of \em e (measured in radians).
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto tan(E&& e) noexcept -> detail::xfunction_type_t<math::tan_fun, E>
    {
        return detail::make_xfunction<math::tan_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Arcsine function.
     *
     * Returns an \ref xfunction for the element-wise arcsine
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto asin(E&& e) noexcept -> detail::xfunction_type_t<math::asin_fun, E>
    {
        return detail::make_xfunction<math::asin_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Arccosine function.
     *
     * Returns an \ref xfunction for the element-wise arccosine
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto acos(E&& e) noexcept -> detail::xfunction_type_t<math::acos_fun, E>
    {
        return detail::make_xfunction<math::acos_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Arctangent function.
     *
     * Returns an \ref xfunction for the element-wise arctangent
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto atan(E&& e) noexcept -> detail::xfunction_type_t<math::atan_fun, E>
    {
        return detail::make_xfunction<math::atan_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Artangent function, using signs to determine quadrants.
     *
     * Returns an \ref xfunction for the element-wise arctangent
     * of <em>e1 / e2</em>, using the signs of arguments to determine the
     * correct quadrant.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto atan2(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::atan2_fun, E1, E2>
    {
        return detail::make_xfunction<math::atan2_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /************************
     * hyperbolic functions *
     ************************/

    /**
     * @defgroup hyper_functions Hyperbolic functions
     */

    /**
     * @ingroup hyper_functions
     * @brief Hyperbolic sine function.
     *
     * Returns an \ref xfunction for the element-wise hyperbolic
     * sine of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sinh(E&& e) noexcept -> detail::xfunction_type_t<math::sinh_fun, E>
    {
        return detail::make_xfunction<math::sinh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Hyperbolic cosine function.
     *
     * Returns an \ref xfunction for the element-wise hyperbolic
     * cosine of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto cosh(E&& e) noexcept -> detail::xfunction_type_t<math::cosh_fun, E>
    {
        return detail::make_xfunction<math::cosh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Hyperbolic tangent function.
     *
     * Returns an \ref xfunction for the element-wise hyperbolic
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto tanh(E&& e) noexcept -> detail::xfunction_type_t<math::tanh_fun, E>
    {
        return detail::make_xfunction<math::tanh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Inverse hyperbolic sine function.
     *
     * Returns an \ref xfunction for the element-wise inverse hyperbolic
     * sine of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto asinh(E&& e) noexcept -> detail::xfunction_type_t<math::asinh_fun, E>
    {
        return detail::make_xfunction<math::asinh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Inverse hyperbolic cosine function.
     *
     * Returns an \ref xfunction for the element-wise inverse hyperbolic
     * cosine of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto acosh(E&& e) noexcept -> detail::xfunction_type_t<math::acosh_fun, E>
    {
        return detail::make_xfunction<math::acosh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Inverse hyperbolic tangent function.
     *
     * Returns an \ref xfunction for the element-wise inverse hyperbolic
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto atanh(E&& e) noexcept -> detail::xfunction_type_t<math::atanh_fun, E>
    {
        return detail::make_xfunction<math::atanh_fun>(std::forward<E>(e));
    }

    /*****************************
     * error and gamma functions *
     *****************************/

    /**
     * @defgroup err_functions Error and gamma functions
     */

    /**
     * @ingroup err_functions
     * @brief Error function.
     *
     * Returns an \ref xfunction for the element-wise error function
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto erf(E&& e) noexcept -> detail::xfunction_type_t<math::erf_fun, E>
    {
        return detail::make_xfunction<math::erf_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup err_functions
     * @brief Complementary error function.
     *
     * Returns an \ref xfunction for the element-wise complementary
     * error function of \em e, whithout loss of precision for large argument.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto erfc(E&& e) noexcept -> detail::xfunction_type_t<math::erfc_fun, E>
    {
        return detail::make_xfunction<math::erfc_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup err_functions
     * @brief Gamma function.
     *
     * Returns an \ref xfunction for the element-wise gamma function
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto tgamma(E&& e) noexcept -> detail::xfunction_type_t<math::tgamma_fun, E>
    {
        return detail::make_xfunction<math::tgamma_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup err_functions
     * @brief Natural logarithm of the gamma function.
     *
     * Returns an \ref xfunction for the element-wise logarithm of
     * the asbolute value fo the gamma function of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto lgamma(E&& e) noexcept -> detail::xfunction_type_t<math::lgamma_fun, E>
    {
        return detail::make_xfunction<math::lgamma_fun>(std::forward<E>(e));
    }

    /*********************************************
     * nearest integer floating point operations *
     *********************************************/

    /**
     * @defgroup nearint_functions Nearest integer floating point operations
     */

    /**
     * @ingroup nearint_functions
     * @brief ceil function.
     *
     * Returns an \ref xfunction for the element-wise smallest integer value
     * not less than \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto ceil(E&& e) noexcept -> detail::xfunction_type_t<math::ceil_fun, E>
    {
        return detail::make_xfunction<math::ceil_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup nearint_functions
     * @brief floor function.
     *
     * Returns an \ref xfunction for the element-wise smallest integer value
     * not greater than \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto floor(E&& e) noexcept -> detail::xfunction_type_t<math::floor_fun, E>
    {
        return detail::make_xfunction<math::floor_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup nearint_functions
     * @brief trunc function.
     *
     * Returns an \ref xfunction for the element-wise nearest integer not greater
     * in magnitude than \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto trunc(E&& e) noexcept -> detail::xfunction_type_t<math::trunc_fun, E>
    {
        return detail::make_xfunction<math::trunc_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup nearint_functions
     * @brief round function.
     *
     * Returns an \ref xfunction for the element-wise nearest integer value
     * to \em e, rounding halfway cases away from zero, regardless of the
     * current rounding mode.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto round(E&& e) noexcept -> detail::xfunction_type_t<math::round_fun, E>
    {
        return detail::make_xfunction<math::round_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup nearint_functions
     * @brief nearbyint function.
     *
     * Returns an \ref xfunction for the element-wise rounding of \em e to integer
     * values in floating point format, using the current rounding mode. nearbyint
     * never raises FE_INEXACT error.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto nearbyint(E&& e) noexcept -> detail::xfunction_type_t<math::nearbyint_fun, E>
    {
        return detail::make_xfunction<math::nearbyint_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup nearint_functions
     * @brief rint function.
     *
     * Returns an \ref xfunction for the element-wise rounding of \em e to integer
     * values in floating point format, using the current rounding mode. Contrary
     * to nearbyint, rint may raise FE_INEXACT error.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto rint(E&& e) noexcept -> detail::xfunction_type_t<math::rint_fun, E>
    {
        return detail::make_xfunction<math::rint_fun>(std::forward<E>(e));
    }

    /****************************
     * classification functions *
     ****************************/

    /**
     * @defgroup classif_functions Classification functions
     */

    /**
     * @ingroup classif_functions
     * @brief finite value check
     *
     * Returns an \ref xfunction for the element-wise finite value check
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto isfinite(E&& e) noexcept -> detail::xfunction_type_t<math::isfinite_fun, E>
    {
        return detail::make_xfunction<math::isfinite_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup classif_functions
     * @brief infinity check
     *
     * Returns an \ref xfunction for the element-wise infinity check
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto isinf(E&& e) noexcept -> detail::xfunction_type_t<math::isinf_fun, E>
    {
        return detail::make_xfunction<math::isinf_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup classif_functions
     * @brief NaN check
     *
     * Returns an \ref xfunction for the element-wise NaN check
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto isnan(E&& e) noexcept -> detail::xfunction_type_t<math::isnan_fun, E>
    {
        return detail::make_xfunction<math::isnan_fun>(std::forward<E>(e));
    }

    namespace detail
    {
        template <class FUNCTOR, class T, std::size_t... Is>
        inline auto get_functor(T&& args, std::index_sequence<Is...>)
        {
            return FUNCTOR(std::get<Is>(args)...);
        }

        template <class F, class... A, class... E>
        inline auto make_xfunction(std::tuple<A...>&& f_args, E&&... e) noexcept
        {
            using functor_type = F;
            using expression_tag = xexpression_tag_t<E...>;
            using type = select_xfunction_expression_t<expression_tag,
                                                       functor_type,
                                                       const_xclosure_t<E>...>;
            auto functor = get_functor<functor_type>(std::forward<std::tuple<A...>>(f_args),
                                                     std::make_index_sequence<sizeof...(A)>{});
            return type(std::move(functor), std::forward<E>(e)...);
        }

        struct isclose
        {
            using result_type = bool;
            isclose(double rtol, double atol, bool equal_nan)
                : m_rtol(rtol)
                , m_atol(atol)
                , m_equal_nan(equal_nan)
            {
            }

            template <class A1, class A2>
            bool operator()(const A1& a, const A2& b) const
            {
                using internal_type = xtl::promote_type_t<A1, A2, double>;
                if (math::isnan(a) && math::isnan(b))
                {
                    return m_equal_nan;
                }
                if (math::isinf(a) && math::isinf(b))
                {
                    // check for both infinity signs equal
                    return a == b;
                }
                auto d = math::abs(internal_type(a) - internal_type(b));
                return d <= m_atol
                       || d <= m_rtol
                                   * double((std::max)(math::abs(internal_type(a)),
                                                       math::abs(internal_type(b))));
            }

        private:

            double m_rtol;
            double m_atol;
            bool m_equal_nan;
        };
    }

    /**
     * @ingroup classif_functions
     * @brief Element-wise closeness detection
     *
     * Returns an \ref xfunction that evaluates to
     * true if the elements in ``e1`` and ``e2`` are close to each other
     * according to parameters ``atol`` and ``rtol``.
     * The equation is: ``std::abs(a - b) <= (m_atol + m_rtol * std::abs(b))``.
     * @param e1 input array to compare
     * @param e2 input array to compare
     * @param rtol the relative tolerance parameter (default 1e-05)
     * @param atol the absolute tolerance parameter (default 1e-08)
     * @param equal_nan if true, isclose returns true if both elements of e1 and e2 are NaN
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto isclose(
        E1&& e1, E2&& e2, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false) noexcept
    {
        return detail::make_xfunction<detail::isclose>(
            std::make_tuple(rtol, atol, equal_nan), std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup classif_functions
     * @brief Check if all elements in \em e1 are close to the
     * corresponding elements in \em e2.
     *
     * Returns true if all elements in ``e1`` and ``e2`` are close to each other
     * according to parameters ``atol`` and ``rtol``.
     * @param e1 input array to compare
     * @param e2 input arrays to compare
     * @param rtol the relative tolerance parameter (default 1e-05)
     * @param atol the absolute tolerance parameter (default 1e-08)
     * @return a boolean
     */
    template <class E1, class E2>
    inline auto allclose(E1&& e1, E2&& e2, double rtol = 1e-05, double atol = 1e-08) noexcept
    {
        return xt::all(isclose(std::forward<E1>(e1), std::forward<E2>(e2), rtol, atol));
    }

    /**********************
     * Reducing functions *
     **********************/


    /**
     * @defgroup  red_functions reducing functions
     */

    /**
     * @ingroup red_functions
     * @brief Sum of elements over given axes.
     *
     * Returns an \ref xreducer for the sum of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the sum is performed (optional)
     * @param es evaluation strategy of the reducer
     * @return an \ref xreducer
     */
    XTENSOR_REDUCER_FUNCTION(sum,
                             std::plus,
                             xtl::big_promote_type_t<typename std::decay_t<E>::value_type>,
                             0)
#ifdef X_OLD_CLANG
    XTENSOR_OLD_CLANG_REDUCER(sum,
                              std::plus,
                              xtl::big_promote_type_t<typename std::decay_t<E>::value_type>,
                              0)
#else
    XTENSOR_MODERN_CLANG_REDUCER(sum,
                                 std::plus,
                                 xtl::big_promote_type_t<typename std::decay_t<E>::value_type>,
                                 0)
#endif

    /**
     * @ingroup red_functions
     * @brief Product of elements over given axes.
     *
     * Returns an \ref xreducer for the product of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the product is computed (optional)
     * @param es evaluation strategy of the reducer
     * @return an \ref xreducer
     */
    XTENSOR_REDUCER_FUNCTION(prod,
                             std::multiplies,
                             xtl::big_promote_type_t<typename std::decay_t<E>::value_type>,
                             1)
#ifdef X_OLD_CLANG
    XTENSOR_OLD_CLANG_REDUCER(prod,
                              std::multiplies,
                              xtl::big_promote_type_t<typename std::decay_t<E>::value_type>,
                              1)
#else
    XTENSOR_MODERN_CLANG_REDUCER(prod,
                                 std::multiplies,
                                 xtl::big_promote_type_t<typename std::decay_t<E>::value_type>,
                                 1)
#endif

    /**
     * @ingroup red_functions
     * @brief Mean of elements over given axes.
     *
     * Returns an \ref xreducer for the mean of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the mean is computed (optional)
     * @return an \ref xexpression
     */
    template <class T = void,
              class E,
              class X,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(xtl::negation<is_reducer_options<X>>)>
    inline auto mean(E&& e, X&& axes, EVS es = EVS())
    {
        using value_type = typename std::conditional_t<std::is_same<T, void>::value, double, T>;
        auto size = e.size();
        // sum cannot always be a double. It could be a complex number which cannot operate on
        // std::plus<double>.
        auto s = sum<T>(std::forward<E>(e), std::forward<X>(axes), es);
        return std::move(s) / static_cast<value_type>(size / s.size());
    }

    template <class T = void,
              class E,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(is_reducer_options<EVS>)>
    inline auto mean(E&& e, EVS es = EVS())
    {
        using value_type = typename std::conditional_t<std::is_same<T, void>::value, double, T>;
        auto size = e.size();
        return sum<T>(std::forward<E>(e), es) / static_cast<value_type>(size);
    }

#ifdef X_OLD_CLANG
    template <class T = void, class E, class I, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto mean(E&& e, std::initializer_list<I> axes, EVS es = EVS())
    {
        using value_type = typename std::conditional_t<std::is_same<T, void>::value, double, T>;
        auto size = e.size();
        auto s = sum<T>(std::forward<E>(e), axes, es);
        return std::move(s) / static_cast<value_type>(size / s.size());
    }
#else
    template <class T = void,
              class E,
              class I,
              std::size_t N,
              class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto mean(E&& e, const I (&axes)[N], EVS es = EVS())
    {
        using value_type = typename std::conditional_t<std::is_same<T, void>::value, double, T>;
        auto size = e.size();
        auto s = sum<T>(std::forward<E>(e), axes, es);
        return std::move(s) / static_cast<value_type>(size / s.size());
    }
#endif

    /**
     * @ingroup red_functions
     * @brief Average of elements over given axes using weights.
     *
     * Returns an \ref xreducer for the mean of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the mean is computed (optional)
     * @return an \ref xexpression
     *
     * @sa mean
     */
    template <class E,
              class W,
              class X,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(is_reducer_options<EVS>, xtl::negation<std::is_integral<X>>)>
    inline auto average(E&& e, W&& weights, X&& axes, EVS ev = EVS())
    {
        xindex_type_t<typename std::decay_t<E>::shape_type> broadcast_shape;
        xt::resize_container(broadcast_shape, e.dimension());
        auto ax = normalize_axis(e, axes);
        if (weights.dimension() == 1)
        {
            if (weights.size() != e.shape()[ax[0]])
            {
                throw std::runtime_error(
                    "Weights need to have the same shape as expression at axes.");
            }

            std::fill(broadcast_shape.begin(), broadcast_shape.end(), std::size_t(1));
            broadcast_shape[ax[0]] = weights.size();
        }
        else
        {
            if (!same_shape(e.shape(), weights.shape()))
            {
                throw std::runtime_error(
                    "Weights with dim > 1 need to have the same shape as expression.");
            }

            std::copy(e.shape().begin(), e.shape().end(), broadcast_shape.begin());
        }

        constexpr layout_type L = default_assignable_layout(std::decay_t<W>::static_layout);
        auto weights_view
            = reshape_view<L>(std::forward<W>(weights), std::move(broadcast_shape), L);
        auto scl = sum(weights_view, ax, xt::evaluation_strategy::immediate);
        return sum(std::forward<E>(e) * std::move(weights_view), std::move(ax), ev)
               / std::move(scl);
    }

#ifndef X_OLD_CLANG
    template <class E, class W, class X, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto average(E&& e, W&& weights, const X (&axes)[N], EVS ev = EVS())
    {
        // need to select the X&& overload and forward to different type
        using ax_t = std::array<std::size_t, N>;
        return average(
            std::forward<E>(e), std::forward<W>(weights), xt::forward_normalize<ax_t>(e, axes), ev);
    }
#else
    template <class E, class W, class I, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto average(E&& e, W&& weights, std::initializer_list<I> axes, EVS ev = EVS())
    {
        using ax_t = dynamic_shape<std::size_t>;
        return average(
            std::forward<E>(e), std::forward<W>(weights), xt::forward_normalize<ax_t>(e, axes), ev);
    }
#endif

    template <class E,
              class W,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(is_reducer_options<EVS>)>
    inline auto average(E&& e, W&& weights, EVS ev = EVS())
    {
        if (weights.dimension() != e.dimension()
            || !std::equal(weights.shape().begin(), weights.shape().end(), e.shape().begin()))
        {
            throw std::runtime_error("Weights need to have the same shape as expression.");
        }

        auto div = sum(weights, evaluation_strategy::immediate)();
        auto s = sum(std::forward<E>(e) * std::forward<W>(weights), ev) / std::move(div);
        return s;
    }

    namespace detail
    {
        template <typename E>
        std::enable_if_t<std::is_lvalue_reference<E>::value, E> shared_forward(E e) noexcept
        {
            return e;
        }

        template <typename E>
        std::enable_if_t<!std::is_lvalue_reference<E>::value, xshared_expression<E>> shared_forward(
            E e) noexcept
        {
            return make_xshared(std::move(e));
        }
    }

    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(is_reducer_options<EVS>)>
    inline auto variance(E&& e, EVS es = EVS())
    {
        decltype(auto) sc = detail::shared_forward<E>(e);
        return mean(square(abs(sc - mean(sc, es))), es);
    }

    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(is_reducer_options<EVS>)>
    inline auto stddev(E&& e, EVS es = EVS())
    {
        return sqrt(variance(std::forward<E>(e), es));
    }

    /**
     * @ingroup red_functions
     * @brief Compute the variance along the specified axes
     *
     * Returns the variance of the array elements, a measure of the spread of a
     * distribution. The variance is computed for the flattened array by default,
     * otherwise over the specified axes.
     *
     * Note: this function is not yet specialized for complex numbers.
     *
     * @param e an \ref xexpression
     * @param axes the axes along which the variance is computed (optional)
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xexpression
     *
     * @sa stddev, mean
     */
    template <class E,
              class X,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(xtl::negation<is_reducer_options<X>>)>
    inline auto variance(E&& e, X&& axes, EVS es = EVS())
    {
        decltype(auto) sc = detail::shared_forward<E>(e);
        // note: forcing copy of first axes argument -- is there a better solution?
        auto axes_copy = axes;
        auto inner_mean = mean(sc, std::move(axes_copy));

        // fake keep_dims = 1
        auto keep_dim_shape = e.shape();
        for (const auto& el : axes)
        {
            keep_dim_shape[el] = 1;
        }

        auto mrv = reshape_view<XTENSOR_DEFAULT_LAYOUT>(
            std::move(inner_mean), std::move(keep_dim_shape), XTENSOR_DEFAULT_LAYOUT);
        return mean(square(abs(sc - std::move(mrv))), std::forward<X>(axes), es);
    }

    /**
     * @ingroup red_functions
     * @brief Compute the standard deviation along the specified axis.
     *
     * Returns the standard deviation, a measure of the spread of a distribution,
     * of the array elements. The standard deviation is computed for the flattened
     * array by default, otherwise over the specified axis.
     *
     * Note: this function is not yet specialized for complex numbers.
     *
     * @param e an \ref xexpression
     * @param axes the axes along which the standard deviation is computed (optional)
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xexpression
     *
     * @sa variance, mean
     */
    template <class E,
              class X,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(xtl::negation<is_reducer_options<X>>)>
    inline auto stddev(E&& e, X&& axes, EVS es = EVS())
    {
        return sqrt(variance(std::forward<E>(e), std::forward<X>(axes), es));
    }

#ifndef X_OLD_CLANG
    template <class E, class A, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto stddev(E&& e, const A (&axes)[N], EVS es = EVS())
    {
        return stddev(std::forward<E>(e),
                      xtl::forward_sequence<std::array<std::size_t, N>, decltype(axes)>(axes),
                      es);
    }

    template <class E, class A, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto variance(E&& e, const A (&axes)[N], EVS es = EVS())
    {
        return variance(std::forward<E>(e),
                        xtl::forward_sequence<std::array<std::size_t, N>, decltype(axes)>(axes),
                        es);
    }
#else
    template <class E, class A, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto stddev(E&& e, std::initializer_list<A> axes, EVS es = EVS())
    {
        return stddev(std::forward<E>(e),
                      xtl::forward_sequence<dynamic_shape<std::size_t>, decltype(axes)>(axes),
                      es);
    }

    template <class E, class A, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto variance(E&& e, std::initializer_list<A> axes, EVS es = EVS())
    {
        return variance(std::forward<E>(e),
                        xtl::forward_sequence<dynamic_shape<std::size_t>, decltype(axes)>(axes),
                        es);
    }
#endif

    /**
     * @ingroup red_functions
     * @brief Minimum and maximum among the elements of an array or expression.
     *
     * Returns an \ref xreducer for the minimum and maximum of an expression's elements.
     * @param e an \ref xexpression
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xexpression of type ``std::array<value_type, 2>``, whose first
     *         and second element represent the minimum and maximum respectively
     */
    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(is_reducer_options<EVS>)>
    inline auto minmax(E&& e, EVS es = EVS())
    {
        using std::max;
        using std::min;
        using value_type = typename std::decay_t<E>::value_type;
        using result_type = std::array<value_type, 2>;

        auto reduce_func = [](result_type r, value_type const& v) {
            r[0] = (min)(r[0], v);
            r[1] = (max)(r[1], v);
            return r;
        };
        auto init_func = []() {
            return result_type{ std::numeric_limits<value_type>::max(),
                                std::numeric_limits<value_type>::lowest() };
        };
        auto merge_func = [](result_type r, result_type const& s) {
            r[0] = (min)(r[0], s[0]);
            r[1] = (max)(r[1], s[1]);
            return r;
        };
        return xt::reduce(make_xreducer_functor(
                              std::move(reduce_func), std::move(init_func), std::move(merge_func)),
                          std::forward<E>(e),
                          arange(e.dimension()),
                          es);
    }

    /**
     * @defgroup acc_functions accumulating functions
     */

    /**
     * @ingroup acc_functions
     * @brief Cumulative sum.
     *
     * Returns the accumulated sum for the elements over given
     * \em axis (or flattened).
     * @param e an \ref xexpression
     * @param axis the axes along which the cumulative sum is computed (optional)
     * @return an \ref xarray<T>
     */
    template <class E>
    inline auto cumsum(E&& e, std::ptrdiff_t axis)
    {
        using result_type = xtl::big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(
            make_xaccumulator_functor(std::plus<result_type>()), std::forward<E>(e), axis);
    }

    template <class E>
    inline auto cumsum(E&& e)
    {
        using result_type = xtl::big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(std::plus<result_type>()), std::forward<E>(e));
    }

    /**
     * @ingroup acc_functions
     * @brief Cumulative product.
     *
     * Returns the accumulated product for the elements over given
     * \em axis (or flattened).
     * @param e an \ref xexpression
     * @param axis the axes along which the cumulative product is computed (optional)
     * @return an \ref xarray<T>
     */
    template <class E>
    inline auto cumprod(E&& e, std::ptrdiff_t axis)
    {
        using result_type = xtl::big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(
            make_xaccumulator_functor(std::multiplies<result_type>()), std::forward<E>(e), axis);
    }

    template <class E>
    inline auto cumprod(E&& e)
    {
        using result_type = xtl::big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(std::multiplies<result_type>()),
                          std::forward<E>(e));
    }

    /*****************
     * nan functions *
     *****************/

    namespace detail
    {
        struct nan_to_num_functor
        {
            template <class A>
            inline auto operator()(const A& a) const
            {
                if (math::isnan(a))
                {
                    return A(0);
                }
                if (math::isinf(a))
                {
                    if (a < 0)
                    {
                        return std::numeric_limits<A>::lowest();
                    }
                    else
                    {
                        return (std::numeric_limits<A>::max)();
                    }
                }
                return a;
            }
        };

        template <class T>
        struct nan_plus
        {
            using value_type = T;
            using result_type = value_type;

            constexpr result_type operator()(const value_type lhs, const value_type rhs) const
            {
                return !math::isnan(rhs) ? lhs + rhs : lhs;
            }
        };

        template <class T>
        struct nan_multiplies
        {
            using value_type = T;
            using result_type = value_type;

            constexpr result_type operator()(const value_type lhs, const value_type rhs) const
            {
                return !math::isnan(rhs) ? lhs * rhs : lhs;
            }
        };

        template <class T, int V>
        struct nan_init
        {
            using value_type = T;
            using result_type = T;
            constexpr result_type operator()(const value_type lhs) const
            {
                return math::isnan(lhs) ? result_type(V) : lhs;
            }
        };
    }

    /**
     * @defgroup  nan_functions nan functions
     */

    /**
     * @ingroup nan_functions
     * @brief Convert nan or +/- inf to numbers
     *
     * This functions converts nan to 0, and +inf to the highest, -inf to the lowest
     * floating point value of the same type.
     *
     * @param e input \ref xexpression
     * @return an \ref xexpression
     */
    template <class E>
    inline auto nan_to_num(E&& e)
    {
        return detail::make_xfunction<detail::nan_to_num_functor>(std::forward<E>(e));
    }

#define XTENSOR_NAN_REDUCER_FUNCTION(NAME, FUNCTOR, RESULT_TYPE, NAN)                              \
    template <class T = void,                                                                      \
              class E,                                                                             \
              class X,                                                                             \
              class EVS = DEFAULT_STRATEGY_REDUCERS,                                               \
              XTL_REQUIRES(xtl::negation<is_reducer_options<X>>)>                                  \
    inline auto NAME(E&& e, X&& axes, EVS es = EVS())                                              \
    {                                                                                              \
        using result_type = std::conditional_t<std::is_same<T, void>::value, RESULT_TYPE, T>;      \
        using functor_type = FUNCTOR<result_type>;                                                 \
        using init_functor_type = detail::nan_init<result_type, NAN>;                              \
        return xt::reduce(make_xreducer_functor(functor_type(), init_functor_type()),              \
                          std::forward<E>(e),                                                      \
                          std::forward<X>(axes),                                                   \
                          es);                                                                     \
    }                                                                                              \
                                                                                                   \
    template <class T = void,                                                                      \
              class E,                                                                             \
              class EVS = DEFAULT_STRATEGY_REDUCERS,                                               \
              XTL_REQUIRES(is_reducer_options<EVS>)>                                               \
    inline auto NAME(E&& e, EVS es = EVS())                                                        \
    {                                                                                              \
        using result_type = std::conditional_t<std::is_same<T, void>::value, RESULT_TYPE, T>;      \
        using functor_type = FUNCTOR<result_type>;                                                 \
        using init_functor_type = detail::nan_init<result_type, NAN>;                              \
        return xt::reduce(                                                                         \
            make_xreducer_functor(functor_type(), init_functor_type()), std::forward<E>(e), es);   \
    }

#define OLD_CLANG_NAN_REDUCER(NAME, FUNCTOR, RESULT_TYPE, NAN)                                     \
    template <class T = void, class E, class I, class EVS = DEFAULT_STRATEGY_REDUCERS>             \
    inline auto NAME(E&& e, std::initializer_list<I> axes, EVS es = EVS())                         \
    {                                                                                              \
        using result_type = std::conditional_t<std::is_same<T, void>::value, RESULT_TYPE, T>;      \
        using functor_type = FUNCTOR<result_type>;                                                 \
        using init_functor_type = detail::nan_init<result_type, NAN>;                              \
        return xt::reduce(make_xreducer_functor(functor_type(), init_functor_type()),              \
                          std::forward<E>(e),                                                      \
                          axes,                                                                    \
                          es);                                                                     \
    }

#define MODERN_CLANG_NAN_REDUCER(NAME, FUNCTOR, RESULT_TYPE, NAN)                                  \
    template <class T = void,                                                                      \
              class E,                                                                             \
              class I,                                                                             \
              std::size_t N,                                                                       \
              class EVS = DEFAULT_STRATEGY_REDUCERS>                                               \
    inline auto NAME(E&& e, const I(&axes)[N], EVS es = EVS())                                     \
    {                                                                                              \
        using result_type = std::conditional_t<std::is_same<T, void>::value, RESULT_TYPE, T>;      \
        using functor_type = FUNCTOR<result_type>;                                                 \
        using init_functor_type = detail::nan_init<result_type, NAN>;                              \
        return xt::reduce(make_xreducer_functor(functor_type(), init_functor_type()),              \
                          std::forward<E>(e),                                                      \
                          axes,                                                                    \
                          es);                                                                     \
    }

    /**
     * @ingroup nan_functions
     * @brief Sum of elements over given axes, replacing nan with 0.
     *
     * Returns an \ref xreducer for the sum of elements over given
     * \em axes, replacing nan with 0.
     * @param e an \ref xexpression
     * @param axes the axes along which the sum is performed (optional)
     * @param es evaluation strategy of the reducer (optional)
     * @return an \ref xreducer
     */
    XTENSOR_REDUCER_FUNCTION(nansum, detail::nan_plus, typename std::decay_t<E>::value_type, 0)
#ifdef X_OLD_CLANG
    XTENSOR_OLD_CLANG_REDUCER(nansum, detail::nan_plus, typename std::decay_t<E>::value_type, 0)
#else
    XTENSOR_MODERN_CLANG_REDUCER(nansum, detail::nan_plus, typename std::decay_t<E>::value_type, 0)
#endif

    /**
     * @ingroup nan_functions
     * @brief Product of elements over given axes, replacing nan with 1.
     *
     * Returns an \ref xreducer for the sum of elements over given
     * \em axes, replacing nan with 1.
     * @param e an \ref xexpression
     * @param axes the axes along which the sum is performed (optional)
     * @param es evaluation strategy of the reducer (optional)
     * @return an \ref xreducer
     */
    XTENSOR_REDUCER_FUNCTION(nanprod,
                             detail::nan_multiplies,
                             typename std::decay_t<E>::value_type,
                             1)
#ifdef X_OLD_CLANG
    XTENSOR_OLD_CLANG_REDUCER(nanprod,
                              detail::nan_multiplies,
                              typename std::decay_t<E>::value_type,
                              1)
#else
    XTENSOR_MODERN_CLANG_REDUCER(nanprod,
                                 detail::nan_multiplies,
                                 typename std::decay_t<E>::value_type,
                                 1)
#endif

#undef XTENSOR_NAN_REDUCER_FUNCTION
#undef OLD_CLANG_NAN_REDUCER
#undef MODERN_CLANG_NAN_REDUCER

#define COUNT_NON_ZEROS_CONTENT                                                                    \
    using result_type = std::size_t;                                                               \
    using value_type = typename std::decay_t<E>::value_type;                                       \
    auto init_fct = []() -> result_type { return 0; };                                             \
    auto reduce_fct = [](const result_type& lhs, const value_type& rhs) -> result_type {           \
        return (rhs != value_type(0)) ? lhs + result_type(1) : lhs;                                \
    };                                                                                             \
    auto merge_func = std::plus<result_type>();

    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(is_reducer_options<EVS>)>
    inline auto count_nonzero(E&& e, EVS es = EVS())
    {
        COUNT_NON_ZEROS_CONTENT;
        return xt::reduce(make_xreducer_functor(
                              std::move(reduce_fct), std::move(init_fct), std::move(merge_func)),
                          std::forward<E>(e),
                          es);
    }

    template <class E,
              class X,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(xtl::negation<is_reducer_options<X>>,
                           xtl::negation<std::is_integral<X>>)>
    inline auto count_nonzero(E&& e, X&& axes, EVS es = EVS())
    {
        COUNT_NON_ZEROS_CONTENT;
        return xt::reduce(make_xreducer_functor(
                              std::move(reduce_fct), std::move(init_fct), std::move(merge_func)),
                          std::forward<E>(e),
                          std::forward<X>(axes),
                          es);
    }

    template <class E,
              class X,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(xtl::negation<is_reducer_options<X>>, std::is_integral<X>)>
    inline auto count_nonzero(E&& e, X axis, EVS es = EVS())
    {
        return count_nonzero(std::forward<E>(e), { axis }, es);
    }

#ifdef X_OLD_CLANG
    template <class E, class I, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto count_nonzero(E&& e, std::initializer_list<I> axes, EVS es = EVS())
    {
        COUNT_NON_ZEROS_CONTENT;
        return xt::reduce(make_xreducer_functor(
                              std::move(reduce_fct), std::move(init_fct), std::move(merge_func)),
                          std::forward<E>(e),
                          axes,
                          es);
    }
#else
    template <class E, class I, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto count_nonzero(E&& e, const I (&axes)[N], EVS es = EVS())
    {
        COUNT_NON_ZEROS_CONTENT;
        return xt::reduce(make_xreducer_functor(
                              std::move(reduce_fct), std::move(init_fct), std::move(merge_func)),
                          std::forward<E>(e),
                          axes,
                          es);
    }
#endif

#undef COUNT_NON_ZEROS_CONTENT

    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(is_reducer_options<EVS>)>
    inline auto count_nonnan(E&& e, EVS es = EVS())
    {
        return xt::count_nonzero(!xt::isnan(std::forward<E>(e)), es);
    }

    template <class E,
              class X,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(xtl::negation<is_reducer_options<X>>,
                           xtl::negation<std::is_integral<X>>)>
    inline auto count_nonnan(E&& e, X&& axes, EVS es = EVS())
    {
        return xt::count_nonzero(!xt::isnan(std::forward<E>(e)), std::forward<X>(axes), es);
    }

    template <class E,
              class X,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(xtl::negation<is_reducer_options<X>>, std::is_integral<X>)>
    inline auto count_nonnan(E&& e, X&& axes, EVS es = EVS())
    {
        return xt::count_nonzero(!xt::isnan(std::forward<E>(e)), { axes }, es);
    }

#ifdef X_OLD_CLANG
    template <class E, class I, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto count_nonnan(E&& e, std::initializer_list<I> axes, EVS es = EVS())
    {
        return xt::count_nonzero(!xt::isnan(std::forward<E>(e)), axes, es);
    }
#else
    template <class E, class I, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto count_nonnan(E&& e, const I (&axes)[N], EVS es = EVS())
    {
        return xt::count_nonzero(!xt::isnan(std::forward<E>(e)), axes, es);
    }
#endif

    /**
     * @ingroup nan_functions
     * @brief Cumulative sum, replacing nan with 0.
     *
     * Returns an xaccumulator for the sum of elements over given
     * \em axis, replacing nan with 0.
     * @param e an \ref xexpression
     * @param axis the axis along which the elements are accumulated (optional)
     * @return an xaccumulator
     */
    template <class E>
    inline auto nancumsum(E&& e, std::ptrdiff_t axis)
    {
        using result_type = xtl::big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(detail::nan_plus<result_type>(),
                                                    detail::nan_init<result_type, 0>()),
                          std::forward<E>(e),
                          axis);
    }

    template <class E>
    inline auto nancumsum(E&& e)
    {
        using result_type = xtl::big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(detail::nan_plus<result_type>(),
                                                    detail::nan_init<result_type, 0>()),
                          std::forward<E>(e));
    }

    /**
     * @ingroup nan_functions
     * @brief Cumulative product, replacing nan with 1.
     *
     * Returns an xaccumulator for the product of elements over given
     * \em axis, replacing nan with 1.
     * @param e an \ref xexpression
     * @param axis the axis along which the elements are accumulated (optional)
     * @return an xaccumulator
     */
    template <class E>
    inline auto nancumprod(E&& e, std::ptrdiff_t axis)
    {
        using result_type = xtl::big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(detail::nan_multiplies<result_type>(),
                                                    detail::nan_init<result_type, 1>()),
                          std::forward<E>(e),
                          axis);
    }

    template <class E>
    inline auto nancumprod(E&& e)
    {
        using result_type = xtl::big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(detail::nan_multiplies<result_type>(),
                                                    detail::nan_init<result_type, 1>()),
                          std::forward<E>(e));
    }

    namespace detail
    {
        template <class T>
        struct diff_impl
        {
            template <class Arg>
            inline void operator()(Arg& ad,
                                   const std::size_t& n,
                                   xstrided_slice_vector& slice1,
                                   xstrided_slice_vector& slice2,
                                   std::size_t saxis)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    slice2[saxis] = range(xnone(), ad.shape()[saxis] - 1);
                    ad = strided_view(ad, slice1) - strided_view(ad, slice2);
                }
            }
        };

        template <>
        struct diff_impl<bool>
        {
            template <class Arg>
            inline void operator()(Arg& ad,
                                   const std::size_t& n,
                                   xstrided_slice_vector& slice1,
                                   xstrided_slice_vector& slice2,
                                   std::size_t saxis)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    slice2[saxis] = range(xnone(), ad.shape()[saxis] - 1);
                    ad = not_equal(strided_view(ad, slice1), strided_view(ad, slice2));
                }
            }
        };
    }

    /**
     * @ingroup red_functions
     * @brief Mean of elements over given axes, excluding nans.
     *
     * Returns an \ref xreducer for the mean of elements over given
     * \em axes, excluding nans.
     * @param e an \ref xexpression
     * @param axes the axes along which the mean is computed (optional)
     * @return an \ref xexpression
     */
    template <class T = void,
              class E,
              class X,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(xtl::negation<is_reducer_options<X>>)>
    inline auto nanmean(E&& e, X&& axes, EVS es = EVS())
    {
        decltype(auto) sc = detail::shared_forward<E>(e);
        // note: forcing copy of first axes argument -- is there a better solution?
        auto axes_copy = axes;
        // sum cannot always be a double. It could be a complex number which cannot operate on
        // std::plus<double>.
        return nansum<T>(sc, std::forward<X>(axes), es)
               / xt::cast<double>(count_nonnan(sc, std::move(axes_copy), es));
    }

    template <class T = void,
              class E,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(is_reducer_options<EVS>)>
    inline auto nanmean(E&& e, EVS es = EVS())
    {
        decltype(auto) sc = detail::shared_forward<E>(e);
        return nansum<T>(sc, es) / xt::cast<double>(count_nonnan(sc, es));
    }

#ifdef X_OLD_CLANG
    template <class T = void, class E, class I, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto nanmean(E&& e, std::initializer_list<I> axes, EVS es = EVS())
    {
        return nanmean(std::forward<E>(e),
                       xtl::forward_sequence<dynamic_shape<std::size_t>, decltype(axes)>(axes),
                       es);
    }
#else
    template <class T = void,
              class E,
              class I,
              std::size_t N,
              class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto nanmean(E&& e, const I (&axes)[N], EVS es = EVS())
    {
        return nanmean(std::forward<E>(e),
                       xtl::forward_sequence<std::array<std::size_t, N>, decltype(axes)>(axes),
                       es);
    }
#endif

    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(is_reducer_options<EVS>)>
    inline auto nanvar(E&& e, EVS es = EVS())
    {
        decltype(auto) sc = detail::shared_forward<E>(e);
        return nanmean(square(abs(sc - nanmean(sc))), es);
    }

    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS, XTL_REQUIRES(is_reducer_options<EVS>)>
    inline auto nanstd(E&& e, EVS es = EVS())
    {
        return sqrt(nanvar(std::forward<E>(e), es));
    }

    /**
     * @ingroup red_functions
     * @brief Compute the variance along the specified axes, excluding nans
     *
     * Returns the variance of the array elements, a measure of the spread of a
     * distribution. The variance is computed for the flattened array by default,
     * otherwise over the specified axes.
     *
     * Note: this function is not yet specialized for complex numbers.
     *
     * @param e an \ref xexpression
     * @param axes the axes along which the variance is computed (optional)
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xexpression
     *
     * @sa nanstd, nanmean
     */
    template <class E,
              class X,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(xtl::negation<is_reducer_options<X>>)>
    inline auto nanvar(E&& e, X&& axes, EVS es = EVS())
    {
        decltype(auto) sc = detail::shared_forward<E>(e);
        // note: forcing copy of first axes argument -- is there a better solution?
        auto axes_copy = axes;
        auto inner_mean = nanmean(sc, std::move(axes_copy));

        // fake keep_dims = 1
        auto keep_dim_shape = e.shape();
        for (const auto& el : axes)
        {
            keep_dim_shape[el] = 1;
        }
        auto mrv = reshape_view<XTENSOR_DEFAULT_LAYOUT>(
            std::move(inner_mean), std::move(keep_dim_shape), XTENSOR_DEFAULT_LAYOUT);
        return nanmean(square(abs(sc - std::move(mrv))), std::forward<X>(axes), es);
    }

    /**
     * @ingroup red_functions
     * @brief Compute the standard deviation along the specified axis, excluding nans.
     *
     * Returns the standard deviation, a measure of the spread of a distribution,
     * of the array elements. The standard deviation is computed for the flattened
     * array by default, otherwise over the specified axis.
     *
     * Note: this function is not yet specialized for complex numbers.
     *
     * @param e an \ref xexpression
     * @param axes the axes along which the standard deviation is computed (optional)
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xexpression
     *
     * @sa nanvar, nanmean
     */
    template <class E,
              class X,
              class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTL_REQUIRES(xtl::negation<is_reducer_options<X>>)>
    inline auto nanstd(E&& e, X&& axes, EVS es = EVS())
    {
        return sqrt(nanvar(std::forward<E>(e), std::forward<X>(axes), es));
    }

#ifndef X_OLD_CLANG
    template <class E, class A, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto nanstd(E&& e, const A (&axes)[N], EVS es = EVS())
    {
        return nanstd(std::forward<E>(e),
                      xtl::forward_sequence<std::array<std::size_t, N>, decltype(axes)>(axes),
                      es);
    }

    template <class E, class A, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto nanvar(E&& e, const A (&axes)[N], EVS es = EVS())
    {
        return nanvar(std::forward<E>(e),
                      xtl::forward_sequence<std::array<std::size_t, N>, decltype(axes)>(axes),
                      es);
    }
#else
    template <class E, class A, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto nanstd(E&& e, std::initializer_list<A> axes, EVS es = EVS())
    {
        return nanstd(std::forward<E>(e),
                      xtl::forward_sequence<dynamic_shape<std::size_t>, decltype(axes)>(axes),
                      es);
    }

    template <class E, class A, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto nanvar(E&& e, std::initializer_list<A> axes, EVS es = EVS())
    {
        return nanvar(std::forward<E>(e),
                      xtl::forward_sequence<dynamic_shape<std::size_t>, decltype(axes)>(axes),
                      es);
    }
#endif

    /**
     * @ingroup red_functions
     * @brief Calculate the n-th discrete difference along the given axis.
     *
     * Calculate the n-th discrete difference along the given axis. This function is not lazy (might
     * change in the future).
     * @param a an \ref xexpression
     * @param n The number of times values are differenced. If zero, the input is returned as-is.
     * (optional)
     * @param axis The axis along which the difference is taken, default is the last axis.
     * @return an xarray
     */
    template <class T>
    auto diff(const xexpression<T>& a, std::size_t n = 1, std::ptrdiff_t axis = -1)
    {
        auto ad = a.derived_cast();
        std::size_t saxis = normalize_axis(ad.dimension(), axis);

        if (n == 0)
        {
            return eval(ad);
        }

        xstrided_slice_vector slice1(ad.dimension(), all());
        xstrided_slice_vector slice2(ad.dimension(), all());
        slice1[saxis] = range(1, xnone());

        detail::diff_impl<typename T::value_type> impl;
        impl(ad, n, slice1, slice2, saxis);

        return eval(ad);
    }

    /**
     * @ingroup red_functions
     * @brief Integrate along the given axis using the composite trapezoidal rule.
     *
     * Returns definite integral as approximated by trapezoidal rule. This function is not lazy
     * (might change in the future).
     * @param y an \ref xexpression
     * @param dx the spacing between sample points (optional)
     * @param axis the axis along which to integrate.
     * @return an xarray
     */
    template <class T>
    auto trapz(const xexpression<T>& y, double dx = 1.0, std::ptrdiff_t axis = -1)
    {
        auto& yd = y.derived_cast();
        std::size_t saxis = normalize_axis(yd.dimension(), axis);

        xstrided_slice_vector slice1(yd.dimension(), all());
        xstrided_slice_vector slice2(yd.dimension(), all());
        slice1[saxis] = range(1, xnone());
        slice2[saxis] = range(xnone(), yd.shape()[saxis] - 1);

        auto trap = dx * (strided_view(yd, slice1) + strided_view(yd, slice2)) * 0.5;

        return eval(sum(trap, { saxis }));
    }

    /**
     * @ingroup red_functions
     * @brief Integrate along the given axis using the composite trapezoidal rule.
     *
     * Returns definite integral as approximated by trapezoidal rule. This function is not lazy
     * (might change in the future).
     * @param y an \ref xexpression
     * @param x an \ref xexpression representing the sample points corresponding to the y values.
     * @param axis the axis along which to integrate.
     * @return an xarray
     */
    template <class T, class E>
    auto trapz(const xexpression<T>& y, const xexpression<E>& x, std::ptrdiff_t axis = -1)
    {
        auto& yd = y.derived_cast();
        auto& xd = x.derived_cast();
        decltype(diff(x)) dx;

        std::size_t saxis = normalize_axis(yd.dimension(), axis);

        if (xd.dimension() == 1)
        {
            dx = diff(x);
            typename std::decay_t<decltype(yd)>::shape_type shape;
            resize_container(shape, yd.dimension());
            std::fill(shape.begin(), shape.end(), 1);
            shape[saxis] = dx.shape()[0];
            dx.reshape(shape);
        }
        else
        {
            dx = diff(x, 1, axis);
        }

        xstrided_slice_vector slice1(yd.dimension(), all());
        xstrided_slice_vector slice2(yd.dimension(), all());
        slice1[saxis] = range(1, xnone());
        slice2[saxis] = range(xnone(), yd.shape()[saxis] - 1);

        auto trap = dx * (strided_view(yd, slice1) + strided_view(yd, slice2)) * 0.5;

        return eval(sum(trap, { saxis }));
    }

    /**
     * @ingroup basic_functions
     * @brief Returns the one-dimensional piecewise linear interpolant to a function with given
     * discrete data points (xp, fp), evaluated at x.
     *
     * @param x The x-coordinates at which to evaluate the interpolated values (sorted).
     * @param xp The x-coordinates of the data points (sorted).
     * @param fp The y-coordinates of the data points, same length as xp.
     * @param left Value to return for x < xp[0].
     * @param right Value to return for x > xp[-1]
     * @return an one-dimensional xarray, same length as x.
     */
    template <class E1, class E2, class E3, typename T>
    inline auto interp(const E1& x, const E2& xp, const E3& fp, T left, T right)
    {
        using size_type = common_size_type_t<E1, E2, E3>;
        using value_type = typename E3::value_type;

        // basic checks
        XTENSOR_ASSERT(xp.dimension() == 1);
        XTENSOR_ASSERT(std::is_sorted(x.cbegin(), x.cend()));
        XTENSOR_ASSERT(std::is_sorted(xp.cbegin(), xp.cend()));

        // allocate output
        auto f = xtensor<value_type, 1>::from_shape(x.shape());

        // counter in "x": from left
        size_type i = 0;

        // fill f[i] for x[i] <= xp[0]
        for (; i < x.size(); ++i)
        {
            if (x[i] > xp[0])
            {
                break;
            }
            f[i] = static_cast<value_type>(left);
        }

        // counter in "x": from right
        // (index counts one right, to terminate the reverse loop, without risking being negative)
        size_type imax = x.size();

        // fill f[i] for x[-1] >= xp[-1]
        for (; imax > 0; --imax)
        {
            if (x[imax - 1] < xp[xp.size() - 1])
            {
                break;
            }
            f[imax - 1] = static_cast<value_type>(right);
        }

        // catch edge case: all entries are "right"
        if (imax == 0)
        {
            return f;
        }

        // set "imax" as actual index
        // (counted one right, see above)
        --imax;

        // counter in "xp"
        size_type ip = 1;

        // fill f[i] for the interior
        for (; i <= imax; ++i)
        {
            // - search next value in "xp"
            while (x[i] > xp[ip])
            {
                ++ip;
            }
            // - distances as doubles
            double dfp = static_cast<double>(fp[ip] - fp[ip - 1]);
            double dxp = static_cast<double>(xp[ip] - xp[ip - 1]);
            double dx = static_cast<double>(x[i] - xp[ip - 1]);
            // - interpolate
            f[i] = fp[ip - 1] + static_cast<value_type>(dfp / dxp * dx);
        }

        return f;
    }

    /**
     * @ingroup basic_functions
     * @brief Returns the one-dimensional piecewise linear interpolant to a function with given
     * discrete data points (xp, fp), evaluated at x.
     *
     * @param x The x-coordinates at which to evaluate the interpolated values (sorted).
     * @param xp The x-coordinates of the data points (sorted).
     * @param fp The y-coordinates of the data points, same length as xp.
     * @return an one-dimensional xarray, same length as x.
     */
    template <class E1, class E2, class E3>
    inline auto interp(const E1& x, const E2& xp, const E3& fp)
    {
        return interp(x, xp, fp, fp[0], fp[fp.size() - 1]);
    }
}

#endif
