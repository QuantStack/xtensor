/****************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_COMPLEX_HPP
#define XTENSOR_COMPLEX_HPP

#include <type_traits>
#include <utility>

#include <xtl/xcomplex.hpp>

#include "xtensor/xbuilder.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xoffset_view.hpp"

namespace xt
{
    /******************************
     * real and imag declarations *
     ******************************/

    template <class E>
    decltype(auto) real(E&& e) noexcept;

    template <class E>
    decltype(auto) imag(E&& e) noexcept;

    /********************************
     * real and imag implementation *
     ********************************/

    namespace detail
    {
        template <bool iscomplex = true>
        struct complex_helper
        {
            template <class E>
            static inline auto real(E&& e) noexcept
            {
                using real_type = typename std::decay_t<E>::value_type::value_type;
                return xoffset_view<xclosure_t<E>, real_type, 0>(std::forward<E>(e));
            }

            template <class E>
            static inline auto imag(E&& e) noexcept
            {
                using real_type = typename std::decay_t<E>::value_type::value_type;
                return xoffset_view<xclosure_t<E>, real_type, sizeof(real_type)>(
                    std::forward<E>(e));
            }
        };

        template <>
        struct complex_helper<false>
        {
            template <class E>
            static inline decltype(auto) real(E&& e) noexcept
            {
                return std::forward<E>(e);
            }

            template <class E>
            static inline auto imag(E&& e) noexcept
            {
                return zeros<typename std::decay_t<E>::value_type>(e.shape());
            }
        };

        template <bool isexpression = true>
        struct complex_expression_helper
        {
            template <class E>
            static inline decltype(auto) real(E&& e) noexcept
            {
                return detail::complex_helper<xtl::is_complex<
                    typename std::decay_t<E>::value_type>::value>::real(std::forward<E>(e));
            }

            template <class E>
            static inline decltype(auto) imag(E&& e) noexcept
            {
                return detail::complex_helper<xtl::is_complex<
                    typename std::decay_t<E>::value_type>::value>::imag(std::forward<E>(e));
            }
        };

        template <>
        struct complex_expression_helper<false>
        {
            template <class E>
            static inline decltype(auto) real(E&& e) noexcept
            {
                return xtl::forward_real(std::forward<E>(e));
            }

            template <class E>
            static inline decltype(auto) imag(E&& e) noexcept
            {
                return xtl::forward_imag(std::forward<E>(e));
            }
        };
    }

    /**
     * @brief Returns an \ref xexpression representing the real part of the given expression.
     *
     * @tparam e the \ref xexpression
     *
     * The returned expression either hold a const reference to \p e or a copy
     * depending on whether \p e is an lvalue or an rvalue.
     */
    template <class E>
    inline decltype(auto) real(E&& e) noexcept
    {
        return detail::complex_expression_helper<is_xexpression<std::decay_t<E>>::value>::real(
            std::forward<E>(e));
    }

    /**
     * @brief Returns an \ref xexpression representing the imaginary part of the given expression.
     *
     * @tparam e the \ref xexpression
     *
     * The returned expression either hold a const reference to \p e or a copy
     * depending on whether \p e is an lvalue or an rvalue.
     */
    template <class E>
    inline decltype(auto) imag(E&& e) noexcept
    {
        return detail::complex_expression_helper<is_xexpression<std::decay_t<E>>::value>::imag(
            std::forward<E>(e));
    }

#define UNARY_COMPLEX_FUNCTOR(NS, NAME)                                                            \
    struct NAME##_fun                                                                              \
    {                                                                                              \
        template <class T>                                                                         \
        constexpr auto operator()(const T& t) const                                                \
        {                                                                                          \
            using NS::NAME;                                                                        \
            return NAME(t);                                                                        \
        }                                                                                          \
                                                                                                   \
        template <class B>                                                                         \
        constexpr auto simd_apply(const B& t) const                                                \
        {                                                                                          \
            using NS::NAME;                                                                        \
            return NAME(t);                                                                        \
        }                                                                                          \
    }

    namespace math
    {
        namespace detail
        {
            template <class T>
            constexpr std::complex<T> conj_impl(const std::complex<T>& c)
            {
                return std::complex<T>(c.real(), -c.imag());
            }

#ifdef XTENSOR_USE_XSIMD
            template <class X>
            constexpr X conj_impl(const xsimd::simd_complex_batch<X>& z)
            {
                return xsimd::conj(z);
            }

            template <class T>
            struct not_complex_batch
                : xtl::negation<std::is_base_of<xsimd::simd_complex_batch<T>, T>>
            {
            };

            // libc++ (OSX) conj is unfortunately broken and returns
            // std::complex<T> instead of T.
            // This function must be deactivated for complex batches,
            // otherwise it will be a better match than the previous one.
            template <class T, XTL_REQUIRES(not_complex_batch<T>)>
            constexpr T conj_impl(const T& c)
            {
                return c;
            }
#else
            // libc++ (OSX) conj is unfortunately broken and returns
            // std::complex<T> instead of T.
            template <class T>
            constexpr T conj_impl(const T& c)
            {
                return c;
            }
#endif
        }

        UNARY_COMPLEX_FUNCTOR(std, norm);
        UNARY_COMPLEX_FUNCTOR(std, arg);
        UNARY_COMPLEX_FUNCTOR(detail, conj_impl);
    }

#undef UNARY_COMPLEX_FUNCTOR

    /**
     * @brief Returns an \ref xfunction evaluating to the complex conjugate of the given expression.
     *
     * @param e the \ref xexpression
     */
    template <class E>
    inline auto conj(E&& e) noexcept
    {
        using functor = math::conj_impl_fun;
        using type = xfunction<functor, const_xclosure_t<E>>;
        return type(functor(), std::forward<E>(e));
    }

    /**
     * @brief Calculates the phase angle (in radians) elementwise for the complex numbers in e.
     * @param e the \ref xexpression
     */
    template <class E>
    inline auto arg(E&& e) noexcept
    {
        using functor = math::arg_fun;
        using type = xfunction<functor, const_xclosure_t<E>>;
        return type(functor(), std::forward<E>(e));
    }

    /**
     * @brief Calculates the phase angle elementwise for the complex numbers in e.
     * Note that this function might be slightly less perfomant than \ref arg.
     * @param e the \ref xexpression
     * @param deg calculate angle in degrees instead of radians
     */
    template <class E>
    inline auto angle(E&& e, bool deg = false) noexcept
    {
        using value_type = xtl::complex_value_type_t<typename std::decay_t<E>::value_type>;
        value_type multiplier = 1.0;
        if (deg)
        {
            multiplier = value_type(180) / numeric_constants<value_type>::PI;
        }
        return arg(std::forward<E>(e)) * std::move(multiplier);
    }

    /**
     * Calculates the squared magnitude elementwise for the complex numbers in e.
     * Equivalent to pow(real(e), 2) + pow(imag(e), 2).
     * @param e the \ref xexpression
     */
    template <class E>
    inline auto norm(E&& e) noexcept
    {
        using functor = math::norm_fun;
        using type = xfunction<functor, const_xclosure_t<E>>;
        return type(functor(), std::forward<E>(e));
    }
}
#endif
