/****************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_EXPRESSION_TRAITS_HPP
#define XTENSOR_EXPRESSION_TRAITS_HPP

#include "xexpression.hpp"

namespace xt
{
    /***************
     * xvalue_type *
     ***************/

    namespace detail
    {
        template <class E, class enable = void>
        struct xvalue_type_impl
        {
            using type = E;
        };

        template <class E>
        struct xvalue_type_impl<E, std::enable_if_t<is_xexpression<E>::value>>
        {
            using type = typename E::value_type;
        };
    }

    template <class E>
    using xvalue_type = detail::xvalue_type_impl<E>;

    template <class E>
    using xvalue_type_t = typename xvalue_type<E>::type;

    /*********************
     * common_value_type *
     *********************/

    template <class... C>
    struct common_value_type
    {
        using type = std::common_type_t<typename std::decay_t<C>::value_type...>;
    };

    template <class... C>
    using common_value_type_t = typename common_value_type<C...>::type;

    /********************
     * common_size_type *
     ********************/

    template <class... Args>
    struct common_size_type
    {
        using type = std::common_type_t<typename Args::size_type...>;
    };

    template <>
    struct common_size_type<>
    {
        using type = std::size_t;
    };

    template <class... Args>
    using common_size_type_t = typename common_size_type<Args...>::type;

    /**************************
     * common_difference type *
     **************************/

    template <class... Args>
    struct common_difference_type
    {
        using type = std::common_type_t<typename Args::difference_type...>;
    };

    template <>
    struct common_difference_type<>
    {
        using type = std::ptrdiff_t;
    };

    template <class... Args>
    using common_difference_type_t = typename common_difference_type<Args...>::type;

    /******************
     * temporary_type *
     ******************/

    namespace detail
    {
        template <class S>
        struct xtype_for_shape
        {
            template <class T, layout_type L>
            using type = xarray<T, L>;
        };

#if defined(__GNUC__) && (__GNUC__ > 6)
#if __cplusplus == 201703L
        template <template <class, std::size_t, class, bool> class S,
                  class X,
                  std::size_t N,
                  class A,
                  bool Init>
        struct xtype_for_shape<S<X, N, A, Init>>
        {
            template <class T, layout_type L>
            using type = xarray<T, L>;
        };
#endif  // __cplusplus == 201703L
#endif  // __GNUC__ && (__GNUC__ > 6)

        template <template <class, std::size_t> class S, class X, std::size_t N>
        struct xtype_for_shape<S<X, N>>
        {
            template <class T, layout_type L>
            using type = xtensor<T, N, L>;
        };

        template <template <std::size_t...> class S, std::size_t... X>
        struct xtype_for_shape<S<X...>>
        {
            template <class T, layout_type L>
            using type = xtensor_fixed<T, xshape<X...>, L>;
        };
    }

    template <class T, class S, layout_type L>
    struct temporary_type
    {
        using type = typename detail::xtype_for_shape<S>::template type<T, L>;
    };

    template <class T, class S, layout_type L>
    using temporary_type_t = typename temporary_type<T, S, L>::type;

    /**********************
     * common_tensor_type *
     **********************/

    namespace detail
    {
        template <class... C>
        struct common_tensor_type_impl
        {
            static constexpr layout_type static_layout
                = compute_layout(std::decay_t<C>::static_layout...);
            using type = temporary_type_t<common_value_type_t<C...>,
                                          promote_shape_t<typename C::shape_type...>,
                                          static_layout>;
        };
    }

    template <class... C>
    struct common_tensor_type : detail::common_tensor_type_impl<std::decay_t<C>...>
    {
    };

    template <class... C>
    using common_tensor_type_t = typename common_tensor_type<C...>::type;
}

#endif
