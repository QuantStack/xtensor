/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XPADDED_HPP
#define XPADDED_HPP

#include "xfixed.hpp"

namespace xt
{
    template <class ET, class S, layout_type L, class Tag>
    class xpadded_container;

    namespace detail
    {
        template<class ET, layout_type L, std::size_t I, std::size_t... X>
        struct pad_axis;

        template<class ET, layout_type L, class A, std::size_t... X>
        struct pad_shape_impl;

        template<class ET, layout_type L, class S>
        struct pad_shape;

        template<class ET, std::size_t X>
        struct simd_stride
        {
            constexpr static std::ptrdiff_t value = (X + xsimd::simd_type<ET>::size - 1) & ~(xsimd::simd_type<ET>::size - 1);
        };

        template<class ET, std::size_t I, std::size_t... X>
        struct pad_axis<ET, layout_type::row_major, I, X...>
        {
            constexpr static std::ptrdiff_t value = (I == sizeof...(X) - 1) ? simd_stride<ET, at<I, X...>::value>::value : at<I, X...>::value;
        };

        template<class ET, std::size_t I, std::size_t... X>
        struct pad_axis<ET, layout_type::column_major, I, X...>
        {
            constexpr static std::ptrdiff_t value = I ? at<I, X...>::value : simd_stride<ET, at<I, X...>::value>::value;
        };

        template<class ET, layout_type L, std::size_t... IX, std::size_t... X>
        struct pad_shape_impl<ET, L, std::index_sequence<IX...>, X...>
        {
            using type = fixed_shape<pad_axis<ET, L, IX, X...>::value...>;
        };

        template<class ET, layout_type L, std::size_t... X>
        struct pad_shape<ET, L, fixed_shape<X...>> : pad_shape_impl<ET, L, std::make_index_sequence<sizeof...(X)>, X...> {};
    }

    template<class ET, layout_type L, class S>
    using pad_shape_t = typename detail::pad_shape<ET, L, S>::type;

    template <class ET, class S, layout_type L, class Tag>
    struct xcontainer_inner_types<xpadded_container<ET, S, L, Tag>>
    {
        using shape_type = S;
        using data_shape_type = pad_shape_t<ET, L, S>;
        using strides_type = get_strides_t<data_shape_type>;
        using backstrides_type = strides_type;
        using storage_type = aligned_array<ET, detail::fixed_compute_size<data_shape_type>::value>;
        using temporary_type = xpadded_container<ET, S, L, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class ET, class S, layout_type L, class Tag>
    struct xiterable_inner_types<xpadded_container<ET, S, L, Tag>>
        : xcontainer_iterable_types<xpadded_container<ET, S, L, Tag>>
    {
    };

    template <class ET, class S, layout_type L, class Tag>
    class xpadded_container : public xcontainer<xpadded_container<ET, S, L, Tag>>,
                              public xcontainer_semantic<xpadded_container<ET, S, L, Tag>>
    {
    };

}

#endif
