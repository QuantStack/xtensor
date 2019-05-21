/****************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_STRIDES_HPP
#define XTENSOR_STRIDES_HPP

#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>

#include <xtl/xsequence.hpp>

#include "xexception.hpp"
#include "xshape.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    template <class shape_type>
    std::size_t compute_size(const shape_type& shape) noexcept;

    /***************
     * data offset *
     ***************/

    template <class offset_type, class S>
    offset_type data_offset(const S& strides) noexcept;

    template <class offset_type, class S, class Arg, class... Args>
    offset_type data_offset(const S& strides, Arg arg, Args... args) noexcept;

    template <class offset_type, layout_type L = layout_type::dynamic, class S, class... Args>
    offset_type unchecked_data_offset(const S& strides, Args... args) noexcept;

    template <class offset_type, class S, class It>
    offset_type element_offset(const S& strides, It first, It last) noexcept;

    /*******************
     * strides builder *
     *******************/

    template <layout_type L = layout_type::dynamic, class shape_type, class strides_type>
    std::size_t compute_strides(const shape_type& shape, layout_type l, strides_type& strides);

    template <layout_type L = layout_type::dynamic,
              class shape_type,
              class strides_type,
              class backstrides_type>
    std::size_t compute_strides(const shape_type& shape,
                                layout_type l,
                                strides_type& strides,
                                backstrides_type& backstrides);

    template <class shape_type, class strides_type>
    void adapt_strides(const shape_type& shape, strides_type& strides) noexcept;

    template <class shape_type, class strides_type, class backstrides_type>
    void adapt_strides(const shape_type& shape,
                       strides_type& strides,
                       backstrides_type& backstrides) noexcept;

    /*****************
     * unravel_index *
     *****************/

    template <class S>
    S unravel_from_strides(typename S::value_type index,
                           const S& strides,
                           layout_type l = layout_type::row_major);

    template <class S>
    get_strides_t<S> unravel_index(typename S::value_type index,
                                   const S& shape,
                                   layout_type l = layout_type::row_major);

    template <class S, class T>
    std::vector<get_strides_t<S>> unravel_indices(const T& indices,
                                                  const S& shape,
                                                  layout_type l = layout_type::row_major);

    /***********************
     * broadcast functions *
     ***********************/

    template <class S, class size_type>
    S uninitialized_shape(size_type size);

    template <class S1, class S2>
    bool broadcast_shape(const S1& input, S2& output);

    template <class S1, class S2>
    bool broadcastable(const S1& s1, S2& s2);

    /*************************
     * check strides overlap *
     *************************/

    template <layout_type L>
    struct check_strides_overlap;

    /**********************************
     * check bounds, without throwing *
     **********************************/

    template <class S, class... Args>
    bool in_bounds(const S& shape, Args&... args);

    /********************************
     * apply periodicity to indices *
     *******************************/

    template <class S, class... Args>
    void normalize_periodic(const S& shape, Args&... args);

    /********************************************
     * utility functions for strided containers *
     ********************************************/

    template <class C, class It, class size_type>
    It strided_data_end(const C& c, It begin, layout_type l, size_type offset)
    {
        using difference_type = typename std::iterator_traits<It>::difference_type;
        if (c.dimension() == 0)
        {
            ++begin;
        }
        else
        {
            for (std::size_t i = 0; i != c.dimension(); ++i)
            {
                begin += c.strides()[i] * difference_type(c.shape()[i] - 1);
            }
            if (l == layout_type::row_major)
            {
                begin += c.strides().back();
            }
            else
            {
                if (offset == 0)
                {
                    begin += c.strides().front();
                }
            }
        }
        return begin;
    }

    /******************
     * Implementation *
     ******************/

    namespace detail
    {
        template <class shape_type>
        inline std::size_t compute_size_impl(const shape_type& shape,
                                             std::true_type /* is signed */)
        {
            using size_type = std::decay_t<typename shape_type::value_type>;
            return static_cast<std::size_t>(std::abs(std::accumulate(
                shape.cbegin(), shape.cend(), size_type(1), std::multiplies<size_type>())));
        }

        template <class shape_type>
        inline std::size_t compute_size_impl(const shape_type& shape,
                                             std::false_type /* is not signed */)
        {
            using size_type = std::decay_t<typename shape_type::value_type>;
            return static_cast<std::size_t>(std::accumulate(
                shape.cbegin(), shape.cend(), size_type(1), std::multiplies<size_type>()));
        }
    }

    template <class shape_type>
    inline std::size_t compute_size(const shape_type& shape) noexcept
    {
        return detail::compute_size_impl(
            shape, std::is_signed<std::decay_t<typename std::decay_t<shape_type>::value_type>>());
    }

    namespace detail
    {
        template <std::size_t dim, class S>
        inline auto raw_data_offset(const S&) noexcept
        {
            using strides_value_type = std::decay_t<decltype(std::declval<S>()[0])>;
            return strides_value_type(0);
        }

        template <std::size_t dim, class S, class Arg, class... Args>
        inline auto raw_data_offset(const S& strides, Arg arg, Args... args) noexcept
        {
            return arg * strides[dim] + raw_data_offset<dim + 1>(strides, args...);
        }

        template <layout_type L, std::ptrdiff_t static_dim>
        struct layout_data_offset
        {
            template <std::size_t dim, class S, class Arg, class... Args>
            static inline auto run(const S& strides, Arg arg, Args... args) noexcept
            {
                return raw_data_offset<dim>(strides, arg, args...);
            }
        };

        template <std::ptrdiff_t static_dim>
        struct layout_data_offset<layout_type::row_major, static_dim>
        {
            using self_type = layout_data_offset<layout_type::row_major, static_dim>;

            template <std::size_t dim, class S, class Arg>
            static inline auto run(const S& strides, Arg arg) noexcept
            {
                if (std::ptrdiff_t(dim) + 1 == static_dim)
                {
                    return arg;
                }
                else
                {
                    return arg * strides[dim];
                }
            }

            template <std::size_t dim, class S, class Arg, class... Args>
            static inline auto run(const S& strides, Arg arg, Args... args) noexcept
            {
                return arg * strides[dim] + self_type::template run<dim + 1>(strides, args...);
            }
        };

        template <std::ptrdiff_t static_dim>
        struct layout_data_offset<layout_type::column_major, static_dim>
        {
            using self_type = layout_data_offset<layout_type::column_major, static_dim>;

            template <std::size_t dim, class S, class Arg>
            static inline auto run(const S& strides, Arg arg) noexcept
            {
                if (dim == 0)
                {
                    return arg;
                }
                else
                {
                    return arg * strides[dim];
                }
            }

            template <std::size_t dim, class S, class Arg, class... Args>
            static inline auto run(const S& strides, Arg arg, Args... args) noexcept
            {
                if (dim == 0)
                {
                    return arg + self_type::template run<dim + 1>(strides, args...);
                }
                else
                {
                    return arg * strides[dim] + self_type::template run<dim + 1>(strides, args...);
                }
            }
        };
    }

    template <class offset_type, class S>
    inline offset_type data_offset(const S&) noexcept
    {
        return offset_type(0);
    }

    template <class offset_type, class S, class Arg, class... Args>
    inline offset_type data_offset(const S& strides, Arg arg, Args... args) noexcept
    {
        constexpr std::size_t nargs = sizeof...(Args) + 1;
        if (nargs == strides.size())
        {
            // Correct number of arguments: iterate
            return static_cast<offset_type>(detail::raw_data_offset<0>(strides, arg, args...));
        }
        else if (nargs > strides.size())
        {
            // Too many arguments: drop the first
            return data_offset<offset_type, S>(strides, args...);
        }
        else
        {
            // Too few arguments: right to left scalar product
            auto view = strides.cend() - nargs;
            return static_cast<offset_type>(detail::raw_data_offset<0>(view, arg, args...));
        }
    }

    template <class offset_type, layout_type L, class S, class... Args>
    inline offset_type unchecked_data_offset(const S& strides, Args... args) noexcept
    {
        return static_cast<offset_type>(
            detail::layout_data_offset<L, static_dimension<S>::value>::template run<0>(
                strides.cbegin(), args...));
    }

    template <class offset_type, class S, class It>
    inline offset_type element_offset(const S& strides, It first, It last) noexcept
    {
        using difference_type = typename std::iterator_traits<It>::difference_type;
        auto size = static_cast<difference_type>((std::min)(
            static_cast<typename S::size_type>(std::distance(first, last)), strides.size()));
        return std::inner_product(last - size, last, strides.cend() - size, offset_type(0));
    }

    namespace detail
    {
        template <class shape_type, class strides_type, class bs_ptr>
        inline void adapt_strides(const shape_type& shape,
                                  strides_type& strides,
                                  bs_ptr backstrides,
                                  typename strides_type::size_type i) noexcept
        {
            if (shape[i] == 1)
            {
                strides[i] = 0;
            }
            (*backstrides)[i] = strides[i] * std::ptrdiff_t(shape[i] - 1);
        }

        template <class shape_type, class strides_type>
        inline void adapt_strides(const shape_type& shape,
                                  strides_type& strides,
                                  std::nullptr_t,
                                  typename strides_type::size_type i) noexcept
        {
            if (shape[i] == 1)
            {
                strides[i] = 0;
            }
        }

        template <layout_type L, class shape_type, class strides_type, class bs_ptr>
        inline std::size_t compute_strides(const shape_type& shape,
                                           layout_type l,
                                           strides_type& strides,
                                           bs_ptr bs)
        {
            using strides_value_type = typename std::decay_t<strides_type>::value_type;
            strides_value_type data_size = 1;
            if (L == layout_type::row_major || l == layout_type::row_major)
            {
                for (std::size_t i = shape.size(); i != 0; --i)
                {
                    strides[i - 1] = data_size;
                    data_size = strides[i - 1] * static_cast<strides_value_type>(shape[i - 1]);
                    adapt_strides(shape, strides, bs, i - 1);
                }
            }
            else
            {
                for (std::size_t i = 0; i < shape.size(); ++i)
                {
                    strides[i] = data_size;
                    data_size = strides[i] * static_cast<strides_value_type>(shape[i]);
                    adapt_strides(shape, strides, bs, i);
                }
            }
            return static_cast<std::size_t>(data_size);
        }
    }

    template <layout_type L, class shape_type, class strides_type>
    inline std::size_t compute_strides(const shape_type& shape,
                                       layout_type l,
                                       strides_type& strides)
    {
        return detail::compute_strides<L>(shape, l, strides, nullptr);
    }

    template <layout_type L, class shape_type, class strides_type, class backstrides_type>
    inline std::size_t compute_strides(const shape_type& shape,
                                       layout_type l,
                                       strides_type& strides,
                                       backstrides_type& backstrides)
    {
        return detail::compute_strides<L>(shape, l, strides, &backstrides);
    }

    template <class T1, class T2>
    inline bool stride_match_condition(const T1& stride,
                                       const T2& shape,
                                       const T1& data_size,
                                       bool zero_strides)
    {
        return (shape == T2(1) && stride == T1(0) && zero_strides) || (stride == data_size);
    }

    // zero_strides should be true when strides are set to 0 if the corresponding dimensions are 1
    template <class shape_type, class strides_type>
    inline bool do_strides_match(const shape_type& shape,
                                 const strides_type& strides,
                                 layout_type l,
                                 bool zero_strides)
    {
        using value_type = typename strides_type::value_type;
        value_type data_size = 1;
        if (l == layout_type::row_major)
        {
            for (std::size_t i = strides.size(); i != 0; --i)
            {
                if (!stride_match_condition(strides[i - 1], shape[i - 1], data_size, zero_strides))
                {
                    return false;
                }
                data_size *= static_cast<value_type>(shape[i - 1]);
            }
            return true;
        }
        else if (l == layout_type::column_major)
        {
            for (std::size_t i = 0; i < strides.size(); ++i)
            {
                if (!stride_match_condition(strides[i], shape[i], data_size, zero_strides))
                {
                    return false;
                }
                data_size *= static_cast<value_type>(shape[i]);
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    template <class shape_type, class strides_type>
    inline void adapt_strides(const shape_type& shape, strides_type& strides) noexcept
    {
        for (typename shape_type::size_type i = 0; i < shape.size(); ++i)
        {
            detail::adapt_strides(shape, strides, nullptr, i);
        }
    }

    template <class shape_type, class strides_type, class backstrides_type>
    inline void adapt_strides(const shape_type& shape,
                              strides_type& strides,
                              backstrides_type& backstrides) noexcept
    {
        for (typename shape_type::size_type i = 0; i < shape.size(); ++i)
        {
            detail::adapt_strides(shape, strides, &backstrides, i);
        }
    }

    namespace detail
    {
        template <class S>
        inline S unravel_noexcept(typename S::value_type idx,
                                  const S& strides,
                                  layout_type l) noexcept
        {
            using value_type = typename S::value_type;
            using size_type = typename S::size_type;
            S result = xtl::make_sequence<S>(strides.size(), 0);
            if (l == layout_type::row_major)
            {
                for (size_type i = 0; i < strides.size(); ++i)
                {
                    value_type str = strides[i];
                    value_type quot = str != 0 ? idx / str : 0;
                    idx = str != 0 ? idx % str : idx;
                    result[i] = quot;
                }
            }
            else
            {
                for (size_type i = strides.size(); i != 0; --i)
                {
                    value_type str = strides[i - 1];
                    value_type quot = str != 0 ? idx / str : 0;
                    idx = str != 0 ? idx % str : idx;
                    result[i - 1] = quot;
                }
            }
            return result;
        }
    }

    template <class S>
    inline S unravel_from_strides(typename S::value_type index, const S& strides, layout_type l)
    {
        if (l != layout_type::row_major && l != layout_type::column_major)
        {
            throw std::runtime_error("unravel_index: dynamic layout not supported");
        }
        return detail::unravel_noexcept(index, strides, l);
    }

    template <class S>
    inline get_strides_t<S> ravel_from_strides(const S& index, const S& strides)
    {
        return element_offset<get_strides_t<S>>(strides, index.begin(), index.end());
    }

    template <class S>
    inline get_strides_t<S> unravel_index(typename S::value_type index,
                                          const S& shape,
                                          layout_type l)
    {
        using strides_type = get_strides_t<S>;
        using strides_value_type = typename strides_type::value_type;
        strides_type strides = xtl::make_sequence<strides_type>(shape.size(), 0);
        compute_strides(shape, l, strides);
        return unravel_from_strides(static_cast<strides_value_type>(index), strides, l);
    }

    template <class S, class T>
    inline std::vector<get_strides_t<S>> unravel_indices(const T& idx,
                                                         const S& shape,
                                                         layout_type l)
    {
        using strides_type = get_strides_t<S>;
        using strides_value_type = typename strides_type::value_type;
        strides_type strides = xtl::make_sequence<strides_type>(shape.size(), 0);
        compute_strides(shape, l, strides);
        std::vector<get_strides_t<S>> out(idx.size());
        auto out_iter = out.begin();
        auto idx_iter = idx.begin();
        for (; out_iter != out.end(); ++out_iter, ++idx_iter)
        {
            *out_iter
                = unravel_from_strides(static_cast<strides_value_type>(*idx_iter), strides, l);
        }
        return out;
    }

    template <class S>
    inline get_value_type<S> ravel_index(const S& index, const S& shape, layout_type l)
    {
        using strides_type = get_strides_t<S>;
        using strides_value_type = typename strides_type::value_type;
        strides_type strides = xtl::make_sequence<strides_type>(shape.size(), 0);
        compute_strides(shape, l, strides);
        return ravel_from_strides(static_cast<strides_value_type>(index), strides);
    }

    template <class S, class stype>
    inline S uninitialized_shape(stype size)
    {
        using value_type = typename S::value_type;
        using size_type = typename S::size_type;
        return xtl::make_sequence<S>(static_cast<size_type>(size),
                                     std::numeric_limits<value_type>::max());
    }

    template <class S1, class S2>
    inline bool broadcast_shape(const S1& input, S2& output)
    {
        bool trivial_broadcast = (input.size() == output.size());
        // Indices are faster than reverse iterators
        using value_type = typename S2::value_type;
        auto output_index = output.size();
        auto input_index = input.size();

        if (output_index < input_index)
        {
            throw_broadcast_error(output, input);
        }
        for (; input_index != 0; --input_index, --output_index)
        {
            // First case: output = (MAX, MAX, ...., MAX)
            // output is a new shape that has not been through
            // the broadcast process yet; broadcast is trivial
            if (output[output_index - 1] == std::numeric_limits<value_type>::max())
            {
                output[output_index - 1] = static_cast<value_type>(input[input_index - 1]);
            }
            // Second case: output has been initialized to 1. Broadcast is trivial
            // only if input is 1 to.
            else if (output[output_index - 1] == 1)
            {
                output[output_index - 1] = static_cast<value_type>(input[input_index - 1]);
                trivial_broadcast = trivial_broadcast && (input[input_index - 1] == 1);
            }
            // Third case: output has been initialized to something different from 1.
            // if input is 1, then the broadcast is not trivial
            else if (input[input_index - 1] == 1)
            {
                trivial_broadcast = false;
            }
            // Last case: input and output must have the same value, else
            // shape are not compatible and an exception is thrown
            else if (static_cast<value_type>(input[input_index - 1]) != output[output_index - 1])
            {
                throw_broadcast_error(output, input);
            }
        }
        return trivial_broadcast;
    }

    template <class S1, class S2>
    inline bool broadcastable(const S1& src_shape, const S2& dst_shape)
    {
        auto src_iter = src_shape.crbegin();
        auto dst_iter = dst_shape.crbegin();
        bool res = dst_shape.size() >= src_shape.size();
        for (; src_iter != src_shape.crend() && res; ++src_iter, ++dst_iter)
        {
            res = (static_cast<std::size_t>(*src_iter) == static_cast<std::size_t>(*dst_iter))
                  || (*src_iter == 1);
        }
        return res;
    }

    template <>
    struct check_strides_overlap<layout_type::row_major>
    {
        template <class S1, class S2>
        static std::size_t get(const S1& s1, const S2& s2)
        {
            using value_type = typename S1::value_type;
            // Indices are faster than reverse iterators
            auto s1_index = s1.size();
            auto s2_index = s2.size();

            for (; s2_index != 0; --s1_index, --s2_index)
            {
                if (static_cast<value_type>(s1[s1_index - 1])
                    != static_cast<value_type>(s2[s2_index - 1]))
                {
                    break;
                }
            }
            return s1_index;
        }
    };

    template <>
    struct check_strides_overlap<layout_type::column_major>
    {
        template <class S1, class S2>
        static std::size_t get(const S1& s1, const S2& s2)
        {
            // Indices are faster than reverse iterators
            using size_type = typename S1::size_type;
            using value_type = typename S1::value_type;
            size_type index = 0;

            // This check is necessary as column major "broadcasting" is still
            // performed in a row major fashion
            if (s1.size() != s2.size())
                return 0;

            auto size = s2.size();

            for (; index < size; ++index)
            {
                if (static_cast<value_type>(s1[index]) != static_cast<value_type>(s2[index]))
                {
                    break;
                }
            }
            return index;
        }
    };

    namespace detail
    {
        template <class S, std::size_t dim>
        inline bool check_in_bounds_impl(const S&)
        {
            return true;
        }

        template <class S, std::size_t dim, class T, class... Args>
        inline bool check_in_bounds_impl(const S& shape, T& arg, Args&... args)
        {
            if (sizeof...(Args) + 1 > shape.size())
            {
                return check_in_bounds_impl<S, dim>(shape, args...);
            }
            else
            {
                return arg >= T(0) && arg < static_cast<T>(shape[dim])
                       && check_in_bounds_impl<S, dim + 1>(shape, args...);
            }
        }
    }

    template <class S, class... Args>
    inline bool check_in_bounds(const S& shape, Args&... args)
    {
        return detail::check_in_bounds_impl<S, 0>(shape, args...);
    }

    namespace detail
    {
        template <class S, std::size_t dim>
        inline void normalize_periodic_impl(const S&)
        {
        }

        template <class S, std::size_t dim, class T, class... Args>
        inline void normalize_periodic_impl(const S& shape, T& arg, Args&... args)
        {
            if (sizeof...(Args) + 1 > shape.size())
            {
                normalize_periodic_impl<S, dim>(shape, args...);
            }
            else
            {
                T n = static_cast<T>(shape[dim]);
                arg = (n + (arg % n)) % n;
                normalize_periodic_impl<S, dim + 1>(shape, args...);
            }
        }
    }

    template <class S, class... Args>
    inline void normalize_periodic(const S& shape, Args&... args)
    {
        check_dimension(shape, args...);
        detail::normalize_periodic_impl<S, 0>(shape, args...);
    }
}

#endif
