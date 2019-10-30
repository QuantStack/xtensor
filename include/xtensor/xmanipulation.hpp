/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_MANIPULATION_HPP
#define XTENSOR_MANIPULATION_HPP

#include "xstrided_view.hpp"
#include "xtensor_config.hpp"
#include "xutils.hpp"

namespace xt
{
    namespace check_policy
    {
        struct none
        {
        };
        struct full
        {
        };
    }

    template <class E>
    auto transpose(E&& e) noexcept;

    template <class E, class S, class Tag = check_policy::none>
    auto transpose(E&& e, S&& permutation, Tag check_policy = Tag());

    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E>
    auto ravel(E&& e);

    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E>
    auto flatten(E&& e);

    template <class E>
    auto trim_zeros(E&& e, const std::string& direction = "fb");

    template <class E>
    auto squeeze(E&& e);

    template <class E, class S, class Tag = check_policy::none, std::enable_if_t<!std::is_integral<S>::value, int> = 0>
    auto squeeze(E&& e, S&& axis, Tag check_policy = Tag());

    /****************************
     * transpose implementation *
     ****************************/

    namespace detail
    {
        inline layout_type transpose_layout_noexcept(layout_type l) noexcept
        {
            layout_type result = l;
            if (l == layout_type::row_major)
            {
                result = layout_type::column_major;
            }
            else if (l == layout_type::column_major)
            {
                result = layout_type::row_major;
            }
            return result;
        }

        inline layout_type transpose_layout(layout_type l)
        {
            if (l != layout_type::row_major && l != layout_type::column_major)
            {
                throw transpose_error("cannot compute transposed layout of dynamic layout");
            }
            return transpose_layout_noexcept(l);
        }

        template <class E, class S>
        inline auto transpose_impl(E&& e, S&& permutation, check_policy::none)
        {
            if (sequence_size(permutation) != e.dimension())
            {
                throw transpose_error("Permutation does not have the same size as shape");
            }

            // permute stride and shape
            using shape_type = xindex_type_t<typename std::decay_t<E>::shape_type>;
            shape_type temp_shape;
            resize_container(temp_shape, e.shape().size());

            using strides_type = get_strides_t<shape_type>;
            strides_type temp_strides;
            resize_container(temp_strides, e.strides().size());

            using size_type = typename std::decay_t<E>::size_type;
            for (std::size_t i = 0; i < e.shape().size(); ++i)
            {
                if (std::size_t(permutation[i]) >= e.dimension())
                {
                    throw transpose_error("Permutation contains wrong axis");
                }
                size_type perm = static_cast<size_type>(permutation[i]);
                temp_shape[i] = e.shape()[perm];
                temp_strides[i] = e.strides()[perm];
            }

            layout_type new_layout = layout_type::dynamic;
            if (std::is_sorted(std::begin(permutation), std::end(permutation)))
            {
                // keep old layout
                new_layout = e.layout();
            }
            else if (std::is_sorted(std::begin(permutation), std::end(permutation), std::greater<>()))
            {
                new_layout = transpose_layout_noexcept(e.layout());
            }

            return strided_view(std::forward<E>(e), std::move(temp_shape), std::move(temp_strides), get_offset<XTENSOR_DEFAULT_LAYOUT>(e), new_layout);
        }

        template <class E, class S>
        inline auto transpose_impl(E&& e, S&& permutation, check_policy::full)
        {
            // check if axis appears twice in permutation
            for (std::size_t i = 0; i < sequence_size(permutation); ++i)
            {
                for (std::size_t j = i + 1; j < sequence_size(permutation); ++j)
                {
                    if (permutation[i] == permutation[j])
                    {
                        throw transpose_error("Permutation contains axis more than once");
                    }
                }
            }
            return transpose_impl(std::forward<E>(e), std::forward<S>(permutation), check_policy::none());
        }

        template <class E, class S, class X, std::enable_if_t<has_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline void compute_transposed_strides(E&& e, const S&, X& strides)
        {
            std::copy(e.strides().crbegin(), e.strides().crend(), strides.begin());
        }

        template <class E, class S, class X, std::enable_if_t<!has_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline void compute_transposed_strides(E&&, const S& shape, X& strides)
        {
            // In the case where E does not have a data interface, the transposition
            // makes use of a flat storage adaptor that has layout XTENSOR_DEFAULT_TRAVERSAL
            // which should be the one inverted.
            layout_type l = transpose_layout(XTENSOR_DEFAULT_TRAVERSAL);
            compute_strides(shape, l, strides);
        }
    }

    /**
     * Returns a transpose view by reversing the dimensions of xexpression e
     * @param e the input expression
     */
    template <class E>
    inline auto transpose(E&& e) noexcept
    {
        using shape_type = xindex_type_t<typename std::decay_t<E>::shape_type>;
        shape_type shape;
        resize_container(shape, e.shape().size());
        std::copy(e.shape().crbegin(), e.shape().crend(), shape.begin());

        get_strides_t<shape_type> strides;
        resize_container(strides, e.shape().size());
        detail::compute_transposed_strides(e, shape, strides);

        layout_type new_layout = detail::transpose_layout_noexcept(e.layout());

        return strided_view(std::forward<E>(e), std::move(shape), std::move(strides), detail::get_offset<XTENSOR_DEFAULT_TRAVERSAL>(e), new_layout);
    }

    /**
     * Returns a transpose view by permuting the xexpression e with @p permutation.
     * @param e the input expression
     * @param permutation the sequence containing permutation
     * @param check_policy the check level (check_policy::full() or check_policy::none())
     * @tparam Tag selects the level of error checking on permutation vector defaults to check_policy::none.
     */
    template <class E, class S, class Tag>
    inline auto transpose(E&& e, S&& permutation, Tag check_policy)
    {
        return detail::transpose_impl(std::forward<E>(e), std::forward<S>(permutation), check_policy);
    }

    /// @cond DOXYGEN_INCLUDE_SFINAE
#ifdef X_OLD_CLANG
    template <class E, class I, class Tag = check_policy::none>
    inline auto transpose(E&& e, std::initializer_list<I> permutation, Tag check_policy = Tag())
    {
        dynamic_shape<I> perm(permutation);
        return detail::transpose_impl(std::forward<E>(e), std::move(perm), check_policy);
    }
#else
    template <class E, class I, std::size_t N, class Tag = check_policy::none>
    inline auto transpose(E&& e, const I(&permutation)[N], Tag check_policy = Tag())
    {
        return detail::transpose_impl(std::forward<E>(e), permutation, check_policy);
    }
#endif
    /// @endcond

    /***************************
     * ravel and flatten views *
     ***************************/

    template <class I, class CI>
    class xiterator_adaptor;

    /**
     * Returns a flatten view of the given expression. No copy is made.
     * @param e the input expression
     * @tparam L the layout used to read the elements of e. If no parameter
     * is specified, XTENSOR_DEFAULT_TRAVERSAL is used.
     * @tparam E the type of the expression
     */
    template <layout_type L, class E>
    inline auto ravel(E&& e)
    {
        using iterator = decltype(e.template begin<L>());
        using const_iterator = decltype(e.template cbegin<L>());
        using adaptor_type = xiterator_adaptor<iterator, const_iterator>;
        constexpr layout_type layout = std::is_pointer<iterator>::value ? L : layout_type::dynamic;
        using type = xtensor_view<adaptor_type, 1, layout, extension::get_expression_tag_t<E>>;
        return type(adaptor_type(e.template begin<L>(), e.template cbegin<L>(), e.size()), { e.size() });
    }

    /**
     * Returns a flatten view of the given expression. No copy is made. This
     * method is equivalent to ravel and is provided for API sameness with
     * Numpy.
     * @param e the input expression
     * @tparam L the layout used to read the elements of e. If no parameter
     * is specified, XTENSOR_DEFAULT_TRAVERSAL is used.
     * @tparam E the type of the expression
     * @sa ravel
     */
    template <layout_type L, class E>
    inline auto flatten(E&& e)
    {
        return ravel<L>(std::forward<E>(e));
    }

    /**
     * Trim zeros at beginning, end or both of 1D sequence.
     *
     * @param e input xexpression
     * @param direction string of either 'f' for trim from beginning, 'b' for trim from end
     *                  or 'fb' (default) for both.
     * @return returns a view without zeros at the beginning and end
     */
    template <class E>
    inline auto trim_zeros(E&& e, const std::string& direction)
    {
        XTENSOR_ASSERT_MSG(e.dimension() == 1, "Dimension for trim_zeros has to be 1.");

        std::ptrdiff_t begin = 0, end = static_cast<std::ptrdiff_t>(e.size());

        auto find_fun = [](const auto& i) {
            return i != 0;
        };

        if (direction.find("f") != std::string::npos)
        {
            begin = std::find_if(e.cbegin(), e.cend(), find_fun) - e.cbegin();
        }

        if (direction.find("b") != std::string::npos && begin != end)
        {
            end -= std::find_if(e.crbegin(), e.crend(), find_fun) - e.crbegin();
        }

        return strided_view(std::forward<E>(e), { range(begin, end) });
    }

    /**
     * Returns a squeeze view of the given expression. No copy is made.
     * Squeezing an expression removes dimensions of extent 1.
     *
     * @param e the input expression
     * @tparam E the type of the expression
     */
    template <class E>
    inline auto squeeze(E&& e)
    {
        dynamic_shape<std::size_t> new_shape;
        dynamic_shape<std::ptrdiff_t> new_strides;
        std::copy_if(e.shape().cbegin(), e.shape().cend(), std::back_inserter(new_shape),
                     [](std::size_t i) { return i != 1; });
        decltype(auto) old_strides = detail::get_strides<XTENSOR_DEFAULT_LAYOUT>(e);
        std::copy_if(old_strides.cbegin(), old_strides.cend(), std::back_inserter(new_strides),
                     [](std::ptrdiff_t i) { return i != 0; });

        return strided_view(std::forward<E>(e), std::move(new_shape), std::move(new_strides), 0, e.layout());
    }

    namespace detail
    {
        template <class E, class S>
        inline auto squeeze_impl(E&& e, S&& axis, check_policy::none)
        {
            std::size_t new_dim = e.dimension() - axis.size();
            dynamic_shape<std::size_t> new_shape(new_dim);
            dynamic_shape<std::ptrdiff_t> new_strides(new_dim);

            decltype(auto) old_strides = detail::get_strides<XTENSOR_DEFAULT_LAYOUT>(e);

            for (std::size_t i = 0, ix = 0; i < e.dimension(); ++i)
            {
                if (axis.cend() == std::find(axis.cbegin(), axis.cend(), i))
                {
                    new_shape[ix] = e.shape()[i];
                    new_strides[ix++] = old_strides[i];
                }
            }

            return strided_view(std::forward<E>(e), std::move(new_shape), std::move(new_strides), 0, e.layout());
        }

        template <class E, class S>
        inline auto squeeze_impl(E&& e, S&& axis, check_policy::full)
        {
            for (auto ix : axis)
            {
                if (static_cast<std::size_t>(ix) > e.dimension())
                {
                    throw std::runtime_error("Axis argument to squeeze > dimension of expression");
                }
                if (e.shape()[static_cast<std::size_t>(ix)] != 1)
                {
                    throw std::runtime_error("Trying to squeeze axis != 1");
                }
            }
            return squeeze_impl(std::forward<E>(e), std::forward<S>(axis), check_policy::none());
        }
    }

    /**
     * @brief Remove single-dimensional entries from the shape of an xexpression
     *
     * @param e input xexpression
     * @param axis integer or container of integers, select a subset of single-dimensional
     *        entries of the shape.
     * @param check_policy select check_policy. With check_policy::full(), selecting an axis
     *        which is greater than one will throw a runtime_error.
     */
    template <class E, class S, class Tag, std::enable_if_t<!std::is_integral<S>::value, int>>
    inline auto squeeze(E&& e, S&& axis, Tag check_policy)
    {
        return detail::squeeze_impl(std::forward<E>(e), std::forward<S>(axis), check_policy);
    }

    /// @cond DOXYGEN_INCLUDE_SFINAE
#ifdef X_OLD_CLANG
    template <class E, class I, class Tag = check_policy::none>
    inline auto squeeze(E&& e, std::initializer_list<I> axis, Tag check_policy = Tag())
    {
        dynamic_shape<I> ax(axis);
        return detail::squeeze_impl(std::forward<E>(e), std::move(ax), check_policy);
    }
#else
    template <class E, class I, std::size_t N, class Tag = check_policy::none>
    inline auto squeeze(E&& e, const I(&axis)[N], Tag check_policy = Tag())
    {
        using arr_t = std::array<I, N>;
        return detail::squeeze_impl(std::forward<E>(e), xtl::forward_sequence<arr_t, decltype(axis)>(axis), check_policy);
    }
#endif

    template <class E, class Tag = check_policy::none>
    inline auto squeeze(E&& e, std::size_t axis, Tag check_policy = Tag())
    {
        return squeeze(std::forward<E>(e), std::array<std::size_t, 1>{ axis }, check_policy);
    }
    /// @endcond

    /**
     * @brief Expand the shape of an xexpression.
     *
     * Insert a new axis that will appear at the axis position in the expanded array shape.
     * This will return a ``strided_view`` with a ``xt::newaxis()`` at the indicated axis.
     *
     * @param e input xexpression
     * @param axis axis to expand
     * @return returns a ``strided_view`` with expanded dimension
     */
    template <class E>
    auto expand_dims(E&& e, std::size_t axis)
    {
        xstrided_slice_vector sv(e.dimension() + 1, all());
        sv[axis] = newaxis();
        return strided_view(std::forward<E>(e), std::move(sv));
    }

    /**
     * Expand dimensions of xexpression to at least `N`
     *
     * This adds ``newaxis()`` slices to a ``strided_view`` until
     * the dimension of the view reaches at least `N`.
     * Note: dimensions are added equally at the beginning and the end.
     * For example, a 1-D array of shape (N,) becomes a view of shape (1, N, 1).
     *
     * @param e input xexpression
     * @tparam N the number of requested dimensions
     * @return ``strided_view`` with expanded dimensions
     */
    template <std::size_t N, class E>
    auto atleast_Nd(E&& e)
    {
        xstrided_slice_vector sv((std::max)(e.dimension(), N), all());
        if (e.dimension() < N)
        {
            std::size_t i = 0;
            std::size_t end = static_cast<std::size_t>(std::round(double(N - e.dimension()) / double(N)));
            for (; i < end; ++i)
            {
                sv[i] = newaxis();
            }
            i += e.dimension();
            for (; i < N; ++i)
            {
                sv[i] = newaxis();
            }
        }
        return strided_view(std::forward<E>(e), std::move(sv));
    }

    /**
     * Expand to at least 1D
     * @sa atleast_Nd
     */
    template <class E>
    auto atleast_1d(E&& e)
    {
        return atleast_Nd<1>(std::forward<E>(e));
    }

    /**
     * Expand to at least 2D
     * @sa atleast_Nd
     */
    template <class E>
    auto atleast_2d(E&& e)
    {
        return atleast_Nd<2>(std::forward<E>(e));
    }

    /**
     * Expand to at least 3D
     * @sa atleast_Nd
     */
    template <class E>
    auto atleast_3d(E&& e)
    {
        return atleast_Nd<3>(std::forward<E>(e));
    }

    /**
     * @brief Split xexpression along axis into subexpressions
     *
     * This splits an xexpression along the axis in `n` equal parts and
     * returns a vector of ``strided_view``.
     * Calling split with axis > dimension of e or a `n` that does not result in
     * an equal division of the xexpression will throw a runtime_error.
     *
     * @param e input xexpression
     * @param n number of elements to return
     * @param axis axis along which to split the expression
     */
    template <class E>
    auto split(E& e, std::size_t n, std::size_t axis = 0)
    {
        if (axis >= e.dimension())
        {
            throw std::runtime_error("Split along axis > dimension.");
        }

        std::size_t ax_sz = e.shape()[axis];
        xstrided_slice_vector sv(e.dimension(), all());
        std::size_t step = ax_sz / n;
        std::size_t rest = ax_sz % n;

        if (rest)
        {
            throw std::runtime_error("Split does not result in equal division.");
        }

        std::vector<decltype(strided_view(e, sv))> result;
        for (std::size_t i = 0; i < n; ++i)
        {
            sv[axis] = range(i * step, (i + 1) * step);
            result.emplace_back(strided_view(e, sv));
        }
        return result;
    }

    /**
     * @brief Reverse the order of elements in an xexpression along the given axis.
     * Note: A NumPy/Matlab style `flipud(arr)` is equivalent to `xt::flip(arr, 0)`,
     * `fliplr(arr)` to `xt::flip(arr, 1)`.
     *
     * @param e the input xexpression
     * @param axis the axis along which elements should be reversed
     *
     * @return returns a view with the result of the flip
     */
    template <class E>
    inline auto flip(E&& e, std::size_t axis)
    {
        using shape_type = xindex_type_t<typename std::decay_t<E>::shape_type>;

        shape_type shape;
        resize_container(shape, e.shape().size());
        std::copy(e.shape().cbegin(), e.shape().cend(), shape.begin());

        get_strides_t<shape_type> strides;
        decltype(auto) old_strides = detail::get_strides<XTENSOR_DEFAULT_LAYOUT>(e);
        resize_container(strides, old_strides.size());
        std::copy(old_strides.cbegin(), old_strides.cend(), strides.begin());

        strides[axis] *= -1;
        std::size_t offset = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(e.data_offset()) + old_strides[axis] * (static_cast<std::ptrdiff_t>(e.shape()[axis]) - 1));

        return strided_view(std::forward<E>(e), std::move(shape), std::move(strides), offset);
    }

    template <std::ptrdiff_t N>
    struct rot90_impl;

    template <>
    struct rot90_impl<0>
    {
        template <class E>
        inline auto operator()(E&& e, const std::array<std::size_t, 2>& /*axes*/)
        {
            return std::forward<E>(e);
        }
    };

    template <>
    struct rot90_impl<1>
    {
        template <class E>
        inline auto operator()(E&& e, const std::array<std::size_t, 2>& axes)
        {
            using std::swap;

            dynamic_shape<std::ptrdiff_t> axes_list(e.shape().size());
            std::iota(axes_list.begin(), axes_list.end(), 0);
            swap(axes_list[axes[0]], axes_list[axes[1]]);

            return transpose(flip(std::forward<E>(e), axes[1]), std::move(axes_list));
        }
    };

    template <>
    struct rot90_impl<2>
    {
        template <class E>
        inline auto operator()(E&& e, const std::array<std::size_t, 2>& axes)
        {
            return flip(flip(std::forward<E>(e), axes[0]), axes[1]);
        }
    };

    template <>
    struct rot90_impl<3>
    {
        template <class E>
        inline auto operator()(E&& e, const std::array<std::size_t, 2>& axes)
        {
            using std::swap;

            dynamic_shape<std::ptrdiff_t> axes_list(e.shape().size());
            std::iota(axes_list.begin(), axes_list.end(), 0);
            swap(axes_list[axes[0]], axes_list[axes[1]]);

            return flip(transpose(std::forward<E>(e), std::move(axes_list)), axes[1]);
        }
    };

    /**
     * @brief Rotate an array by 90 degrees in the plane specified by axes.
     * Rotation direction is from the first towards the second axis.
     *
     * @param e the input xexpression
     * @param axes the array is rotated in the plane defined by the axes. Axes must be different.
     * @tparam N number of times the array is rotated by 90 degrees. Default is 1.
     *
     * @return returns a view with the result of the rotation
     */
    template <std::ptrdiff_t N = 1, class E>
    inline auto rot90(E&& e, const std::array<std::ptrdiff_t, 2>& axes = {0, 1})
    {
        auto ndim = std::ptrdiff_t(e.shape().size());

        if (axes[0] == axes[1] || std::abs(axes[0] - axes[1]) == ndim)
        {
            throw std::runtime_error("Axes must be different");
        }

        auto norm_axes = forward_normalize<std::array<std::size_t, 2>>(e, axes);
        constexpr std::ptrdiff_t n = (4 + (N % 4)) % 4;

        return rot90_impl<n>()(std::forward<E>(e), norm_axes);
    }
}

#endif
