/****************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_SORT_HPP
#define XTENSOR_SORT_HPP

#include <algorithm>
#include <utility>

#include "xarray.hpp"
#include "xeval.hpp"
#include "xslice.hpp"  // for xnone
#include "xmanipulation.hpp"
#include "xtensor.hpp"

namespace xt
{
    namespace detail
    {
        template <class T>
        std::ptrdiff_t adjust_secondary_stride(std::ptrdiff_t stride, T shape)
        {
            return stride != 0 ? stride : static_cast<std::ptrdiff_t>(shape);
        }

        template <class E, class F>
        inline void call_over_leading_axis(E& ev, F&& fct)
        {
            std::size_t n_iters = 1;
            std::ptrdiff_t secondary_stride;

            if (ev.layout() == layout_type::row_major)
            {
                n_iters = std::accumulate(
                    ev.shape().begin(), ev.shape().end() - 1, std::size_t(1), std::multiplies<>());
                secondary_stride = adjust_secondary_stride(ev.strides()[ev.dimension() - 2],
                                                           *(ev.shape().end() - 1));
            }
            else
            {
                n_iters = std::accumulate(
                    ev.shape().begin() + 1, ev.shape().end(), std::size_t(1), std::multiplies<>());
                secondary_stride = adjust_secondary_stride(ev.strides()[1], *(ev.shape().begin()));
            }

            std::ptrdiff_t offset = 0;

            for (std::size_t i = 0; i < n_iters; ++i, offset += secondary_stride)
            {
                fct(ev.data() + offset, ev.data() + offset + secondary_stride);
            }
        }

        template <class E>
        inline std::size_t leading_axis(const E& e)
        {
            if (e.layout() == layout_type::row_major)
            {
                return e.dimension() - 1;
            }
            else if (e.layout() == layout_type::column_major)
            {
                return 0;
            }
            throw std::runtime_error("Layout not supported.");
        }

        // get permutations to transpose and reverse-transpose array
        inline std::pair<dynamic_shape<std::size_t>, dynamic_shape<std::size_t>> get_permutations(
            std::size_t dim, std::size_t ax, layout_type layout)
        {
            dynamic_shape<std::size_t> permutation(dim);
            std::iota(permutation.begin(), permutation.end(), std::size_t(0));
            permutation.erase(permutation.begin() + std::ptrdiff_t(ax));

            if (layout == layout_type::row_major)
            {
                permutation.push_back(ax);
            }
            else
            {
                permutation.insert(permutation.begin(), ax);
            }

            // TODO find a more clever way to get reverse permutation?
            dynamic_shape<std::size_t> reverse_permutation;
            for (std::size_t i = 0; i < dim; ++i)
            {
                auto it = std::find(permutation.begin(), permutation.end(), i);
                reverse_permutation.push_back(std::size_t(std::distance(permutation.begin(), it)));
            }

            return std::make_pair(std::move(permutation), std::move(reverse_permutation));
        }

        template <class E, class R, class F>
        inline auto run_lambda_over_axis(const E& e, R& res, std::size_t axis, F&& lambda)
        {
            if (axis != detail::leading_axis(res))
            {
                dynamic_shape<std::size_t> permutation, reverse_permutation;
                std::tie(permutation, reverse_permutation)
                    = get_permutations(e.dimension(), axis, e.layout());

                res = transpose(e, permutation);
                detail::call_over_leading_axis(res, std::forward<F>(lambda));
                res = transpose(res, reverse_permutation);
            }
            else
            {
                res = e;
                detail::call_over_leading_axis(res, std::forward<F>(lambda));
            }
        }

        template <class VT>
        struct flatten_sort_result_type_impl
        {
            using type = VT;
        };

        template <class VT, std::size_t N, layout_type L>
        struct flatten_sort_result_type_impl<xtensor<VT, N, L>>
        {
            using type = xtensor<VT, 1, L>;
        };

        template <class VT, class S, layout_type L>
        struct flatten_sort_result_type_impl<xtensor_fixed<VT, S, L>>
        {
            using type = xtensor_fixed<VT, xshape<fixed_compute_size<S>::value>, L>;
        };

        template <class VT>
        struct flatten_sort_result_type : flatten_sort_result_type_impl<common_tensor_type_t<VT>>
        {
        };

        template <class VT>
        using flatten_sort_result_type_t = typename flatten_sort_result_type<VT>::type;

        template <class E, class R = flatten_sort_result_type_t<E>>
        inline auto flat_sort_impl(const xexpression<E>& e)
        {
            const auto& de = e.derived_cast();
            R ev;
            ev.resize({ de.size() });

            std::copy(de.cbegin(), de.cend(), ev.begin());
            std::sort(ev.begin(), ev.end());

            return ev;
        }
    }

    template <class E>
    inline auto sort(const xexpression<E>& e, placeholders::xtuph /*t*/)
    {
        return detail::flat_sort_impl(e);
    }

    namespace detail
    {
        template <class T>
        struct sort_eval_type
        {
            using type = typename T::temporary_type;
        };

        template <class T, std::size_t... I, layout_type L>
        struct sort_eval_type<xtensor_fixed<T, fixed_shape<I...>, L>>
        {
            using type = xtensor<T, sizeof...(I), L>;
        };
    }

    /**
     * Sort xexpression (optionally along axis)
     * The sort is performed using the ``std::sort`` functions.
     * A copy of the xexpression is created and returned.
     *
     * @param e xexpression to sort
     * @param axis axis along which sort is performed
     *
     * @return sorted array (copy)
     */
    template <class E>
    inline auto sort(const xexpression<E>& e, std::ptrdiff_t axis = -1)
    {
        using eval_type = typename detail::sort_eval_type<E>::type;

        const auto& de = e.derived_cast();

        if (de.dimension() == 1)
        {
            return detail::flat_sort_impl<std::decay_t<decltype(de)>, eval_type>(de);
        }

        std::size_t ax = normalize_axis(de.dimension(), axis);

        eval_type res;
        detail::run_lambda_over_axis(
            de, res, ax, [](auto begin, auto end) { std::sort(begin, end); });
        return res;
    }

    namespace detail
    {
        template <class VT, class T>
        struct rebind_value_type
        {
            using type = xarray<VT, xt::layout_type::dynamic>;
        };

        template <class VT, class EC, layout_type L>
        struct rebind_value_type<VT, xarray<EC, L>>
        {
            using type = xarray<VT, L>;
        };

        template <class VT, class EC, std::size_t N, layout_type L>
        struct rebind_value_type<VT, xtensor<EC, N, L>>
        {
            using type = xtensor<VT, N, L>;
        };

        template <class VT, class ET, class S, layout_type L>
        struct rebind_value_type<VT, xtensor_fixed<ET, S, L>>
        {
            using type = xtensor_fixed<VT, S, L>;
        };

        template <class VT, class T>
        struct flatten_rebind_value_type
        {
            using type = typename rebind_value_type<VT, T>::type;
        };

        template <class VT, class EC, std::size_t N, layout_type L>
        struct flatten_rebind_value_type<VT, xtensor<EC, N, L>>
        {
            using type = xtensor<VT, 1, L>;
        };

        template <class VT, class ET, class S, layout_type L>
        struct flatten_rebind_value_type<VT, xtensor_fixed<ET, S, L>>
        {
            using type = xtensor_fixed<VT, xshape<fixed_compute_size<S>::value>, L>;
        };

        template <class T>
        struct argsort_result_type
        {
            using type = typename rebind_value_type<typename T::temporary_type::size_type,
                                                    typename T::temporary_type>::type;
        };

        template <class T>
        struct linear_argsort_result_type
        {
            using type = typename flatten_rebind_value_type<typename T::temporary_type::size_type,
                                                            typename T::temporary_type>::type;
        };

        template <class Ed, class Ei>
        inline void argsort_over_leading_axis(const Ed& data, Ei& inds)
        {
            std::size_t n_iters = 1;
            std::ptrdiff_t data_secondary_stride, inds_secondary_stride;

            if (data.layout() == layout_type::row_major)
            {
                n_iters = std::accumulate(data.shape().begin(),
                                          data.shape().end() - 1,
                                          std::size_t(1),
                                          std::multiplies<>());
                data_secondary_stride = data.strides()[data.dimension() - 2];
                inds_secondary_stride = inds.strides()[inds.dimension() - 2];
            }
            else
            {
                n_iters = std::accumulate(data.shape().begin() + 1,
                                          data.shape().end(),
                                          std::size_t(1),
                                          std::multiplies<>());
                data_secondary_stride = data.strides()[1];
                inds_secondary_stride = inds.strides()[1];
            }

            auto ptr = data.data();
            auto indices_ptr = inds.data();

            for (std::size_t i = 0; i < n_iters;
                 ++i, ptr += data_secondary_stride, indices_ptr += inds_secondary_stride)
            {
                auto comp
                    = [&ptr](std::size_t x, std::size_t y) { return *(ptr + x) < *(ptr + y); };
                std::iota(indices_ptr, indices_ptr + inds_secondary_stride, 0);
                std::sort(indices_ptr, indices_ptr + inds_secondary_stride, comp);
            }
        }

        template <class E, class R = typename detail::linear_argsort_result_type<E>::type>
        inline auto flatten_argsort_impl(const xexpression<E>& e)
        {
            const auto& de = e.derived_cast();

            auto cit = de.template begin<layout_type::row_major>();
            using const_iterator = decltype(cit);
            auto ad = xiterator_adaptor<const_iterator, const_iterator>(cit, cit, de.size());

            using result_type = R;
            result_type result;
            result.resize({ de.size() });
            auto comp = [&ad](std::size_t x, std::size_t y) { return ad[x] < ad[y]; };
            std::iota(result.begin(), result.end(), 0);
            std::sort(result.begin(), result.end(), comp);

            return result;
        }
    }

    template <class E>
    inline auto argsort(const xexpression<E>& e, placeholders::xtuph /*t*/)
    {
        return detail::flatten_argsort_impl(e);
    }

    /**
     * Argsort xexpression (optionally along axis)
     * Performs an indirect sort along the given axis. Returns an xarray
     * of indices of the same shape as e that index data along the given axis in
     * sorted order.
     *
     * @param e xexpression to argsort
     * @param axis axis along which argsort is performed
     *
     * @return argsorted index array
     */
    template <class E>
    inline auto argsort(const xexpression<E>& e, std::ptrdiff_t axis = -1)
    {
        using eval_type = typename detail::sort_eval_type<E>::type;
        using result_type = typename detail::argsort_result_type<eval_type>::type;

        const auto& de = e.derived_cast();

        std::size_t ax = normalize_axis(de.dimension(), axis);

        if (de.dimension() == 1)
        {
            return detail::flatten_argsort_impl<E, result_type>(e);
        }

        if (ax != detail::leading_axis(de))
        {
            dynamic_shape<std::size_t> permutation, reverse_permutation;
            std::tie(permutation, reverse_permutation)
                = detail::get_permutations(de.dimension(), ax, de.layout());

            eval_type ev = transpose(de, permutation);
            result_type res = result_type::from_shape(ev.shape());
            detail::argsort_over_leading_axis(ev, res);
            res = transpose(res, reverse_permutation);
            return res;
        }
        else
        {
            result_type res = result_type::from_shape(de.shape());
            detail::argsort_over_leading_axis(de, res);
            return res;
        }
    }

    /************************************************
     * Implementation of partition and argpartition *
     ************************************************/

    /**
     * Partially sort xexpression
     *
     * Partition shuffles the xexpression in a way so that the kth element
     * in the returned xexpression is in the place it would appear in a sorted
     * array and all elements smaller than this entry are placed (unsorted) before.
     *
     * The optional third parameter can either be an axis or ``xnone()`` in which case
     * the xexpression will be flattened.
     *
     * This function uses ``std::nth_element`` internally.
     *
     * \code{cpp}
     * xt::xarray<float> a = {1, 10, -10, 123};
     * std::cout << xt::partition(a, 0) << std::endl; // {-10, 1, 123, 10} the correct entry at
     * index 0 std::cout << xt::partition(a, 3) << std::endl; // {1, 10, -10, 123} the correct entry
     * at index 3 std::cout << xt::partition(a, {0, 3}) << std::endl; // {-10, 1, 10, 123} the
     * correct entries at index 0 and 3 \endcode
     *
     * @param e input xexpression
     * @param kth_container a container of ``indices`` that should contain the correctly sorted
     * value
     * @param axis either integer (default = -1) to sort along last axis or ``xnone()`` to flatten
     * before sorting
     *
     * @return partially sorted xcontainer
     */
    template <class E,
              class C,
              class R = detail::flatten_sort_result_type_t<E>,
              class = std::enable_if_t<!std::is_integral<C>::value, int>>
    inline R partition(const xexpression<E>& e, const C& kth_container, placeholders::xtuph /*ax*/)
    {
        const auto& de = e.derived_cast();

        R ev = R::from_shape({ de.size() });
        C kth_copy = kth_container;
        if (kth_copy.size() > 1)
        {
            std::sort(kth_copy.begin(), kth_copy.end());
        }

        std::copy(de.storage_cbegin(), de.storage_cend(), ev.storage_begin());  // flatten
        std::size_t k_last = kth_copy.back();
        std::nth_element(ev.storage_begin(), ev.storage_begin() + k_last, ev.storage_end());

        for (auto it = (kth_copy.rbegin() + 1); it != kth_copy.rend(); ++it)
        {
            std::nth_element(
                ev.storage_begin(), ev.storage_begin() + *it, ev.storage_begin() + k_last);
            k_last = *it;
        }

        return ev;
    }

#ifdef X_OLD_CLANG
    template <class E, class I, class R = detail::flatten_sort_result_type_t<E>>
    inline R partition(const xexpression<E>& e,
                       std::initializer_list<I> kth_container,
                       placeholders::xtuph tag)
    {
        return partition(
            e,
            xtl::forward_sequence<std::vector<std::size_t>, decltype(kth_container)>(kth_container),
            tag);
    }
#else
    template <class E, class I, std::size_t N, class R = detail::flatten_sort_result_type_t<E>>
    inline R partition(const xexpression<E>& e,
                       const I (&kth_container)[N],
                       placeholders::xtuph tag)
    {
        return partition(e,
                         xtl::forward_sequence<std::array<std::size_t, N>, decltype(kth_container)>(
                             kth_container),
                         tag);
    }
#endif

    template <class E, class R = detail::flatten_sort_result_type_t<E>>
    inline R partition(const xexpression<E>& e, std::size_t kth, placeholders::xtuph tag)
    {
        return partition(e, std::array<std::size_t, 1>({ kth }), tag);
    }

    template <class E, class C, class = std::enable_if_t<!std::is_integral<C>::value, int>>
    inline auto partition(const xexpression<E>& e, const C& kth_container, std::ptrdiff_t axis = -1)
    {
        using eval_type = typename detail::sort_eval_type<E>::type;

        const auto& de = e.derived_cast();

        if (de.dimension() == 1)
        {
            return partition<E, C, eval_type>(de, kth_container, xnone());
        }

        C kth_copy = kth_container;
        if (kth_copy.size() > 1)
        {
            std::sort(kth_copy.begin(), kth_copy.end());
        }

        std::size_t ax = normalize_axis(de.dimension(), axis);

        eval_type res;

        std::size_t kth = kth_copy.back();

        dynamic_shape<std::size_t> permutation, reverse_permutation;
        bool is_leading_axis = (ax == detail::leading_axis(res));

        if (!is_leading_axis)
        {
            std::tie(permutation, reverse_permutation)
                = detail::get_permutations(de.dimension(), ax, de.layout());
            res = transpose(de, permutation);
        }
        else
        {
            res = de;
        }

        auto lambda = [&kth](auto begin, auto end) { std::nth_element(begin, begin + kth, end); };
        detail::call_over_leading_axis(res, lambda);

        for (auto it = kth_copy.rbegin() + 1; it != kth_copy.rend(); ++it)
        {
            kth = *it;
            detail::call_over_leading_axis(res, lambda);
        }

        if (!is_leading_axis)
        {
            res = transpose(res, reverse_permutation);
        }

        return res;
    }

#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto partition(const xexpression<E>& e,
                          std::initializer_list<I> kth_container,
                          std::ptrdiff_t axis = -1)
    {
        return partition(
            e,
            xtl::forward_sequence<std::vector<std::size_t>, decltype(kth_container)>(kth_container),
            axis);
    }
#else
    template <class E, class T, std::size_t N>
    inline auto partition(const xexpression<E>& e,
                          const T (&kth_container)[N],
                          std::ptrdiff_t axis = -1)
    {
        return partition(e,
                         xtl::forward_sequence<std::array<std::size_t, N>, decltype(kth_container)>(
                             kth_container),
                         axis);
    }
#endif
    template <class E>
    inline auto partition(const xexpression<E>& e, std::size_t kth, std::ptrdiff_t axis = -1)
    {
        return partition(e, std::array<std::size_t, 1>({ kth }), axis);
    }

    /**
     * Partially sort arguments
     *
     * Argpartition shuffles the indices to a xexpression in a way so that the index for the
     * kth element in the returned xexpression is in the place it would appear in a sorted
     * array and all elements smaller than this entry are placed (unsorted) before.
     *
     * The optional third parameter can either be an axis or ``xnone()`` in which case
     * the xexpression will be flattened.
     *
     * This function uses ``std::nth_element`` internally.
     *
     * \code{cpp}
     * xt::xarray<float> a = {1, 10, -10, 123};
     * std::cout << xt::argpartition(a, 0) << std::endl; // {2, 0, 3, 1} the correct entry at index
     * 0 std::cout << xt::argpartition(a, 3) << std::endl; // {0, 1, 2, 3} the correct entry at
     * index 3 std::cout << xt::argpartition(a, {0, 3}) << std::endl; // {2, 0, 1, 3} the correct
     * entries at index 0 and 3 \endcode
     *
     * @param e input xexpression
     * @param kth_container a container of ``indices`` that should contain the correctly sorted
     * value
     * @param axis either integer (default = -1) to sort along last axis or ``xnone()`` to flatten
     * before sorting
     *
     * @return xcontainer with indices of partial sort of input
     */
    template <class E,
              class C,
              class R = typename detail::linear_argsort_result_type<
                  typename detail::sort_eval_type<E>::type>::type,
              class = std::enable_if_t<!std::is_integral<C>::value, int>>
    inline R argpartition(const xexpression<E>& e, const C& kth_container, placeholders::xtuph)
    {
        using eval_type = typename detail::sort_eval_type<E>::type;
        using result_type = typename detail::linear_argsort_result_type<eval_type>::type;

        const auto& de = e.derived_cast();

        result_type ev = result_type::from_shape({ de.size() });

        C kth_copy = kth_container;
        if (kth_copy.size() > 1)
        {
            std::sort(kth_copy.begin(), kth_copy.end());
        }

        auto arg_lambda = [&de](std::size_t a, std::size_t b) { return de[a] < de[b]; };

        std::iota(ev.storage_begin(), ev.storage_end(), 0);
        std::size_t k_last = kth_copy.back();
        std::nth_element(
            ev.storage_begin(), ev.storage_begin() + k_last, ev.storage_end(), arg_lambda);

        for (auto it = (kth_copy.rbegin() + 1); it != kth_copy.rend(); ++it)
        {
            std::nth_element(ev.storage_begin(),
                             ev.storage_begin() + *it,
                             ev.storage_begin() + k_last,
                             arg_lambda);
            k_last = *it;
        }

        return ev;
    }

#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto argpartition(const xexpression<E>& e,
                             std::initializer_list<I> kth_container,
                             placeholders::xtuph tag)
    {
        return argpartition(
            e,
            xtl::forward_sequence<std::vector<std::size_t>, decltype(kth_container)>(kth_container),
            tag);
    }
#else
    template <class E, class I, std::size_t N>
    inline auto argpartition(const xexpression<E>& e,
                             const I (&kth_container)[N],
                             placeholders::xtuph tag)
    {
        return argpartition(
            e,
            xtl::forward_sequence<std::array<std::size_t, N>, decltype(kth_container)>(
                kth_container),
            tag);
    }
#endif

    template <class E>
    inline auto argpartition(const xexpression<E>& e, std::size_t kth, placeholders::xtuph tag)
    {
        return argpartition(e, std::array<std::size_t, 1>({ kth }), tag);
    }

    namespace detail
    {
        template <class Ed, class Ei>
        inline void argpartition_over_leading_axis(const Ed& data,
                                                   Ei& inds,
                                                   std::size_t kth,
                                                   std::ptrdiff_t last)
        {
            std::size_t n_iters = 1;
            std::ptrdiff_t data_secondary_stride, inds_secondary_stride;

            if (data.layout() == layout_type::row_major)
            {
                n_iters = std::accumulate(data.shape().begin(),
                                          data.shape().end() - 1,
                                          std::size_t(1),
                                          std::multiplies<>());
                data_secondary_stride = data.strides()[data.dimension() - 2];
                inds_secondary_stride = inds.strides()[inds.dimension() - 2];
            }
            else
            {
                n_iters = std::accumulate(data.shape().begin() + 1,
                                          data.shape().end(),
                                          std::size_t(1),
                                          std::multiplies<>());
                data_secondary_stride = data.strides()[1];
                inds_secondary_stride = inds.strides()[1];
            }

            auto ptr = data.data();
            auto indices_ptr = inds.data();
            auto comp = [&ptr](std::size_t x, std::size_t y) { return *(ptr + x) < *(ptr + y); };

            if (last == -1)  // initialize
            {
                for (std::size_t i = 0; i < n_iters;
                     ++i, ptr += data_secondary_stride, indices_ptr += inds_secondary_stride)
                {
                    std::iota(indices_ptr, indices_ptr + inds_secondary_stride, 0);
                    std::nth_element(
                        indices_ptr, indices_ptr + kth, indices_ptr + inds_secondary_stride, comp);
                }
            }
            else
            {
                for (std::size_t i = 0; i < n_iters;
                     ++i, ptr += data_secondary_stride, indices_ptr += inds_secondary_stride)
                {
                    std::nth_element(indices_ptr, indices_ptr + kth, indices_ptr + last, comp);
                }
            }
        }
    }

    template <class E, class C, class = std::enable_if_t<!std::is_integral<C>::value, int>>
    inline auto argpartition(const xexpression<E>& e,
                             const C& kth_container,
                             std::ptrdiff_t axis = -1)
    {
        using eval_type = typename detail::sort_eval_type<E>::type;
        using result_type = typename detail::argsort_result_type<eval_type>::type;

        const auto& de = e.derived_cast();

        std::size_t ax = normalize_axis(de.dimension(), axis);

        if (de.dimension() == 1)
        {
            return argpartition<E, C, result_type>(e, kth_container, xnone());
        }

        C kth_copy = kth_container;
        if (kth_copy.size() > 1)
        {
            std::sort(kth_copy.begin(), kth_copy.end());
        }

        eval_type ev;
        result_type res;

        dynamic_shape<std::size_t> permutation, reverse_permutation;
        bool is_leading_axis = (ax == detail::leading_axis(res));

        if (!is_leading_axis)
        {
            std::tie(permutation, reverse_permutation)
                = detail::get_permutations(de.dimension(), ax, de.layout());
            ev = transpose(de, permutation);
        }
        else
        {
            ev = de;
        }
        res.resize(ev.shape());

        std::size_t kth = kth_copy.back();
        detail::argpartition_over_leading_axis(ev, res, kth, -1);

        for (auto it = (kth_copy.rbegin() + 1); it != kth_copy.rend(); ++it)
        {
            detail::argpartition_over_leading_axis(ev, res, *it, static_cast<std::ptrdiff_t>(kth));
            kth = *it;
        }

        if (!is_leading_axis)
        {
            res = transpose(res, reverse_permutation);
        }

        return res;
    }

#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto argpartition(const xexpression<E>& e,
                             std::initializer_list<I> kth_container,
                             std::ptrdiff_t axis = -1)
    {
        return argpartition(
            e,
            xtl::forward_sequence<std::vector<std::size_t>, decltype(kth_container)>(kth_container),
            axis);
    }
#else
    template <class E, class I, std::size_t N>
    inline auto argpartition(const xexpression<E>& e,
                             const I (&kth_container)[N],
                             std::ptrdiff_t axis = -1)
    {
        return argpartition(
            e,
            xtl::forward_sequence<std::array<std::size_t, N>, decltype(kth_container)>(
                kth_container),
            axis);
    }
#endif

    template <class E>
    inline auto argpartition(const xexpression<E>& e, std::size_t kth, std::ptrdiff_t axis = -1)
    {
        return argpartition(e, std::array<std::size_t, 1>({ kth }), axis);
    }

    template <class E>
    inline typename std::decay_t<E>::value_type median(E&& e)
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto sz = e.size();
        if (sz % 2 == 0)
        {
            std::size_t szh = sz / 2;  // integer floor div
            std::array<std::size_t, 2> kth = { szh - 1, szh };
            auto values = xt::partition(xt::flatten(e), kth);
            return (values[kth[0]] + values[kth[1]]) / value_type(2);
        }
        else
        {
            std::array<std::size_t, 1> kth = { (sz - 1) / 2 };
            auto values = xt::partition(xt::flatten(e), kth);
            return values[kth[0]];
        }
    }

    /**
     * Find the median along the specified axis
     *
     * Given a vector V of length N, the median of V is the middle value of a
     * sorted copy of V, V_sorted - i e., V_sorted[(N-1)/2], when N is odd,
     * and the average of the two middle values of V_sorted when N is even.
     *
     * @param axis axis along which the medians are computed.
     *             If not set, computes the median along a flattened version of the input.
     * @param e input xexpression
     * @return median value
     */
    template <class E>
    inline auto median(E&& e, std::ptrdiff_t axis)
    {
        std::size_t ax = normalize_axis(e.dimension(), axis);
        std::size_t sz = e.shape()[ax];
        xstrided_slice_vector sv(e.dimension(), xt::all());

        if (sz % 2 == 0)
        {
            std::size_t szh = sz / 2;  // integer floor div
            std::array<std::size_t, 2> kth = { szh - 1, szh };
            auto values = xt::partition(std::forward<E>(e), kth, static_cast<ptrdiff_t>(ax));
            sv[ax] = xt::range(szh - 1, szh + 1);
            return xt::mean(xt::strided_view(std::move(values), std::move(sv)), { ax });
        }
        else
        {
            std::size_t szh = (sz - 1) / 2;
            std::array<std::size_t, 1> kth = { (sz - 1) / 2 };
            auto values = xt::partition(std::forward<E>(e), kth, static_cast<ptrdiff_t>(ax));
            sv[ax] = xt::range(szh, szh + 1);
            return xt::mean(xt::strided_view(std::move(values), std::move(sv)), { ax });
        }
    }

    namespace detail
    {
        template <class T>
        struct argfunc_result_type
        {
            using type = xarray<std::size_t>;
        };

        template <class T, std::size_t N>
        struct argfunc_result_type<xtensor<T, N>>
        {
            using type = xtensor<std::size_t, N - 1>;
        };

        template <class IT, class F>
        inline std::size_t cmp_idx(IT iter, IT end, std::ptrdiff_t inc, F&& cmp)
        {
            std::size_t idx = 0;
            auto min = *iter;
            iter += inc;
            for (std::size_t i = 1; iter < end; iter += inc, ++i)
            {
                if (cmp(*iter, min))
                {
                    min = *iter;
                    idx = i;
                }
            }
            return idx;
        }

        template <layout_type L, class E, class F>
        inline xtensor<std::size_t, 0> arg_func_impl(const E& e, F&& f)
        {
            return cmp_idx(e.template begin<L>(), e.template end<L>(), 1, std::forward<F>(f));
        }

        template <layout_type L, class E, class F>
        inline typename argfunc_result_type<E>::type arg_func_impl(const E& e,
                                                                   std::size_t axis,
                                                                   F&& cmp)
        {
            using eval_type = typename detail::sort_eval_type<E>::type;
            using value_type = typename E::value_type;
            using result_type = typename argfunc_result_type<E>::type;
            using result_shape_type = typename result_type::shape_type;

            if (e.dimension() == 1)
            {
                return arg_func_impl<L>(e, std::forward<F>(cmp));
            }

            result_shape_type alt_shape;
            xt::resize_container(alt_shape, e.dimension() - 1);

            // Excluding copy, copy all of shape except for axis
            std::copy(
                e.shape().cbegin(), e.shape().cbegin() + std::ptrdiff_t(axis), alt_shape.begin());
            std::copy(e.shape().cbegin() + std::ptrdiff_t(axis) + 1,
                      e.shape().cend(),
                      alt_shape.begin() + std::ptrdiff_t(axis));

            result_type result = result_type::from_shape(std::move(alt_shape));
            auto result_iter = result.template begin<L>();

            auto arg_func_lambda = [&result_iter, &cmp](auto begin, auto end) {
                std::size_t idx = 0;
                value_type val = *begin;
                ++begin;
                for (std::size_t i = 1; begin != end; ++begin, ++i)
                {
                    if (cmp(*begin, val))
                    {
                        val = *begin;
                        idx = i;
                    }
                }
                *result_iter = idx;
                ++result_iter;
            };

            if (axis != detail::leading_axis(e))
            {
                dynamic_shape<std::size_t> permutation, reverse_permutation;
                std::tie(permutation, reverse_permutation)
                    = detail::get_permutations(e.dimension(), axis, e.layout());

                // note: creating copy
                eval_type input = transpose(e, permutation);
                detail::call_over_leading_axis(input, arg_func_lambda);
                return result;
            }
            else
            {
                auto&& input = eval(e);
                detail::call_over_leading_axis(input, arg_func_lambda);
                return result;
            }
        }
    }

    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E>
    inline auto argmin(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        return detail::arg_func_impl<L>(ed, std::less<value_type>());
    }

    /**
     * Find position of minimal value in xexpression
     *
     * @param e input xexpression
     * @param axis select axis (or none)
     *
     * @return returns xarray with positions of minimal value
     */
    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E>
    inline auto argmin(const xexpression<E>& e, std::ptrdiff_t axis)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        std::size_t ax = normalize_axis(ed.dimension(), axis);
        return detail::arg_func_impl<L>(ed, ax, std::less<value_type>());
    }

    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E>
    inline auto argmax(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        return detail::arg_func_impl<L>(ed, std::greater<value_type>());
    }

    /**
     * Find position of maximal value in xexpression
     *
     * @param e input xexpression
     * @param axis select axis (or none)
     *
     * @return returns xarray with positions of maximal value
     */
    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E>
    inline auto argmax(const xexpression<E>& e, std::ptrdiff_t axis)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        std::size_t ax = normalize_axis(ed.dimension(), axis);
        return detail::arg_func_impl<L>(ed, ax, std::greater<value_type>());
    }

    /**
     * Find unique elements of a xexpression. This returns a flattened xtensor with
     * sorted, unique elements from the original expression.
     *
     * @param e input xexpression (will be flattened)
     */
    template <class E>
    inline auto unique(const xexpression<E>& e)
    {
        auto sorted = sort(e, xnone());
        auto end = std::unique(sorted.begin(), sorted.end());
        std::size_t sz = static_cast<std::size_t>(std::distance(sorted.begin(), end));
        // TODO check if we can shrink the vector without reallocation
        using value_type = typename E::value_type;
        auto result = xtensor<value_type, 1>::from_shape({ sz });
        std::copy(sorted.begin(), end, result.begin());
        return result;
    }

    /**
     * Find the set difference of two xexpressions. This returns a flattened xtensor with
     * the sorted, unique values in ar1 that are not in ar2.
     *
     * @param ar1 input xexpression (will be flattened)
     * @param ar2 input xexpression
     */
    template <class E1, class E2>
    inline auto setdiff1d(const xexpression<E1>& ar1, const xexpression<E2>& ar2)
    {
        using value_type = typename E1::value_type;

        auto unique1 = unique(ar1);
        auto unique2 = unique(ar2);

        auto tmp = xtensor<value_type, 1>::from_shape({ unique1.size() });

        auto end = std::set_difference(
            unique1.begin(), unique1.end(), unique2.begin(), unique2.end(), tmp.begin());

        std::size_t sz = static_cast<std::size_t>(std::distance(tmp.begin(), end));

        auto result = xtensor<value_type, 1>::from_shape({ sz });

        std::copy(tmp.begin(), end, result.begin());

        return result;
    }
}

#endif
