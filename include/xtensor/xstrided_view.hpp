/****************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_STRIDED_VIEW_HPP
#define XTENSOR_STRIDED_VIEW_HPP

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <xtl/xsequence.hpp>
#include <xtl/xvariant.hpp>

#include "xexpression.hpp"
#include "xiterable.hpp"
#include "xlayout.hpp"
#include "xsemantic.hpp"
#include "xstorage.hpp"
#include "xstrided_view_base.hpp"
#include "xutils.hpp"

namespace xt
{
    /***************************
     * xstrided_view extension *
     ***************************/

    namespace extension
    {
        template <class Tag, class CT, class S, layout_type L, class FST>
        struct xstrided_view_base_impl;

        template <class CT, class S, layout_type L, class FST>
        struct xstrided_view_base_impl<xtensor_expression_tag, CT, S, L, FST>
        {
            using type = xtensor_empty_base;
        };

        template <class CT, class S, layout_type L, class FST>
        struct xstrided_view_base : xstrided_view_base_impl<xexpression_tag_t<CT>, CT, S, L, FST>
        {
        };

        template <class CT, class S, layout_type L, class FST>
        using xstrided_view_base_t = typename xstrided_view_base<CT, S, L, FST>::type;
    }

    template <class CT, class S, layout_type L, class FST>
    class xstrided_view;

    template <class CT, class S, layout_type L, class FST>
    struct xcontainer_inner_types<xstrided_view<CT, S, L, FST>>
    {
        using xexpression_type = std::decay_t<CT>;
        using undecay_expression = CT;
        using reference = inner_reference_t<undecay_expression>;
        using const_reference = typename xexpression_type::const_reference;
        using size_type = typename xexpression_type::size_type;
        using shape_type = std::decay_t<S>;
        using undecay_shape = S;
        using storage_getter = FST;
        using inner_storage_type = typename storage_getter::type;
        using temporary_type = temporary_type_t<typename xexpression_type::value_type, S, L>;
        static constexpr layout_type layout = L;
    };

    template <class CT, class S, layout_type L, class FST>
    struct xiterable_inner_types<xstrided_view<CT, S, L, FST>>
    {
        using inner_shape_type = std::decay_t<S>;
        using inner_strides_type = get_strides_t<inner_shape_type>;
        using inner_backstrides_type_type = inner_strides_type;

        using const_stepper
            = std::conditional_t<is_indexed_stepper<typename std::decay_t<CT>::stepper>::value,
                                 xindexed_stepper<const xstrided_view<CT, S, L, FST>, true>,
                                 xstepper<const xstrided_view<CT, S, L, FST>>>;

        using stepper
            = std::conditional_t<is_indexed_stepper<typename std::decay_t<CT>::stepper>::value,
                                 xindexed_stepper<xstrided_view<CT, S, L, FST>, false>,
                                 xstepper<xstrided_view<CT, S, L, FST>>>;
    };

    /*****************
     * xstrided_view *
     *****************/

    /**
     * @class xstrided_view
     * @brief View of an xexpression using strides
     *
     * The xstrided_view class implements a view utilizing an initial offset
     * and strides.
     *
     * @tparam CT the closure type of the \ref xexpression type underlying this view
     * @tparam L the layout of the strided view
     * @tparam S the strides type of the strided view
     * @tparam FST the flat storage type used for the strided view
     *
     * @sa strided_view, transpose
     */
    template <class CT,
              class S,
              layout_type L = layout_type::dynamic,
              class FST = detail::flat_storage_getter<CT, XTENSOR_DEFAULT_TRAVERSAL>>
    class xstrided_view
        : public xview_semantic<xstrided_view<CT, S, L, FST>>
        , public xiterable<xstrided_view<CT, S, L, FST>>
        , private xstrided_view_base<xstrided_view<CT, S, L, FST>>
        , public extension::xstrided_view_base_t<CT, S, L, FST>
    {
    public:

        using self_type = xstrided_view<CT, S, L, FST>;
        using base_type = xstrided_view_base<self_type>;
        using semantic_base = xview_semantic<self_type>;
        using extension_base = extension::xstrided_view_base_t<CT, S, L, FST>;
        using expression_tag = typename extension_base::expression_tag;

        using xexpression_type = typename base_type::xexpression_type;
        using base_type::is_const;

        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using size_type = typename base_type::size_type;
        using difference_type = typename base_type::difference_type;

        using inner_storage_type = typename base_type::inner_storage_type;
        using storage_type = typename base_type::storage_type;
        using storage_iterator = typename storage_type::iterator;
        using const_storage_iterator = typename storage_type::const_iterator;

        using iterable_base = xiterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using inner_strides_type = typename base_type::inner_strides_type;
        using inner_backstrides_type = typename base_type::inner_backstrides_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using base_type::contiguous_layout;
        using base_type::static_layout;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        using base_index_type = xindex_type_t<shape_type>;

        using simd_value_type = xt_simd::simd_type<value_type>;

        template <class CTA, class SA>
        xstrided_view(CTA&& e,
                      SA&& shape,
                      strides_type&& strides,
                      std::size_t offset,
                      layout_type layout) noexcept;

        xstrided_view(const xstrided_view& rhs) = default;
        xstrided_view& operator=(const xstrided_view& rhs);

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        using base_type::backstrides;
        using base_type::dimension;
        using base_type::layout;
        using base_type::shape;
        using base_type::size;
        using base_type::strides;

        using base_type::operator();
        using base_type::at;
        using base_type::unchecked;
        using base_type::operator[];
        using base_type::data;
        using base_type::data_offset;
        using base_type::element;
        using base_type::expression;
        using base_type::storage;

        using base_type::broadcast_shape;
        using base_type::has_linear_assign;

        template <class T>
        void fill(const T& value);

        storage_iterator storage_begin();
        storage_iterator storage_end();
        const_storage_iterator storage_cbegin() const;
        const_storage_iterator storage_cend() const;

        template <class ST>
        stepper stepper_begin(const ST& shape);
        template <class ST>
        stepper stepper_end(const ST& shape, layout_type l);

        template <class ST, class STEP = const_stepper>
        std::enable_if_t<!is_indexed_stepper<STEP>::value, STEP> stepper_begin(
            const ST& shape) const;
        template <class ST, class STEP = const_stepper>
        std::enable_if_t<!is_indexed_stepper<STEP>::value, STEP> stepper_end(const ST& shape,
                                                                             layout_type l) const;

        template <class ST, class STEP = const_stepper>
        std::enable_if_t<is_indexed_stepper<STEP>::value, STEP> stepper_begin(
            const ST& shape) const;
        template <class ST, class STEP = const_stepper>
        std::enable_if_t<is_indexed_stepper<STEP>::value, STEP> stepper_end(const ST& shape,
                                                                            layout_type l) const;

        using container_iterator = std::conditional_t<is_const,
                                                      typename storage_type::const_iterator,
                                                      typename storage_type::iterator>;
        using const_container_iterator = typename storage_type::const_iterator;

        template <class E>
        using rebind_t = xstrided_view<E, S, L, typename FST::template rebind_t<E>>;

        template <class E>
        rebind_t<E> build_view(E&& e) const;

    private:

        container_iterator data_xbegin() noexcept;
        const_container_iterator data_xbegin() const noexcept;
        container_iterator data_xend(layout_type l, size_type offset) noexcept;
        const_container_iterator data_xend(layout_type l, size_type offset) const noexcept;

        template <class It>
        It data_xbegin_impl(It begin) const noexcept;

        template <class It>
        It data_xend_impl(It end, layout_type l, size_type offset) const noexcept;

        void assign_temporary_impl(temporary_type&& tmp);

        template <class C>
        friend class xstepper;
        friend class xview_semantic<self_type>;
        friend class xaccessible<self_type>;
        friend class xconst_accessible<self_type>;
    };

    /**************************
     * xstrided_view builders *
     **************************/

    template <class T>
    using xstrided_slice = xtl::variant<
        T,

        xrange_adaptor<placeholders::xtuph, T, T>,
        xrange_adaptor<T, placeholders::xtuph, T>,
        xrange_adaptor<T, T, placeholders::xtuph>,

        xrange_adaptor<T, placeholders::xtuph, placeholders::xtuph>,
        xrange_adaptor<placeholders::xtuph, T, placeholders::xtuph>,
        xrange_adaptor<placeholders::xtuph, placeholders::xtuph, T>,

        xrange_adaptor<T, T, T>,
        xrange_adaptor<placeholders::xtuph, placeholders::xtuph, placeholders::xtuph>,

        xrange<T>,
        xstepped_range<T>,

        xall_tag,
        xellipsis_tag,
        xnewaxis_tag>;

    /**
     * @typedef xstrided_slice_vector
     * @brief vector of slices used to build a `xstrided_view`
     */
    using xstrided_slice_vector = std::vector<xstrided_slice<std::ptrdiff_t>>;

    template <layout_type L = layout_type::dynamic, class E, class S, class X>
    auto strided_view(
        E&& e, S&& shape, X&& stride, std::size_t offset = 0, layout_type layout = L) noexcept;

    template <class E>
    auto strided_view(E&& e, const xstrided_slice_vector& slices);

    /********************************
     * xstrided_view implementation *
     ********************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xstrided_view
     *
     * @param e the underlying xexpression for this view
     * @param shape the shape of the view
     * @param strides the strides of the view
     * @param offset the offset of the first element in the underlying container
     * @param layout the layout of the view
     */
    template <class CT, class S, layout_type L, class FST>
    template <class CTA, class SA>
    inline xstrided_view<CT, S, L, FST>::xstrided_view(CTA&& e,
                                                       SA&& shape,
                                                       strides_type&& strides,
                                                       std::size_t offset,
                                                       layout_type layout) noexcept
        : base_type(
              std::forward<CTA>(e), std::forward<SA>(shape), std::move(strides), offset, layout)
    {
    }
    //@}

    template <class CT, class S, layout_type L, class FST>
    inline xstrided_view<CT, S, L, FST>& xstrided_view<CT, S, L, FST>::operator=(
        const xstrided_view<CT, S, L, FST>& rhs)
    {
        temporary_type tmp(rhs);
        return this->assign_temporary(std::move(tmp));
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class CT, class S, layout_type L, class FST>
    template <class E>
    inline auto xstrided_view<CT, S, L, FST>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class CT, class S, layout_type L, class FST>
    template <class E>
    inline auto xstrided_view<CT, S, L, FST>::operator=(const E& e)
        -> disable_xexpression<E, self_type>&
    {
        this->fill(e);
        return *this;
    }

    namespace xstrided_view_detail
    {
        template <class V, class T>
        inline void run_assign_temporary_impl(V& v,
                                              const T& t,
                                              std::true_type /* enable strided assign */)
        {
            strided_loop_assigner<true>::run(v, t);
        }

        template <class V, class T>
        inline void run_assign_temporary_impl(V& v,
                                              const T& t,
                                              std::false_type /* fallback to iterator assign */)
        {
            std::copy(t.cbegin(), t.cend(), v.begin());
        }
    }

    template <class CT, class S, layout_type L, class FST>
    inline void xstrided_view<CT, S, L, FST>::assign_temporary_impl(temporary_type&& tmp)
    {
        constexpr bool fast_assign
            = xassign_traits<xstrided_view<CT, S, L, FST>, temporary_type>::simd_strided_loop();
        xstrided_view_detail::run_assign_temporary_impl(
            *this, tmp, std::integral_constant<bool, fast_assign>{});
    }

    /**
     * @name Data
     */
    //@{

    /**
     * Fills the view with the given value.
     * @param value the value to fill the view with.
     */
    template <class CT, class S, layout_type L, class FST>
    template <class T>
    inline void xstrided_view<CT, S, L, FST>::fill(const T& value)
    {
        if (layout() != layout_type::dynamic)
        {
            std::fill(this->storage_begin(), this->storage_end(), value);
        }
        else
        {
            std::fill(this->begin(), this->end(), value);
        }
    }
    //@}

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::storage_begin() -> storage_iterator
    {
        return this->storage().begin() + static_cast<std::ptrdiff_t>(data_offset());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::storage_end() -> storage_iterator
    {
        return this->storage().begin() + static_cast<std::ptrdiff_t>(data_offset() + size());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::storage_cbegin() const -> const_storage_iterator
    {
        return this->storage().cbegin() + static_cast<std::ptrdiff_t>(data_offset());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::storage_cend() const -> const_storage_iterator
    {
        return this->storage().cbegin() + static_cast<std::ptrdiff_t>(data_offset() + size());
    }

    /***************
     * stepper api *
     ***************/

    template <class CT, class S, layout_type L, class FST>
    template <class ST>
    inline auto xstrided_view<CT, S, L, FST>::stepper_begin(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, data_xbegin(), offset);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class ST>
    inline auto xstrided_view<CT, S, L, FST>::stepper_end(const ST& shape, layout_type l) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, data_xend(l, offset), offset);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class ST, class STEP>
    inline auto xstrided_view<CT, S, L, FST>::stepper_begin(const ST& shape) const
        -> std::enable_if_t<!is_indexed_stepper<STEP>::value, STEP>
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, data_xbegin(), offset);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class ST, class STEP>
    inline auto xstrided_view<CT, S, L, FST>::stepper_end(const ST& shape, layout_type l) const
        -> std::enable_if_t<!is_indexed_stepper<STEP>::value, STEP>
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, data_xend(l, offset), offset);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class ST, class STEP>
    inline auto xstrided_view<CT, S, L, FST>::stepper_begin(const ST& shape) const
        -> std::enable_if_t<is_indexed_stepper<STEP>::value, STEP>
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class ST, class STEP>
    inline auto xstrided_view<CT, S, L, FST>::stepper_end(const ST& shape, layout_type /*l*/) const
        -> std::enable_if_t<is_indexed_stepper<STEP>::value, STEP>
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset, true);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline It xstrided_view<CT, S, L, FST>::data_xbegin_impl(It begin) const noexcept
    {
        return begin + static_cast<std::ptrdiff_t>(this->data_offset());
    }

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline It xstrided_view<CT, S, L, FST>::data_xend_impl(It begin,
                                                           layout_type l,
                                                           size_type offset) const noexcept
    {
        return strided_data_end(*this, begin + std::ptrdiff_t(this->data_offset()), l, offset);
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::data_xbegin() noexcept -> container_iterator
    {
        return data_xbegin_impl(this->storage().begin());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::data_xbegin() const noexcept
        -> const_container_iterator
    {
        return data_xbegin_impl(this->storage().cbegin());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::data_xend(layout_type l, size_type offset) noexcept
        -> container_iterator
    {
        return data_xend_impl(this->storage().begin(), l, offset);
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::data_xend(layout_type l, size_type offset) const
        noexcept -> const_container_iterator
    {
        return data_xend_impl(this->storage().cbegin(), l, offset);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class E>
    inline auto xstrided_view<CT, S, L, FST>::build_view(E&& e) const -> rebind_t<E>
    {
        inner_shape_type sh(this->shape());
        inner_strides_type str(this->strides());
        return rebind_t<E>(std::forward<E>(e),
                           std::move(sh),
                           std::move(str),
                           base_type::data_offset(),
                           this->layout());
    }

    /*****************************************
     * xstrided_view builders implementation *
     *****************************************/

    /**
     * Construct a strided view from an xexpression, shape, strides and offset.
     *
     * @param e xexpression
     * @param shape the shape of the view
     * @param strides the new strides of the view
     * @param offset the offset of the first element in the underlying container
     * @param layout the new layout of the expression
     *
     * @tparam L the static layout type of the view (default: dynamic)
     * @tparam E type of xexpression
     * @tparam S strides type
     * @tparam X strides type
     *
     * @return the view
     */
    template <layout_type L, class E, class S, class X>
    inline auto strided_view(
        E&& e, S&& shape, X&& strides, std::size_t offset, layout_type layout) noexcept
    {
        using view_type = xstrided_view<xclosure_t<E>, S, L>;
        return view_type(
            std::forward<E>(e), std::forward<S>(shape), std::forward<X>(strides), offset, layout);
    }

    namespace detail
    {
        struct no_adj_strides_policy
        {
        protected:
            inline void resize(std::size_t)
            {
            }
            inline void set_fake_slice(std::size_t)
            {
            }

            template <class ST, class S>
            bool fill_args(const xstrided_slice_vector& /*slices*/,
                           std::size_t /*sl_idx*/,
                           std::size_t /*i*/,
                           std::size_t /*old_shape*/,
                           const ST& /*old_stride*/,
                           S& /*shape*/,
                           get_strides_t<S>& /*strides*/)
            {
                return false;
            }
        };
    }

    /**
     * Function to create a dynamic view from
     * an xexpression and an xstrided_slice_vector.
     *
     * @param e xexpression
     * @param slices the slice vector
     *
     * @return initialized strided_view according to slices
     *
     * \code{.cpp}
     * xt::xarray<double> a = {{1, 2, 3}, {4, 5, 6}};
     * xt::slice_vector sv({xt::range(0, 1)});
     * sv.push_back(xt::range(0, 3, 2));
     * auto v = xt::strided_view(a, sv);
     * // ==> {{1, 3}}
     * \endcode
     *
     * You can also achieve the same with the following short-hand syntax:
     *
     * \code{.cpp}
     * xt::xarray<double> a = {{1, 2, 3}, {4, 5, 6}};
     * auto v = xt::strided_view(a, {xt::range(0, 1), xt::range(0, 3, 2)});
     * // ==> {{1, 3}}
     * \endcode
     */
    template <class E>
    inline auto strided_view(E&& e, const xstrided_slice_vector& slices)
    {
        detail::strided_view_args<detail::no_adj_strides_policy> args;
        args.fill_args(e.shape(),
                       detail::get_strides<XTENSOR_DEFAULT_TRAVERSAL>(e),
                       detail::get_offset<XTENSOR_DEFAULT_TRAVERSAL>(e),
                       e.layout(),
                       slices);
        using view_type = xstrided_view<xclosure_t<E>, decltype(args.new_shape)>;
        return view_type(std::forward<E>(e),
                         std::move(args.new_shape),
                         std::move(args.new_strides),
                         args.new_offset,
                         args.new_layout);
    }

    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E, class S>
    inline auto reshape_view(E&& e, S&& shape)
    {
        using shape_type = std::decay_t<S>;
        get_strides_t<shape_type> strides;

        xt::resize_container(strides, shape.size());
        compute_strides(shape, L, strides);
        using view_type = xstrided_view<xclosure_t<E>,
                                        shape_type,
                                        layout_type::dynamic,
                                        detail::flat_adaptor_getter<xclosure_t<E>, L>>;
        return view_type(
            std::forward<E>(e), std::forward<S>(shape), std::move(strides), 0, e.layout());
    }

    /**
     * @deprecated
     * @brief Return a view on a container with a new shape
     *
     * Note: if you resize the underlying container, this view becomes
     * invalidated.
     *
     * @param e xexpression to reshape
     * @param shape new shape
     * @param order traversal order (optional)
     *
     * @return view on xexpression with new shape
     */
    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E, class S>
    inline auto reshape_view(E&& e, S&& shape, layout_type /*order*/)
    {
        return reshape_view<L>(std::forward<E>(e), std::forward<S>(shape));
    }

#if !defined(X_OLD_CLANG)
    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E, class I, std::size_t N>
    inline auto reshape_view(E&& e, const I (&shape)[N], layout_type order)
    {
        using shape_type = std::array<std::size_t, N>;
        return reshape_view<L>(
            std::forward<E>(e), xtl::forward_sequence<shape_type, decltype(shape)>(shape), order);
    }

    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E, class I, std::size_t N>
    inline auto reshape_view(E&& e, const I (&shape)[N])
    {
        using shape_type = std::array<std::size_t, N>;
        return reshape_view<L>(std::forward<E>(e),
                               xtl::forward_sequence<shape_type, decltype(shape)>(shape));
    }
#else
    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E, class I>
    inline auto reshape_view(E&& e, const std::initializer_list<I>& shape)
    {
        using shape_type = xt::dynamic_shape<std::size_t>;
        return reshape_view<L>(std::forward<E>(e),
                               xtl::forward_sequence<shape_type, decltype(shape)>(shape));
    }

    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E, class I>
    inline auto reshape_view(E&& e, const std::initializer_list<I>& shape, layout_type order)
    {
        using shape_type = xt::dynamic_shape<std::size_t>;
        return reshape_view<L>(
            std::forward<E>(e), xtl::forward_sequence<shape_type, decltype(shape)>(shape), order);
    }
#endif
}

#endif
