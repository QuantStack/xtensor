/****************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_FUNCTION_HPP
#define XTENSOR_FUNCTION_HPP

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

#include <xtl/xsequence.hpp>
#include <xtl/xtype_traits.hpp>

#include "xaccessible.hpp"
#include "xexpression_traits.hpp"
#include "xiterable.hpp"
#include "xlayout.hpp"
#include "xscalar.hpp"
#include "xshape.hpp"
#include "xstrides.hpp"
#include "xtensor_simd.hpp"
#include "xutils.hpp"

namespace xt
{
    template <class F, class... CT>
    struct has_simd_interface<xfunction<F, CT...>> : xfunction<F, CT...>::has_simd_interface
    {
    };

    namespace detail
    {
        template <bool... B>
        using conjunction_c = xtl::conjunction<std::integral_constant<bool, B>...>;

        /************************
         * xfunction_cache_impl *
         ************************/

        template <class S, class is_shape_trivial>
        struct xfunction_cache_impl
        {
            S shape;
            bool is_trivial;
            bool is_initialized;

            xfunction_cache_impl()
                : shape(xtl::make_sequence<S>(0, std::size_t(0)))
                , is_initialized(false)
            {
            }
        };

        template <std::size_t... N, class is_shape_trivial>
        struct xfunction_cache_impl<fixed_shape<N...>, is_shape_trivial>
        {
            XTENSOR_CONSTEXPR_ENHANCED_STATIC fixed_shape<N...> shape = fixed_shape<N...>();
            XTENSOR_CONSTEXPR_ENHANCED_STATIC bool is_trivial = is_shape_trivial::value;
            XTENSOR_CONSTEXPR_ENHANCED_STATIC bool is_initialized = true;
        };

#ifdef XTENSOR_HAS_CONSTEXPR_ENHANCED
        // Out of line definitions to prevent linker errors prior to C++17
        template <std::size_t... N, class is_shape_trivial>
        constexpr fixed_shape<N...>
            xfunction_cache_impl<fixed_shape<N...>, is_shape_trivial>::shape;

        template <std::size_t... N, class is_shape_trivial>
        constexpr bool xfunction_cache_impl<fixed_shape<N...>, is_shape_trivial>::is_trivial;

        template <std::size_t... N, class is_shape_trivial>
        constexpr bool xfunction_cache_impl<fixed_shape<N...>, is_shape_trivial>::is_initialized;
#endif

        /**********************
         * simd metafunctions *
         **********************/

        template <class F, class B, class = void>
        struct has_simd_apply : std::false_type
        {
        };

        template <class F, class B>
        struct has_simd_apply<F, B, void_t<decltype(&F::template simd_apply<B>)>> : std::true_type
        {
        };

        template <class V>
        struct has_simd_type
            : std::integral_constant<bool, !std::is_same<V, xt_simd::simd_type<V>>::value>
        {
        };

        // This meta struct checks wether SIMD should be activated for our
        // functor "F"
        template <class V, class F, class... CT>
        struct xsimd_meta_getter
        {
            using scalar_result_type = V;

            // check if all arguments are supported by SIMD
            using simd_arguments_exist
                = xtl::conjunction<has_simd_type<scalar_result_type>,
                                   has_simd_type<xvalue_type_t<std::decay_t<CT>>>...>;
            // if yes, insert correct type here
            using simd_value_type
                = xtl::mpl::eval_if_t<simd_arguments_exist,
                                      meta_identity<xt_simd::simd_type<scalar_result_type>>,
                                      make_invalid_type<>>;
            // if all types are supported, check that the functor has a working
            // simd_apply and all arguments have the simd interface
            using use_xsimd = xtl::conjunction<simd_arguments_exist,
                                               has_simd_apply<F, scalar_result_type>,
                                               has_simd_interface<std::decay_t<CT>>...>;
        };
    }

    /************************
     * xfunction extensions *
     ************************/

    namespace extension
    {
        template <class Tag, class F, class... CT>
        struct xfunction_base_impl;

        template <class F, class... CT>
        struct xfunction_base_impl<xtensor_expression_tag, F, CT...>
        {
            using type = xtensor_empty_base;
        };

        template <class F, class... CT>
        struct xfunction_base : xfunction_base_impl<xexpression_tag_t<CT...>, F, CT...>
        {
        };

        template <class F, class... CT>
        using xfunction_base_t = typename xfunction_base<F, CT...>::type;
    }

    template <class promote>
    struct xfunction_cache : detail::xfunction_cache_impl<typename promote::type, promote>
    {
    };

    template <class F, class... CT>
    class xfunction_iterator;

    template <class F, class... CT>
    class xfunction_stepper;

    template <class F, class... CT>
    class xfunction;

    template <class F, class... CT>
    struct xiterable_inner_types<xfunction<F, CT...>>
    {
        using inner_shape_type = promote_shape_t<typename std::decay_t<CT>::shape_type...>;
        using const_stepper = xfunction_stepper<F, CT...>;
        using stepper = const_stepper;
    };

    template <class F, class... CT>
    struct xcontainer_inner_types<xfunction<F, CT...>>
    {
        // Added indirection for MSVC 2017 bug with the operator value_type()
        using value_type = typename meta_identity<decltype(
            std::declval<F>()(std::declval<xvalue_type_t<std::decay_t<CT>>>()...))>::type;
        using reference = value_type;
        using const_reference = value_type;
        using size_type = common_size_type_t<std::decay_t<CT>...>;
    };

    /*************
     * xfunction *
     *************/

    /**
     * @class xfunction
     * @brief Multidimensional function operating on
     * xtensor expressions.
     *
     * The xfunction class implements a multidimensional function
     * operating on xtensor expressions.
     *
     * @tparam F the function type
     * @tparam CT the closure types for arguments of the function
     */
    template <class F, class... CT>
    class xfunction
        : private xconst_iterable<xfunction<F, CT...>>
        , public xexpression<xfunction<F, CT...>>
        , private xconst_accessible<xfunction<F, CT...>>
        , public extension::xfunction_base_t<F, CT...>
    {
    public:

        using self_type = xfunction<F, CT...>;
        using accessible_base = xconst_accessible<self_type>;
        using extension_base = extension::xfunction_base_t<F, CT...>;
        using expression_tag = typename extension_base::expression_tag;
        using only_scalar = all_xscalar<CT...>;
        using functor_type = typename std::remove_reference<F>::type;
        using tuple_type = std::tuple<CT...>;

        using inner_types = xcontainer_inner_types<self_type>;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = typename inner_types::size_type;
        using difference_type = common_difference_type_t<std::decay_t<CT>...>;
        using simd_meta_getter = detail::xsimd_meta_getter<value_type, F, CT...>;

        using has_simd_interface = typename simd_meta_getter::use_xsimd;
        using simd_value_type = typename simd_meta_getter::simd_value_type;
        using simd_argument_type = simd_value_type;

        template <class requested_type>
        using simd_return_type = xt_simd::simd_return_type<value_type, requested_type>;

        using iterable_base = xconst_iterable<xfunction<F, CT...>>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        static constexpr layout_type static_layout
            = compute_layout(std::decay_t<CT>::static_layout...);
        static constexpr bool contiguous_layout
            = detail::conjunction_c<std::decay_t<CT>::contiguous_layout...>::value;

        template <layout_type L>
        using layout_iterator = typename iterable_base::template layout_iterator<L>;
        template <layout_type L>
        using const_layout_iterator = typename iterable_base::template const_layout_iterator<L>;
        template <layout_type L>
        using reverse_layout_iterator = typename iterable_base::template reverse_layout_iterator<L>;
        template <layout_type L>
        using const_reverse_layout_iterator =
            typename iterable_base::template const_reverse_layout_iterator<L>;

        template <class S, layout_type L>
        using broadcast_iterator = typename iterable_base::template broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_broadcast_iterator =
            typename iterable_base::template const_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using reverse_broadcast_iterator =
            typename iterable_base::template reverse_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_reverse_broadcast_iterator =
            typename iterable_base::template const_reverse_broadcast_iterator<S, L>;

        using const_storage_iterator = xfunction_iterator<F, CT...>;
        using storage_iterator = const_storage_iterator;
        using const_reverse_storage_iterator = std::reverse_iterator<const_storage_iterator>;
        using reverse_storage_iterator = std::reverse_iterator<storage_iterator>;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;
        using reverse_iterator = typename iterable_base::reverse_iterator;
        using const_reverse_iterator = typename iterable_base::const_reverse_iterator;

        template <class Func,
                  class... CTA,
                  class U
                  = std::enable_if_t<!std::is_base_of<std::decay_t<Func>, self_type>::value>>
        xfunction(Func&& f, CTA&&... e) noexcept;

        ~xfunction() = default;

        xfunction(const xfunction&) = default;
        xfunction& operator=(const xfunction&) = default;

        xfunction(xfunction&&) = default;
        xfunction& operator=(xfunction&&) = default;

        using accessible_base::size;
        size_type dimension() const noexcept;
        const inner_shape_type& shape() const;
        layout_type layout() const noexcept;

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class... Args>
        const_reference unchecked(Args... args) const;

        using accessible_base::at;
        using accessible_base::operator[];
        using accessible_base::in_bounds;
        using accessible_base::periodic;

        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool broadcast_shape(S& shape, bool reuse_cache = false) const;

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept;

        using iterable_base::begin;
        using iterable_base::cbegin;
        using iterable_base::cend;
        using iterable_base::crbegin;
        using iterable_base::crend;
        using iterable_base::end;
        using iterable_base::rbegin;
        using iterable_base::rend;

        const_storage_iterator storage_begin() const noexcept;
        const_storage_iterator storage_end() const noexcept;
        const_storage_iterator storage_cbegin() const noexcept;
        const_storage_iterator storage_cend() const noexcept;

        const_reverse_storage_iterator storage_rbegin() const noexcept;
        const_reverse_storage_iterator storage_rend() const noexcept;
        const_reverse_storage_iterator storage_crbegin() const noexcept;
        const_reverse_storage_iterator storage_crend() const noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

        const_reference data_element(size_type i) const;

        template <class UT = self_type,
                  class = typename std::enable_if<UT::only_scalar::value>::type>
        operator value_type() const;

        template <class align,
                  class requested_type = value_type,
                  std::size_t N = xt_simd::simd_traits<requested_type>::size>
        simd_return_type<requested_type> load_simd(size_type i) const;

        const tuple_type& arguments() const noexcept;

    private:

        template <std::size_t... I>
        layout_type layout_impl(std::index_sequence<I...>) const noexcept;

        template <std::size_t... I, class... Args>
        const_reference access_impl(std::index_sequence<I...>, Args... args) const;

        template <std::size_t... I, class... Args>
        const_reference unchecked_impl(std::index_sequence<I...>, Args... args) const;

        template <std::size_t... I, class It>
        const_reference element_access_impl(std::index_sequence<I...>, It first, It last) const;

        template <std::size_t... I>
        const_reference data_element_impl(std::index_sequence<I...>, size_type i) const;

        template <class align, class requested_type, std::size_t N, std::size_t... I>
        auto load_simd_impl(std::index_sequence<I...>, size_type i) const;

        template <class Func, std::size_t... I>
        const_stepper build_stepper(Func&& f, std::index_sequence<I...>) const noexcept;

        template <class Func, std::size_t... I>
        auto build_iterator(Func&& f, std::index_sequence<I...>) const noexcept;

        size_type compute_dimension() const noexcept;

        void compute_cached_shape() const;

        tuple_type m_e;
        functor_type m_f;
        mutable xfunction_cache<detail::promote_index<typename std::decay_t<CT>::shape_type...>>
            m_cache;

        friend class xfunction_iterator<F, CT...>;
        friend class xfunction_stepper<F, CT...>;
        friend class xconst_iterable<self_type>;
        friend class xconst_accessible<self_type>;
    };

    /**********************
     * xfunction_iterator *
     **********************/

    template <class F, class... CT>
    class xfunction_iterator
        : public xtl::xrandom_access_iterator_base<xfunction_iterator<F, CT...>,
                                                   typename xfunction<F, CT...>::value_type,
                                                   typename xfunction<F, CT...>::difference_type,
                                                   typename xfunction<F, CT...>::pointer,
                                                   typename xfunction<F, CT...>::reference>
    {
    public:

        using self_type = xfunction_iterator<F, CT...>;
        using functor_type = typename std::remove_reference<F>::type;
        using xfunction_type = xfunction<F, CT...>;

        using value_type = typename xfunction_type::value_type;
        using reference = typename xfunction_type::value_type;
        using pointer = typename xfunction_type::const_pointer;
        using difference_type = typename xfunction_type::difference_type;
        using iterator_category = std::random_access_iterator_tag;

        template <class... It>
        xfunction_iterator(const xfunction_type* func, It&&... it) noexcept;

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(const self_type& rhs) const;

        reference operator*() const;

        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

    private:

        using data_type
            = std::tuple<decltype(linear_begin(std::declval<const std::decay_t<CT>>()))...>;

        template <std::size_t... I>
        reference deref_impl(std::index_sequence<I...>) const;

        template <std::size_t... I>
        difference_type tuple_max_diff(std::index_sequence<I...>,
                                       const data_type& lhs,
                                       const data_type& rhs) const;

        const xfunction_type* p_f;
        data_type m_it;
    };

    template <class F, class... CT>
    bool operator==(const xfunction_iterator<F, CT...>& it1,
                    const xfunction_iterator<F, CT...>& it2);

    template <class F, class... CT>
    bool operator<(const xfunction_iterator<F, CT...>& it1,
                   const xfunction_iterator<F, CT...>& it2);

    /*********************
     * xfunction_stepper *
     *********************/

    template <class F, class... CT>
    class xfunction_stepper
    {
    public:

        using self_type = xfunction_stepper<F, CT...>;
        using functor_type = typename std::remove_reference<F>::type;
        using xfunction_type = xfunction<F, CT...>;

        using value_type = typename xfunction_type::value_type;
        using reference = typename xfunction_type::value_type;
        using pointer = typename xfunction_type::const_pointer;
        using size_type = typename xfunction_type::size_type;
        using difference_type = typename xfunction_type::difference_type;

        using shape_type = typename xfunction_type::shape_type;

        template <class... St>
        xfunction_stepper(const xfunction_type* func, St&&... st) noexcept;

        void step(size_type dim);
        void step_back(size_type dim);
        void step(size_type dim, size_type n);
        void step_back(size_type dim, size_type n);
        void reset(size_type dim);
        void reset_back(size_type dim);

        void to_begin();
        void to_end(layout_type l);

        reference operator*() const;

        template <class ST>
        ST step_simd();

        void step_leading();

    private:

        template <std::size_t... I>
        reference deref_impl(std::index_sequence<I...>) const;

        template <class ST, std::size_t... I>
        ST step_simd_impl(std::index_sequence<I...>);

        const xfunction_type* p_f;
        std::tuple<typename std::decay_t<CT>::const_stepper...> m_st;
    };

    /*********************************
     * xfunction implementation *
     *********************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xfunction applying the specified function to the given
     * arguments.
     * @param f the function to apply
     * @param e the \ref xexpression arguments
     */
    template <class F, class... CT>
    template <class Func, class... CTA, class U>
    inline xfunction<F, CT...>::xfunction(Func&& f, CTA&&... e) noexcept
        : m_e(std::forward<CTA>(e)...)
        , m_f(std::forward<Func>(f))
    {
    }
    //@}

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of dimensions of the function.
     */
    template <class F, class... CT>
    inline auto xfunction<F, CT...>::dimension() const noexcept -> size_type
    {
        size_type dimension = m_cache.is_initialized ? m_cache.shape.size() : compute_dimension();
        return dimension;
    }

    template <class F, class... CT>
    inline void xfunction<F, CT...>::compute_cached_shape() const
    {
        static_assert(!detail::is_fixed<shape_type>::value,
                      "Calling compute_cached_shape on fixed!");

        m_cache.shape = uninitialized_shape<xindex_type_t<inner_shape_type>>(compute_dimension());
        m_cache.is_trivial = broadcast_shape(m_cache.shape, false);
        m_cache.is_initialized = true;
    }

    /**
     * Returns the shape of the xfunction.
     */
    template <class F, class... CT>
    inline auto xfunction<F, CT...>::shape() const -> const inner_shape_type&
    {
        xtl::mpl::static_if<!detail::is_fixed<inner_shape_type>::value>(
            [&](auto self) {
                if (!m_cache.is_initialized)
                {
                    self(this)->compute_cached_shape();
                }
            },
            [](auto /*self*/) {});
        return m_cache.shape;
    }

    /**
     * Returns the layout_type of the xfunction.
     */
    template <class F, class... CT>
    inline layout_type xfunction<F, CT...>::layout() const noexcept
    {
        return layout_impl(std::make_index_sequence<sizeof...(CT)>());
    }
    //@}

    /**
     * @name Data
     */
    /**
     * Returns a constant reference to the element at the specified position in the function.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the function.
     */
    template <class F, class... CT>
    template <class... Args>
    inline auto xfunction<F, CT...>::operator()(Args... args) const -> const_reference
    {
        // The static cast prevents the compiler from instantiating the template methods with signed
        // integers, leading to warning about signed/unsigned conversions in the deeper layers of
        // the access methods
        return access_impl(std::make_index_sequence<sizeof...(CT)>(),
                           static_cast<size_type>(args)...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param args a list of indices specifying the position in the expression. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the expression, else the behavior is undefined.
     *
     * @warning This method is meant for performance, for expressions with a dynamic
     * number of dimensions (i.e. not known at compile time). Since it may have
     * undefined behavior (see parameters), operator() should be preferred whenever
     * it is possible.
     * @warning This method is NOT compatible with broadcasting, meaning the following
     * code has undefined behavior:
     * \code{.cpp}
     * xt::xarray<double> a = {{0, 1}, {2, 3}};
     * xt::xarray<double> b = {0, 1};
     * auto fd = a + b;
     * double res = fd.unchecked(0, 1);
     * \endcode
     */
    template <class F, class... CT>
    template <class... Args>
    inline auto xfunction<F, CT...>::unchecked(Args... args) const -> const_reference
    {
        // The static cast prevents the compiler from instantiating the template methods with signed
        // integers, leading to warning about signed/unsigned conversions in the deeper layers of
        // the access methods
        return unchecked_impl(std::make_index_sequence<sizeof...(CT)>(),
                              static_cast<size_type>(args)...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the function.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the container.
     */
    template <class F, class... CT>
    template <class It>
    inline auto xfunction<F, CT...>::element(It first, It last) const -> const_reference
    {
        return element_access_impl(std::make_index_sequence<sizeof...(CT)>(), first, last);
    }
    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the function to the specified parameter.
     * @param shape the result shape
     * @param reuse_cache boolean for reusing a previously computed shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class... CT>
    template <class S>
    inline bool xfunction<F, CT...>::broadcast_shape(S& shape, bool reuse_cache) const
    {
        if (m_cache.is_initialized && reuse_cache)
        {
            std::copy(m_cache.shape.cbegin(), m_cache.shape.cend(), shape.begin());
            return m_cache.is_trivial;
        }
        else
        {
            // e.broadcast_shape must be evaluated even if b is false
            auto func = [&shape](bool b, auto&& e) { return e.broadcast_shape(shape) && b; };
            return accumulate(func, true, m_e);
        }
    }

    /**
     * Checks whether the xfunction can be linearly assigned to an expression
     * with the specified strides.
     * @return a boolean indicating whether a linear assign is possible
     */
    template <class F, class... CT>
    template <class S>
    inline bool xfunction<F, CT...>::has_linear_assign(const S& strides) const noexcept
    {
        auto func = [&strides](bool b, auto&& e) { return b && e.has_linear_assign(strides); };
        return accumulate(func, true, m_e);
    }
    //@}

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::storage_begin() const noexcept -> const_storage_iterator
    {
        return storage_cbegin();
    }

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::storage_end() const noexcept -> const_storage_iterator
    {
        return storage_cend();
    }

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::storage_cbegin() const noexcept -> const_storage_iterator
    {
        auto f = [](const auto& e) noexcept
        {
            return linear_begin(e);
        };
        return build_iterator(f, std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::storage_cend() const noexcept -> const_storage_iterator
    {
        auto f = [](const auto& e) noexcept
        {
            return linear_end(e);
        };
        return build_iterator(f, std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::storage_rbegin() const noexcept
        -> const_reverse_storage_iterator
    {
        return storage_crbegin();
    }

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::storage_rend() const noexcept -> const_reverse_storage_iterator
    {
        return storage_crend();
    }

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::storage_crbegin() const noexcept
        -> const_reverse_storage_iterator
    {
        return const_reverse_storage_iterator(storage_cend());
    }

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::storage_crend() const noexcept
        -> const_reverse_storage_iterator
    {
        return const_reverse_storage_iterator(storage_cbegin());
    }

    template <class F, class... CT>
    template <class S>
    inline auto xfunction<F, CT...>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        auto f = [&shape](const auto& e) noexcept
        {
            return e.stepper_begin(shape);
        };
        return build_stepper(f, std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class... CT>
    template <class S>
    inline auto xfunction<F, CT...>::stepper_end(const S& shape, layout_type l) const noexcept
        -> const_stepper
    {
        auto f = [&shape, l](const auto& e) noexcept
        {
            return e.stepper_end(shape, l);
        };
        return build_stepper(f, std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::data_element(size_type i) const -> const_reference
    {
        return data_element_impl(std::make_index_sequence<sizeof...(CT)>(), i);
    }

    template <class F, class... CT>
    template <class UT, class>
    inline xfunction<F, CT...>::operator value_type() const
    {
        return operator()();
    }

    template <class F, class... CT>
    template <class align, class requested_type, std::size_t N>
    inline auto xfunction<F, CT...>::load_simd(size_type i) const
        -> simd_return_type<requested_type>
    {
        return load_simd_impl<align, requested_type, N>(std::make_index_sequence<sizeof...(CT)>(),
                                                        i);
    }

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::arguments() const noexcept -> const tuple_type&
    {
        return m_e;
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline layout_type xfunction<F, CT...>::layout_impl(std::index_sequence<I...>) const noexcept
    {
        return compute_layout(std::get<I>(m_e).layout()...);
    }

    template <class F, class... CT>
    template <std::size_t... I, class... Args>
    inline auto xfunction<F, CT...>::access_impl(std::index_sequence<I...>, Args... args) const
        -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        return m_f(std::get<I>(m_e)(args...)...);
    }

    template <class F, class... CT>
    template <std::size_t... I, class... Args>
    inline auto xfunction<F, CT...>::unchecked_impl(std::index_sequence<I...>, Args... args) const
        -> const_reference
    {
        return m_f(std::get<I>(m_e).unchecked(args...)...);
    }

    template <class F, class... CT>
    template <std::size_t... I, class It>
    inline auto xfunction<F, CT...>::element_access_impl(std::index_sequence<I...>,
                                                         It first,
                                                         It last) const -> const_reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return m_f((std::get<I>(m_e).element(first, last))...);
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline auto xfunction<F, CT...>::data_element_impl(std::index_sequence<I...>, size_type i) const
        -> const_reference
    {
        return m_f((std::get<I>(m_e).data_element(i))...);
    }

    namespace detail
    {
        // This metafunction avoids loading boolean values as batches of floating points and
        // reciprocally. However, we cannot always load data as batches of their scalar type
        // since this prevents mixed arithmetic.
        template <class T, class simd, class common_simd>
        struct get_simd_type
        {
            using simd_value_type = typename std::decay_t<T>::simd_value_type;
            static constexpr bool is_arg_bool = ::xt_simd::is_batch_bool<simd_value_type>::value;
            static constexpr bool is_res_bool = ::xt_simd::is_batch_bool<simd>::value;
            static constexpr bool is_arg_cplx = ::xt_simd::is_batch_complex<simd_value_type>::value;
            using type = std::conditional_t<
                is_res_bool,
                common_simd,
                std::conditional_t<is_arg_bool || is_arg_cplx, simd_value_type, simd>>;
        };

        template <class T, class simd, class common_simd>
        using get_simd_type_t = typename get_simd_type<T, simd, common_simd>::type;
    }

    template <class F, class... CT>
    template <class align, class requested_type, std::size_t N, std::size_t... I>
    inline auto xfunction<F, CT...>::load_simd_impl(std::index_sequence<I...>, size_type i) const
    {
        return m_f.simd_apply((std::get<I>(m_e).template load_simd<align, requested_type>(i))...);
    }

    template <class F, class... CT>
    template <class Func, std::size_t... I>
    inline auto xfunction<F, CT...>::build_stepper(Func&& f, std::index_sequence<I...>) const
        noexcept -> const_stepper
    {
        return const_stepper(this, f(std::get<I>(m_e))...);
    }

    template <class F, class... CT>
    template <class Func, std::size_t... I>
    inline auto xfunction<F, CT...>::build_iterator(Func&& f, std::index_sequence<I...>) const
        noexcept
    {
        return const_storage_iterator(this, f(std::get<I>(m_e))...);
    }

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::compute_dimension() const noexcept -> size_type
    {
        auto func = [](size_type d, auto&& e) noexcept
        {
            return (std::max)(d, e.dimension());
        };
        return accumulate(func, size_type(0), m_e);
    }

    /*************************************
     * xfunction_iterator implementation *
     *************************************/

    template <class F, class... CT>
    template <class... It>
    inline xfunction_iterator<F, CT...>::xfunction_iterator(const xfunction_type* func,
                                                            It&&... it) noexcept
        : p_f(func)
        , m_it(std::forward<It>(it)...)
    {
    }

    template <class F, class... CT>
    inline auto xfunction_iterator<F, CT...>::operator++() -> self_type&
    {
        auto f = [](auto& it) { ++it; };
        for_each(f, m_it);
        return *this;
    }

    template <class F, class... CT>
    inline auto xfunction_iterator<F, CT...>::operator--() -> self_type&
    {
        auto f = [](auto& it) { return --it; };
        for_each(f, m_it);
        return *this;
    }

    template <class F, class... CT>
    inline auto xfunction_iterator<F, CT...>::operator+=(difference_type n) -> self_type&
    {
        auto f = [n](auto& it) { it += n; };
        for_each(f, m_it);
        return *this;
    }

    template <class F, class... CT>
    inline auto xfunction_iterator<F, CT...>::operator-=(difference_type n) -> self_type&
    {
        auto f = [n](auto& it) { it -= n; };
        for_each(f, m_it);
        return *this;
    }

    template <class F, class... CT>
    inline auto xfunction_iterator<F, CT...>::operator-(const self_type& rhs) const
        -> difference_type
    {
        return tuple_max_diff(std::make_index_sequence<sizeof...(CT)>(), m_it, rhs.m_it);
    }

    template <class F, class... CT>
    inline auto xfunction_iterator<F, CT...>::operator*() const -> reference
    {
        return deref_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class... CT>
    inline bool xfunction_iterator<F, CT...>::equal(const self_type& rhs) const
    {
        // Optimization: no need to compare each subiterator since they all
        // are incremented decremented together.
        constexpr std::size_t temp = xtl::mpl::find_if<is_not_xdummy_iterator, data_type>::value;
        constexpr std::size_t index = (temp == std::tuple_size<data_type>::value) ? 0 : temp;
        return std::get<index>(m_it) == std::get<index>(rhs.m_it);
    }

    template <class F, class... CT>
    inline bool xfunction_iterator<F, CT...>::less_than(const self_type& rhs) const
    {
        // Optimization: no need to compare each subiterator since they all
        // are incremented decremented together.
        constexpr std::size_t temp = xtl::mpl::find_if<is_not_xdummy_iterator, data_type>::value;
        constexpr std::size_t index = (temp == std::tuple_size<data_type>::value) ? 0 : temp;
        return std::get<index>(m_it) < std::get<index>(rhs.m_it);
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline auto xfunction_iterator<F, CT...>::deref_impl(std::index_sequence<I...>) const
        -> reference
    {
        return (p_f->m_f)(*std::get<I>(m_it)...);
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline auto xfunction_iterator<F, CT...>::tuple_max_diff(std::index_sequence<I...>,
                                                             const data_type& lhs,
                                                             const data_type& rhs) const
        -> difference_type
    {
        auto diff = std::make_tuple((std::get<I>(lhs) - std::get<I>(rhs))...);
        auto func = [](difference_type n, auto&& v) { return (std::max)(n, v); };
        return accumulate(func, difference_type(0), diff);
    }

    template <class F, class... CT>
    inline bool operator==(const xfunction_iterator<F, CT...>& it1,
                           const xfunction_iterator<F, CT...>& it2)
    {
        return it1.equal(it2);
    }

    template <class F, class... CT>
    inline bool operator<(const xfunction_iterator<F, CT...>& it1,
                          const xfunction_iterator<F, CT...>& it2)
    {
        return it1.less_than(it2);
    }

    /************************************
     * xfunction_stepper implementation *
     ************************************/

    template <class F, class... CT>
    template <class... St>
    inline xfunction_stepper<F, CT...>::xfunction_stepper(const xfunction_type* func,
                                                          St&&... st) noexcept
        : p_f(func)
        , m_st(std::forward<St>(st)...)
    {
    }

    template <class F, class... CT>
    inline void xfunction_stepper<F, CT...>::step(size_type dim)
    {
        auto f = [dim](auto& st) { st.step(dim); };
        for_each(f, m_st);
    }

    template <class F, class... CT>
    inline void xfunction_stepper<F, CT...>::step_back(size_type dim)
    {
        auto f = [dim](auto& st) { st.step_back(dim); };
        for_each(f, m_st);
    }

    template <class F, class... CT>
    inline void xfunction_stepper<F, CT...>::step(size_type dim, size_type n)
    {
        auto f = [dim, n](auto& st) { st.step(dim, n); };
        for_each(f, m_st);
    }

    template <class F, class... CT>
    inline void xfunction_stepper<F, CT...>::step_back(size_type dim, size_type n)
    {
        auto f = [dim, n](auto& st) { st.step_back(dim, n); };
        for_each(f, m_st);
    }

    template <class F, class... CT>
    inline void xfunction_stepper<F, CT...>::reset(size_type dim)
    {
        auto f = [dim](auto& st) { st.reset(dim); };
        for_each(f, m_st);
    }

    template <class F, class... CT>
    inline void xfunction_stepper<F, CT...>::reset_back(size_type dim)
    {
        auto f = [dim](auto& st) { st.reset_back(dim); };
        for_each(f, m_st);
    }

    template <class F, class... CT>
    inline void xfunction_stepper<F, CT...>::to_begin()
    {
        auto f = [](auto& st) { st.to_begin(); };
        for_each(f, m_st);
    }

    template <class F, class... CT>
    inline void xfunction_stepper<F, CT...>::to_end(layout_type l)
    {
        auto f = [l](auto& st) { st.to_end(l); };
        for_each(f, m_st);
    }

    template <class F, class... CT>
    inline auto xfunction_stepper<F, CT...>::operator*() const -> reference
    {
        return deref_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline auto xfunction_stepper<F, CT...>::deref_impl(std::index_sequence<I...>) const
        -> reference
    {
        return (p_f->m_f)(*std::get<I>(m_st)...);
    }

    template <class F, class... CT>
    template <class ST, std::size_t... I>
    inline ST xfunction_stepper<F, CT...>::step_simd_impl(std::index_sequence<I...>)
    {
        return (p_f->m_f.simd_apply)(
            std::get<I>(m_st)
                .template step_simd<detail::get_simd_type_t<
                    std::tuple_element_t<I, typename xfunction_type::tuple_type>,
                    ST,
                    typename xfunction_type::simd_argument_type>>()...);
    }

    template <class F, class... CT>
    template <class ST>
    inline ST xfunction_stepper<F, CT...>::step_simd()
    {
        return step_simd_impl<ST>(std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class... CT>
    inline void xfunction_stepper<F, CT...>::step_leading()
    {
        auto step_leading_lambda = [](auto&& st) { st.step_leading(); };
        for_each(step_leading_lambda, m_st);
    }
}

#endif
