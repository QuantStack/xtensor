/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ASSIGN_HPP
#define XTENSOR_ASSIGN_HPP

#include <algorithm>
#include <type_traits>
#include <utility>

#include <xtl/xsequence.hpp>

#include "xexpression.hpp"
#include "xiterator.hpp"
#include "xstrides.hpp"
#include "xtensor_forward.hpp"
#include "xutils.hpp"
#include "xfunction.hpp"
#include "xtensor_config.hpp"

#if defined(XTENSOR_USE_TBB)
#include <tbb/tbb.h>
#endif

namespace xt
{

    /********************
     * Assign functions *
     ********************/

    template <class E1, class E2>
    void assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial);

    template <class E1, class E2>
    void assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2);

    template <class E1, class E2>
    void computed_assign(xexpression<E1>& e1, const xexpression<E2>& e2);

    template <class E1, class E2, class F>
    void scalar_computed_assign(xexpression<E1>& e1, const E2& e2, F&& f);

    template <class E1, class E2>
    void assert_compatible_shape(const xexpression<E1>& e1, const xexpression<E2>& e2);

    template <class E1, class E2>
    void strided_assign(E1& e1, const E2& e2, std::false_type /*disable*/);

    template <class E1, class E2>
    void strided_assign(E1& e1, const E2& e2, std::true_type /*enable*/);

    /************************
     * xexpression_assigner *
     ************************/

    template <class Tag>
    class xexpression_assigner_base;

    template <>
    class xexpression_assigner_base<xtensor_expression_tag>
    {
    public:

        template <class E1, class E2>
        static void assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial);
    };

    template <class Tag>
    class xexpression_assigner : public xexpression_assigner_base<Tag>
    {
    public:

        using base_type = xexpression_assigner_base<Tag>;

        template <class E1, class E2>
        static void assign_xexpression(E1& e1, const E2& e2);

        template <class E1, class E2>
        static void computed_assign(xexpression<E1>& e1, const xexpression<E2>& e2);

        template <class E1, class E2, class F>
        static void scalar_computed_assign(xexpression<E1>& e1, const E2& e2, F&& f);

        template <class E1, class E2>
        static void assert_compatible_shape(const xexpression<E1>& e1, const xexpression<E2>& e2);

    private:

        template <class E1, class E2>
        static bool resize(E1& e1, const E2& e2);

        template <class E1, class F, class... CT>
        static bool resize(E1& e1, const xfunction<F, CT...>& e2);

    };

    /********************
     * stepper_assigner *
     ********************/

    template <class E1, class E2, layout_type L>
    class stepper_assigner
    {
    public:

        using lhs_iterator = typename E1::stepper;
        using rhs_iterator = typename E2::const_stepper;
        using shape_type = typename E1::shape_type;
        using index_type = xindex_type_t<shape_type>;
        using size_type = typename lhs_iterator::size_type;
        using difference_type = typename lhs_iterator::difference_type;

        stepper_assigner(E1& e1, const E2& e2);

        void run();

        void step(size_type i);
        void step(size_type i, size_type n);
        void reset(size_type i);

        void to_end(layout_type);

    private:

        E1& m_e1;

        lhs_iterator m_lhs;
        rhs_iterator m_rhs;

        index_type m_index;
    };

    /*******************
     * linear_assigner *
     *******************/

    template <bool simd_assign>
    class linear_assigner
    {
    public:

        template <class E1, class E2>
        static void run(E1& e1, const E2& e2);
    };

    template <>
    class linear_assigner<false>
    {
    public:

        template <class E1, class E2>
        static void run(E1& e1, const E2& e2);

    private:

        template <class E1, class E2>
        static void run_impl(E1& e1, const E2& e2, std::true_type);

        template <class E1, class E2>
        static void run_impl(E1& e1, const E2& e2, std::false_type);
    };

    /*************************
     * strided_loop_assigner *
     *************************/

    template <bool simd>
    class strided_loop_assigner
    {
    public:

        template <class E1, class E2>
        static void run(E1& e1, const E2& e2);
    };

    /***********************************
     * Assign functions implementation *
     ***********************************/

    template <class E1, class E2>
    inline void assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial)
    {
        using tag = xexpression_tag_t<E1, E2>;
        xexpression_assigner<tag>::assign_data(e1, e2, trivial);
    }

    template <class E1, class E2>
    inline void assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        xtl::mpl::static_if<has_assign_to<E1, E2>::value>([&](auto self)
        {
            self(e2).derived_cast().assign_to(e1);
        }, /*else*/ [&](auto /*self*/)
        {
            using tag = xexpression_tag_t<E1, E2>;
            xexpression_assigner<tag>::assign_xexpression(e1, e2);
        });
    }

    template <class E1, class E2>
    inline void computed_assign(xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        using tag = xexpression_tag_t<E1, E2>;
        xexpression_assigner<tag>::computed_assign(e1, e2);
    }

    template <class E1, class E2, class F>
    inline void scalar_computed_assign(xexpression<E1>& e1, const E2& e2, F&& f)
    {
        using tag = xexpression_tag_t<E1, E2>;
        xexpression_assigner<tag>::scalar_computed_assign(e1, e2, std::forward<F>(f));
    }

    template <class E1, class E2>
    inline void assert_compatible_shape(const xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        using tag = xexpression_tag_t<E1, E2>;
        xexpression_assigner<tag>::assert_compatible_shape(e1, e2);
    }

    /***************************************
     * xexpression_assigner implementation *
     ***************************************/

    namespace detail
    {
        template <class E1, class E2>
        constexpr bool linear_static_layout()
        {
            // A row_major or column_major container with a dimension <= 1 is computed as
            // layout any, leading to some performance improvements, for example when
            // assigning a col-major vector to a row-major vector etc
            return compute_layout(select_layout<E1::static_layout, typename E1::shape_type>::value,
                                  select_layout<E2::static_layout, typename E2::shape_type>::value) != layout_type::dynamic;
        }

        template <class E1, class E2>
        inline auto is_linear_assign(const E1& e1, const E2& e2) -> std::enable_if_t<has_strides<E1>::value, bool>
        {
            return (E1::contiguous_layout && E2::contiguous_layout && linear_static_layout<E1, E2>()) ||
                   (e1.layout() != layout_type::dynamic && e2.has_linear_assign(e1.strides()));
        }

        template <class E1, class E2>
        inline auto is_linear_assign(const E1&, const E2&) -> std::enable_if_t<!has_strides<E1>::value, bool>
        {
            return false;
        }

        template <class E1, class E2>
        inline bool linear_dynamic_layout(const E1& e1, const E2& e2)
        {
            return e1.is_contiguous()
                && e2.is_contiguous()
                && compute_layout(e1.layout(), e2.layout()) != layout_type::dynamic;
        }

        template <class E, class = void>
        struct has_step_leading : std::false_type
        {
        };

        template <class E>
        struct has_step_leading<E, void_t<decltype(std::declval<E>().step_leading())>>
            : std::true_type
        {
        };

        template <class T>
        struct use_strided_loop
        {
            static constexpr bool stepper_deref() { return std::is_reference<typename T::stepper::reference>::value; }
            static constexpr bool value = has_strides<T>::value && has_step_leading<typename T::stepper>::value && stepper_deref();
        };

        template <class T>
        struct use_strided_loop<xscalar<T>>
        {
            static constexpr bool value = true;
        };

        template <class F, class... CT>
        struct use_strided_loop<xfunction<F, CT...>>
        {
            static constexpr bool value = xtl::conjunction<use_strided_loop<std::decay_t<CT>>...>::value;
        };
    }

    template <class E1, class E2>
    class xassign_traits
    {
    private:

        using e1_value_type = typename E1::value_type;
        using e2_value_type = typename E2::value_type;

        template <class T>
        using is_bool = std::is_same<T, bool>;

        static constexpr bool contiguous_layout() { return E1::contiguous_layout && E2::contiguous_layout; }
        static constexpr bool convertible_types() { return std::is_convertible<e2_value_type, e1_value_type>::value; }

        static constexpr bool use_xsimd() { return xt_simd::simd_traits<int8_t>::size > 1; }

        template <class T>
        static constexpr bool simd_size_impl() { return xt_simd::simd_traits<T>::size > 1 || (is_bool<T>::value && use_xsimd()); }
        static constexpr bool simd_size() { return simd_size_impl<e1_value_type>() && simd_size_impl<e2_value_type>(); }
        static constexpr bool simd_interface() { return has_simd_interface<E2, requested_value_type>(); }

    public:
        
        // constexpr methods instead of constexpr data members avoid the need of definitions at namespace
        // scope of these data members (since they are odr-used).

        static constexpr bool simd_assign() { return convertible_types() && simd_size() && simd_interface(); }
        static constexpr bool linear_assign(const E1& e1, const E2& e2, bool trivial) { return trivial && detail::is_linear_assign(e1, e2); }
        static constexpr bool strided_assign() { return detail::use_strided_loop<E1>::value && detail::use_strided_loop<E2>::value; }
        static constexpr bool simd_linear_assign() { return contiguous_layout() && simd_assign(); }
        static constexpr bool simd_strided_assign() { return strided_assign() && simd_assign(); }

        static constexpr bool simd_linear_assign(const E1& e1, const E2& e2) { return simd_assign()
                                                                                && detail::linear_dynamic_layout(e1, e2); }

        using requested_value_type = std::conditional_t<is_bool<e2_value_type>::value,
                                                        typename E2::bool_load_type,
                                                        e2_value_type>;

    };

    template <class E1, class E2>
    inline void xexpression_assigner_base<xtensor_expression_tag>::assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial)
    {
        E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();
        using traits = xassign_traits<E1, E2>;

        bool linear_assign = traits::linear_assign(de1, de2, trivial);
        constexpr bool simd_assign = traits::simd_linear_assign();
        constexpr bool simd_linear_assign = traits::simd_linear_assign();
        constexpr bool simd_strided_assign = traits::simd_strided_assign();
        if (linear_assign)
        {
            if(simd_linear_assign || traits::simd_linear_assign(de1, de2))
            {
                // Do not use linear_assigner<true> here since it will make the compiler
                // instantiate this branch even if the runtime condition is false, resulting
                // in compilation error for expressions that do not provide a SIMD interface.
                // simd_assign is true if simd_linear_assign() or simd_linear_assign(de1, de2)
                // is true.
                linear_assigner<simd_assign>::run(de1, de2);
            }
            else
            {
                linear_assigner<false>::run(de1, de2);
            }
        }
        else if (simd_strided_assign)
        {
            strided_loop_assigner<simd_strided_assign>::run(de1, de2);
        }
        else
        {
            stepper_assigner<E1, E2, default_assignable_layout(E1::static_layout)> assigner(de1, de2);
            assigner.run();
        }
    }

    template <class Tag>
    template <class E1, class E2>
    inline void xexpression_assigner<Tag>::assign_xexpression(E1& e1, const E2& e2)
    {
        bool trivial_broadcast = resize(e1.derived_cast(), e2.derived_cast());
        base_type::assign_data(e1, e2, trivial_broadcast);
    }

    template <class Tag>
    template <class E1, class E2>
    inline void xexpression_assigner<Tag>::computed_assign(xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        using shape_type = typename E1::shape_type;
        using size_type = typename E1::size_type;

        E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();

        size_type dim = de2.dimension();
        shape_type shape = uninitialized_shape<shape_type>(dim);
        bool trivial_broadcast = de2.broadcast_shape(shape, true);

        if (dim > de1.dimension() || shape > de1.shape())
        {
            typename E1::temporary_type tmp(shape);
            base_type::assign_data(tmp, e2, trivial_broadcast);
            de1.assign_temporary(std::move(tmp));
        }
        else
        {
            base_type::assign_data(e1, e2, trivial_broadcast);
        }
    }

    template <class Tag>
    template <class E1, class E2, class F>
    inline void xexpression_assigner<Tag>::scalar_computed_assign(xexpression<E1>& e1, const E2& e2, F&& f)
    {
        E1& d = e1.derived_cast();
        using size_type = typename E1::size_type;
        auto dst = d.storage().begin();
        for (size_type i = d.size(); i > 0; --i)
        {
            *dst = f(*dst, e2);
            ++dst;
        }
    }

    template <class Tag>
    template <class E1, class E2>
    inline void xexpression_assigner<Tag>::assert_compatible_shape(const xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        const E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();
        if (!broadcastable(de2.shape(), de1.shape()))
        {
            throw_broadcast_error(de2.shape(), de1.shape());
        }
    }

    namespace detail
    {
        template <bool B, class... CT>
        struct static_trivial_broadcast;

        template <class... CT>
        struct static_trivial_broadcast<true, CT...>
        {
            static constexpr bool value = detail::promote_index<typename std::decay_t<CT>::shape_type...>::value;
        };

        template <class... CT>
        struct static_trivial_broadcast<false, CT...>
        {
            static constexpr bool value = false;
        };
    }

    template <class Tag>
    template <class E1, class E2>
    inline bool xexpression_assigner<Tag>::resize(E1& e1, const E2& e2)
    {
        // If our RHS is not a xfunction, we know that the RHS is at least potentially trivial
        // We check the strides of the RHS in detail::is_trivial_broadcast to see if they match up!
        // So we can skip a shape copy and a call to broadcast_shape(...)
        e1.resize(e2.shape());
        return true;
    }

    template <class Tag>
    template <class E1, class F, class... CT>
    inline bool xexpression_assigner<Tag>::resize(E1& e1, const xfunction<F, CT...>& e2)
    {
        return xtl::mpl::static_if<detail::is_fixed<typename xfunction<F, CT...>::shape_type>::value>(
            [&](auto /*self*/) {
                /*
                 * If the shape of the xfunction is statically known, we can compute the broadcast triviality
                 * at compile time plus we can resize right away.
                 */
                // resize in case LHS is not a fixed size container. If it is, this is a NOP
                e1.resize(typename xfunction<F, CT...>::shape_type{});
                return detail::static_trivial_broadcast<detail::is_fixed<typename xfunction<F, CT...>::shape_type>::value, CT...>::value;
            },
            /* else */ [&](auto /*self*/)
            {
                using index_type = xindex_type_t<typename E1::shape_type>;
                using size_type = typename E1::size_type;
                size_type size = e2.dimension();
                index_type shape = uninitialized_shape<index_type>(size);
                bool trivial_broadcast = e2.broadcast_shape(shape, true);
                e1.resize(std::move(shape));
                return trivial_broadcast;
            }
        );
    }

    /***********************************
     * stepper_assigner implementation *
     ***********************************/

    template <class FROM, class TO>
    struct is_narrowing_conversion
    {
        using argument_type = std::decay_t<FROM>;
        using result_type = std::decay_t<TO>;

        static const bool value = std::is_arithmetic<result_type>::value &&
            (sizeof(result_type) < sizeof(argument_type) ||
             (std::is_integral<result_type>::value && std::is_floating_point<argument_type>::value));
    };

    template <class FROM, class TO>
    struct has_sign_conversion
    {
        using argument_type = std::decay_t<FROM>;
        using result_type = std::decay_t<TO>;

        static const bool value = std::is_signed<argument_type>::value != std::is_signed<result_type>::value;
    };

    template <class FROM, class TO>
    struct has_assign_conversion
    {
        using argument_type = std::decay_t<FROM>;
        using result_type = std::decay_t<TO>;

        static const bool value = is_narrowing_conversion<argument_type, result_type>::value ||
                                  has_sign_conversion<argument_type, result_type>::value;
    };

    template <class E1, class E2, layout_type L>
    inline stepper_assigner<E1, E2, L>::stepper_assigner(E1& e1, const E2& e2)
        : m_e1(e1), m_lhs(e1.stepper_begin(e1.shape())),
          m_rhs(e2.stepper_begin(e1.shape())),
          m_index(xtl::make_sequence<index_type>(e1.shape().size(), size_type(0)))
    {
    }

    template <class E1, class E2, layout_type L>
    inline void stepper_assigner<E1, E2, L>::run()
    {
        using size_type = typename E1::size_type;
        using argument_type = std::decay_t<decltype(*m_rhs)>;
        using result_type = std::decay_t<decltype(*m_lhs)>;
        constexpr bool needs_cast = has_assign_conversion<argument_type, result_type>::value;

        size_type s = m_e1.size();
        for (size_type i = 0; i < s; ++i)
        {
            *m_lhs = conditional_cast<needs_cast, result_type>(*m_rhs);
            stepper_tools<L>::increment_stepper(*this, m_index, m_e1.shape());
        }
    }

    template <class E1, class E2, layout_type L>
    inline void stepper_assigner<E1, E2, L>::step(size_type i)
    {
        m_lhs.step(i);
        m_rhs.step(i);
    }

    template <class E1, class E2, layout_type L>
    inline void stepper_assigner<E1, E2, L>::step(size_type i, size_type n)
    {
        m_lhs.step(i, n);
        m_rhs.step(i, n);
    }

    template <class E1, class E2, layout_type L>
    inline void stepper_assigner<E1, E2, L>::reset(size_type i)
    {
        m_lhs.reset(i);
        m_rhs.reset(i);
    }

    template <class E1, class E2, layout_type L>
    inline void stepper_assigner<E1, E2, L>::to_end(layout_type l)
    {
        m_lhs.to_end(l);
        m_rhs.to_end(l);
    }

    /**********************************
     * linear_assigner implementation *
     **********************************/

    template <bool simd_assign>
    template <class E1, class E2>
    inline void linear_assigner<simd_assign>::run(E1& e1, const E2& e2)
    {
        using lhs_align_mode = xt_simd::container_alignment_t<E1>;
        constexpr bool is_aligned = std::is_same<lhs_align_mode, aligned_mode>::value;
        using rhs_align_mode = std::conditional_t<is_aligned, inner_aligned_mode, unaligned_mode>;
        using e1_value_type = typename E1::value_type;
        using e2_value_type = typename E2::value_type;
        using value_type = typename xassign_traits<E1, E2>::requested_value_type;
        using simd_type = xt_simd::simd_type<value_type>;
        using size_type = typename E1::size_type;
        size_type size = e1.size();
        constexpr size_type simd_size = simd_type::size;
        constexpr bool needs_cast = has_assign_conversion<e1_value_type, e2_value_type>::value;

        size_type align_begin = is_aligned ? 0 : xt_simd::get_alignment_offset(e1.data(), size, simd_size);
        size_type align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));

        for (size_type i = 0; i < align_begin; ++i)
        {
            e1.data_element(i) = conditional_cast<needs_cast, e1_value_type>(e2.data_element(i));
        }

#if defined(XTENSOR_USE_TBB)
        tbb::parallel_for(align_begin, align_end, simd_size, [&](size_t i)
          {
            e1.template store_simd<lhs_align_mode>(i, e2.template load_simd<rhs_align_mode, value_type>(i));
          });
#elif defined(XTENSOR_USE_OPENMP)
        #pragma omp parallel for default(none) shared(align_begin, align_end, e1, e2)
        for (size_type i = align_begin; i < align_end; i += simd_size)
        {
          e1.template store_simd<lhs_align_mode>(i, e2.template load_simd<rhs_align_mode, value_type>(i));
        }
#else
        for (size_type i = align_begin; i < align_end; i += simd_size)
        {
          e1.template store_simd<lhs_align_mode>(i, e2.template load_simd<rhs_align_mode, value_type>(i));
        }
#endif
        for (size_type i = align_end; i < size; ++i)
        {
            e1.data_element(i) = conditional_cast<needs_cast, e1_value_type>(e2.data_element(i));
        }
    }

    template <class E1, class E2>
    inline void linear_assigner<false>::run(E1& e1, const E2& e2)
    {
        using is_convertible = std::is_convertible<typename std::decay_t<E2>::value_type,
                                                   typename std::decay_t<E1>::value_type>;
        // If the types are not compatible, this function is still instantiated but never called.
        // To avoid compilation problems in effectively unused code trivial_assigner_run_impl is
        // empty in this case.
        run_impl(e1, e2, is_convertible());
    }

    template <class E1, class E2>
    inline void linear_assigner<false>::run_impl(E1& e1, const E2& e2, std::true_type /*is_convertible*/)
    {
        using value_type = typename E1::value_type;
        using size_type = typename E1::size_type;
        auto src = linear_begin(e2);
        auto dst = linear_begin(e1);
        size_type n = e1.size();

#if defined(XTENSOR_USE_TBB)
        tbb::parallel_for(std::ptrdiff_t(0), static_cast<std::ptrdiff_t>(n), [&](std::ptrdiff_t i)
        {
            *(dst + i) = static_cast<value_type>(*(src + i));
        });
#elif defined(XTENSOR_USE_OPENMP)
        #pragma omp parallel for default(none) shared(src, dst, n)
        for (std::ptrdiff_t i = std::ptrdiff_t(0); i < static_cast<std::ptrdiff_t>(n) ; i++)
        {
            *(dst + i) = static_cast<value_type>(*(src + i));
        }
#else
        for (; n > size_type(0); --n)
        {
            *dst = static_cast<value_type>(*src);
            ++src;
            ++dst;
        }
#endif
    }

    template <class E1, class E2>
    inline void linear_assigner<false>::run_impl(E1&, const E2&, std::false_type /*is_convertible*/)
    {
        XTENSOR_PRECONDITION(false,
            "Internal error: linear_assigner called with unrelated types.");

    }

    /****************************************
     * strided_loop_assigner implementation *
     ****************************************/

    namespace strided_assign_detail
    {
        template <layout_type layout>
        struct idx_tools;

        template <>
        struct idx_tools<layout_type::row_major>
        {
            template <class T>
            static void next_idx(T& outer_index, T& outer_shape)
            {
                auto i = outer_index.size();
                for (; i > 0; --i)
                {
                    if (outer_index[i - 1] + 1 >= outer_shape[i - 1])
                    {
                        outer_index[i - 1] = 0;
                    }
                    else
                    {
                        outer_index[i - 1]++;
                        break;
                    }
                }
            }
        };

        template <>
        struct idx_tools<layout_type::column_major>
        {
            template <class T>
            static void next_idx(T& outer_index, T& outer_shape)
            {
                using size_type = typename T::size_type;
                size_type i = 0;
                auto sz = outer_index.size();
                for (; i < sz; ++i)
                {
                    if (outer_index[i] + 1 >= outer_shape[i])
                    {
                        outer_index[i] = 0;
                    }
                    else
                    {
                        outer_index[i]++;
                        break;
                    }
                }
            }
        };

        template <layout_type L, class S>
        struct check_strides_functor
        {
            using strides_type = S;

            check_strides_functor(const S& strides)
                : m_cut(L == layout_type::row_major ? 0 : strides.size()),
                  m_strides(strides)
            {
            }

            template <class T, layout_type LE = L>
            std::enable_if_t<LE == layout_type::row_major, std::size_t>
            operator()(const T& el)
            {
                auto var = check_strides_overlap<layout_type::row_major>::get(m_strides, el.strides());
                if (var > m_cut)
                {
                    m_cut = var;
                }
                return m_cut;
            }

            template <class T, layout_type LE = L>
            std::enable_if_t<LE == layout_type::column_major, std::size_t>
            operator()(const T& el)
            {
                auto var = check_strides_overlap<layout_type::column_major>::get(m_strides, el.strides());
                if (var < m_cut)
                {
                    m_cut = var;
                }
                return m_cut;
            }

            template <class T>
            std::size_t operator()(const xt::xscalar<T>& /*el*/)
            {
                return m_cut;
            }

            template <class F, class... CT>
            std::size_t operator()(const xt::xfunction<F, CT...>& xf)
            {
                xt::for_each(*this, xf.arguments());
                return m_cut;
            }

        private:

            std::size_t m_cut;
            const strides_type& m_strides;
        };

        template <class E1, class E2>
        auto get_loop_sizes(const E1& e1, const E2& e2, bool is_row_major)
        {
            std::size_t cut = 0;

            // TODO! if E1 is !contiguous --> initialize cut to sensible value!
            if (E1::static_layout == layout_type::row_major || is_row_major)
            {
                auto csf = check_strides_functor<layout_type::row_major, decltype(e1.strides())>(e1.strides());
                cut = csf(e2);
            }
            else if (E1::static_layout == layout_type::column_major || !is_row_major)
            {
                auto csf = check_strides_functor<layout_type::column_major, decltype(e1.strides())>(e1.strides());
                cut = csf(e2);
            } // can't reach here because this would have already triggered the fallback

            using shape_value_type = typename E1::shape_type::value_type;
            std::size_t outer_loop_size = static_cast<std::size_t>(
                            std::accumulate(e1.shape().begin(), e1.shape().begin() + static_cast<std::ptrdiff_t>(cut),
                                shape_value_type(1), std::multiplies<shape_value_type>{}));
            std::size_t inner_loop_size = static_cast<std::size_t>(
                            std::accumulate(e1.shape().begin() + static_cast<std::ptrdiff_t>(cut), e1.shape().end(),
                                shape_value_type(1), std::multiplies<shape_value_type>{}));

            if (E1::static_layout == layout_type::column_major || !is_row_major)
            {
                std::swap(outer_loop_size, inner_loop_size);
            }

            return std::make_tuple(inner_loop_size, outer_loop_size, cut);
        }
    }

    template <bool simd>
    template <class E1, class E2>
    inline void strided_loop_assigner<simd>::run(E1& e1, const E2& e2)
    {
        bool is_row_major = true;
        using fallback_assigner = stepper_assigner<E1, E2, default_assignable_layout(E1::static_layout)>;

        if (E1::static_layout == layout_type::dynamic)
        {
            layout_type dynamic_layout = e1.layout();
            switch (dynamic_layout)
            {
                case layout_type::row_major:
                    is_row_major = true;
                    break;
                case layout_type::column_major:
                    is_row_major = false;
                    break;
                default:
                    return fallback_assigner(e1, e2).run();
            }
        }
        else if (E1::static_layout == layout_type::row_major)
        {
            is_row_major = true;
        }
        else if (E1::static_layout == layout_type::column_major)
        {
            is_row_major = false;
        }
        else
        {
            throw std::runtime_error("Illegal layout set (layout_type::any?).");
        }

        std::size_t inner_loop_size, outer_loop_size, cut;
        std::tie(inner_loop_size, outer_loop_size, cut) = strided_assign_detail::get_loop_sizes(e1, e2, is_row_major);

        if ((is_row_major && cut == e1.dimension()) || (!is_row_major && cut == 0)) 
        {
            return fallback_assigner(e1, e2).run();
        }

        // TODO can we get rid of this and use `shape_type`?
        dynamic_shape<std::size_t> idx, max_shape;

        if (is_row_major)
        {
            xt::resize_container(idx, cut);
            max_shape.assign(e1.shape().begin(), e1.shape().begin() + static_cast<std::ptrdiff_t>(cut));
        }
        else
        {
            xt::resize_container(idx, e1.shape().size() - cut);
            max_shape.assign(e1.shape().begin() + static_cast<std::ptrdiff_t>(cut), e1.shape().end());
        }

        // add this when we have std::array index!
        // std::fill(idx.begin(), idx.end(), 0);
        using e1_value_type = typename E1::value_type;
        using e2_value_type = typename E2::value_type;
        constexpr bool needs_cast = has_assign_conversion<e1_value_type, e2_value_type>::value;
        using value_type = typename xassign_traits<E1, E2>::requested_value_type;
        using simd_type = std::conditional_t<std::is_same<e1_value_type, bool>::value,
                                             xt_simd::simd_bool_type<value_type>,
                                             xt_simd::simd_type<value_type>>;

        std::size_t simd_size = inner_loop_size / simd_type::size;
        std::size_t simd_rest = inner_loop_size % simd_type::size;

        auto fct_stepper = e2.stepper_begin(e1.shape());
        auto res_stepper = e1.stepper_begin(e1.shape());

        // TODO in 1D case this is ambigous -- could be RM or CM.
        //      Use default layout to make decision
        std::size_t step_dim = 0;
        if (!is_row_major) // row major case
        {
            step_dim = cut;
        }

        for (std::size_t ox = 0; ox < outer_loop_size; ++ox)
        {
            for (std::size_t i = 0; i < simd_size; ++i)
            {
                res_stepper.template store_simd<simd_type>(fct_stepper.template step_simd<value_type>());
            }
            for (std::size_t i = 0; i < simd_rest; ++i)
            {
                *(res_stepper) = conditional_cast<needs_cast, e1_value_type>(*(fct_stepper));
                res_stepper.step_leading();
                fct_stepper.step_leading();
            }

            is_row_major ?
                strided_assign_detail::idx_tools<layout_type::row_major>::next_idx(idx, max_shape) :
                strided_assign_detail::idx_tools<layout_type::column_major>::next_idx(idx, max_shape);

            fct_stepper.to_begin();

            // need to step E1 as well if not contigous assign (e.g. view)
            if (!E1::contiguous_layout)
            {
                res_stepper.to_begin();
                for (std::size_t i = 0; i < idx.size(); ++i)
                {
                    fct_stepper.step(i + step_dim, idx[i]);
                    res_stepper.step(i + step_dim, idx[i]);
                }
            }
            else
            {
                for (std::size_t i = 0; i < idx.size(); ++i)
                {
                    fct_stepper.step(i + step_dim, idx[i]);
                }
            }
        }
    }

    template <>
    template <class E1, class E2>
    inline void strided_loop_assigner<false>::run(E1& /*e1*/, const E2& /*e2*/)
    {
    }
}

#endif
