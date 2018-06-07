/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
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

#include "xconcepts.hpp"
#include "xexpression.hpp"
#include "xiterator.hpp"
#include "xstrides.hpp"
#include "xtensor_forward.hpp"
#include "xutils.hpp"

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
    void strided_assign(E1& e1, const E2& e2, std::false_type);

    template <class E1, class E2>
    void strided_assign(E1& e1, const E2& e2, std::true_type);

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
        static void assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2);

        template <class E1, class E2>
        static void computed_assign(xexpression<E1>& e1, const xexpression<E2>& e2);

        template <class E1, class E2, class F>
        static void scalar_computed_assign(xexpression<E1>& e1, const E2& e2, F&& f);

        template <class E1, class E2>
        static void assert_compatible_shape(const xexpression<E1>& e1, const xexpression<E2>& e2);

    private:

        template <class E1, class E2>
        static bool resize(xexpression<E1>& e1, const xexpression<E2>& e2);
    };

    /*****************
     * data_assigner *
     *****************/

    template <class E1, class E2, layout_type L>
    class data_assigner
    {
    public:

        using lhs_iterator = typename E1::stepper;
        using rhs_iterator = typename E2::const_stepper;
        using shape_type = typename E1::shape_type;
        using index_type = xindex_type_t<shape_type>;
        using size_type = typename lhs_iterator::size_type;
        using difference_type = typename lhs_iterator::difference_type;

        data_assigner(E1& e1, const E2& e2);

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

    /********************
     * trivial_assigner *
     ********************/

    template <bool simd_assign>
    struct trivial_assigner
    {
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

    template <class E1, class E2, class = void>
    struct has_assign_to : std::false_type
    {
    };

    template <class E1, class E2>
    struct has_assign_to<E1, E2, void_t<decltype(std::declval<const E2&>().assign_to(std::declval<E1&>()))>>
        : std::true_type
    {
    };

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
        inline bool is_trivial_broadcast(const E1& e1, const E2& e2)
        {
            return (E1::contiguous_layout && E2::contiguous_layout && (E1::static_layout == E2::static_layout))
                    || e2.is_trivial_broadcast(e1.strides());
        }

        template <class D, class E2, class... SL>
        inline bool is_trivial_broadcast(const xview<D, SL...>&, const E2&)
        {
            return false;
        }

        template <class E, class = void_t<>>
        struct forbid_simd_assign
        {
            static constexpr bool value = true;
        };

        // Double steps check for xfunction because the default
        // parameter void_t of forbid_simd_assign prevents additional
        // specializations.
        template <class E>
        struct xfunction_forbid_simd;

        template <class E>
        struct forbid_simd_assign<E,
            void_t<decltype(std::declval<E>().template load_simd<aligned_mode>(typename E::size_type(0)))>>
        {
            static constexpr bool value = false || xfunction_forbid_simd<E>::value;
        };

        template <class E>
        struct xfunction_forbid_simd
        {
            static constexpr bool value = false;
        };

        template <class F, class R, class... CT>
        struct xfunction_forbid_simd<xfunction<F, R, CT...>>
        {
            static constexpr bool value = xtl::disjunction<
                std::integral_constant<bool, forbid_simd_assign<typename std::decay<CT>::type>::value>...>::value;
        };

        template <class F, class B, class = void>
        struct has_simd_apply : std::false_type {};

        template <class F, class B>
        struct has_simd_apply<F, B, void_t<decltype(&F::template simd_apply<B>)>>
            : std::true_type
        {
        };

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

        template <class F, class R, class... CT>
        struct use_strided_loop<xfunction<F, R, CT...>>
        {
            static constexpr bool value = xtl::conjunction<use_strided_loop<std::decay_t<CT>>...>::value &&
                                          has_simd_apply<F, xsimd::simd_type<R>>::value;
        };
    }

    template <class E1, class E2>
    struct xassign_traits
    {
        // constexpr methods instead of constexpr data members avoid the need of difinitions at namespace
        // scope of these data members (since they are odr-used).
        static constexpr bool contiguous_layout() { return E1::contiguous_layout && E2::contiguous_layout; }
        static constexpr bool same_type() { return std::is_same<typename E1::value_type, typename E2::value_type>::value; }
        static constexpr bool simd_size() { return xsimd::simd_traits<typename E1::value_type>::size > 1; }
        static constexpr bool forbid_simd() { return detail::forbid_simd_assign<E2>::value; }
        static constexpr bool simd_assign() { return contiguous_layout() && same_type() && simd_size() && !forbid_simd(); }
        static constexpr bool simd_strided_loop() { return same_type() && simd_size() && detail::use_strided_loop<E2>::value && detail::use_strided_loop<E1>::value; }
    };

    template <class E1, class E2>
    inline void xexpression_assigner_base<xtensor_expression_tag>::assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial)
    {
        E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();

        bool trivial_broadcast = trivial && detail::is_trivial_broadcast(de1, de2);
        if (trivial_broadcast)
        {
            constexpr bool simd_assign = xassign_traits<E1, E2>::simd_assign();
            trivial_assigner<simd_assign>::run(de1, de2);
        }
        else if (xassign_traits<E1, E2>::simd_strided_loop())
        {
            strided_assign(de1, de2, std::integral_constant<bool, xassign_traits<E1, E2>::simd_strided_loop()>{});
        }
        else
        {
            data_assigner<E1, E2, default_assignable_layout(E1::static_layout)> assigner(de1, de2);
            assigner.run();
        }
    }

    template <class Tag>
    template <class E1, class E2>
    inline void xexpression_assigner<Tag>::assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        bool trivial_broadcast = resize(e1, e2);
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
        shape_type shape = xtl::make_sequence<shape_type>(dim, size_type(0));
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
        std::transform(d.cbegin(), d.cend(), d.begin(),
                       [e2, &f](const auto& v) { return f(v, e2); });
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

    template <class Tag>
    template <class E1, class E2>
    inline bool xexpression_assigner<Tag>::resize(xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        using shape_type = typename E1::shape_type;
        using size_type = typename E1::size_type;
        const E2& de2 = e2.derived_cast();
        size_type size = de2.dimension();
        shape_type shape = xtl::make_sequence<shape_type>(size, size_type(0));
        bool trivial_broadcast = de2.broadcast_shape(shape, true);
        e1.derived_cast().resize(std::move(shape));
        return trivial_broadcast;
    }

    /********************************
     * data_assigner implementation *
     ********************************/

    template <class E1, class E2, layout_type L>
    inline data_assigner<E1, E2, L>::data_assigner(E1& e1, const E2& e2)
        : m_e1(e1), m_lhs(e1.stepper_begin(e1.shape())),
          m_rhs(e2.stepper_begin(e1.shape())),
          m_index(xtl::make_sequence<index_type>(e1.shape().size(), size_type(0)))
    {
    }

    template <class E1, class E2, layout_type L>
    inline void data_assigner<E1, E2, L>::run()
    {
        using size_type = typename E1::size_type;
        using argument_type = std::decay_t<decltype(*m_rhs)>;
        using result_type = std::decay_t<decltype(*m_lhs)>;
        constexpr bool is_narrowing = is_narrowing_conversion<argument_type, result_type>::value;

        size_type s = m_e1.size();
        for (size_type i = 0; i < s; ++i)
        {
            *m_lhs = conditional_cast<is_narrowing, result_type>(*m_rhs);
            stepper_tools<L>::increment_stepper(*this, m_index, m_e1.shape());
        }
    }

    template <class E1, class E2, layout_type L>
    inline void data_assigner<E1, E2, L>::step(size_type i)
    {
        m_lhs.step(i);
        m_rhs.step(i);
    }

    template <class E1, class E2, layout_type L>
    inline void data_assigner<E1, E2, L>::step(size_type i, size_type n)
    {
        m_lhs.step(i, n);
        m_rhs.step(i, n);
    }

    template <class E1, class E2, layout_type L>
    inline void data_assigner<E1, E2, L>::reset(size_type i)
    {
        m_lhs.reset(i);
        m_rhs.reset(i);
    }

    template <class E1, class E2, layout_type L>
    inline void data_assigner<E1, E2, L>::to_end(layout_type l)
    {
        m_lhs.to_end(l);
        m_rhs.to_end(l);
    }

    /***********************************
     * trivial_assigner implementation *
     ***********************************/

    template <bool simd_assign>
    template <class E1, class E2>
    inline void trivial_assigner<simd_assign>::run(E1& e1, const E2& e2)
    {
        using lhs_align_mode = xsimd::container_alignment_t<E1>;
        constexpr bool is_aligned = std::is_same<lhs_align_mode, aligned_mode>::value;
        using rhs_align_mode = std::conditional_t<is_aligned, inner_aligned_mode, unaligned_mode>;
        using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
        using simd_type = xsimd::simd_type<value_type>;
        using size_type = typename E1::size_type;
        size_type size = e1.size();
        size_type simd_size = simd_type::size;

        size_type align_begin = is_aligned ? 0 : xsimd::get_alignment_offset(e1.data(), size, simd_size);
        size_type align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));

        for (size_type i = 0; i < align_begin; ++i)
        {
            e1.data_element(i) = e2.data_element(i);
        }
        for (size_type i = align_begin; i < align_end; i += simd_size)
        {
            e1.template store_simd<lhs_align_mode, simd_type>(i, e2.template load_simd<rhs_align_mode, simd_type>(i));
        }
        for (size_type i = align_end; i < size; ++i)
        {
            e1.data_element(i) = e2.data_element(i);
        }
    }

    namespace assigner_detail
    {
        template <class E1, class E2>
        inline void trivial_assigner_run_impl(E1& e1, const E2& e2, std::true_type)
        {
            std::transform(e2.storage_cbegin(), e2.storage_cend(), e1.storage_begin(), [](typename E2::value_type x) { return static_cast<typename E1::value_type>(x); });
        }

        template <class E1, class E2>
        inline void trivial_assigner_run_impl(E1&, const E2&, std::false_type)
        {
            XTENSOR_PRECONDITION(false,
                "Internal error: trivial_assigner called with unrelated types.");
        }
    }

    template <>
    template <class E1, class E2>
    inline void trivial_assigner<false>::run(E1& e1, const E2& e2)
    {
        using is_convertible = std::is_convertible<typename std::decay_t<E1>::value_type,
                                                   typename std::decay_t<E2>::value_type>;
        // If the types are not compatible, this function is still instantiated but never called.
        // To avoid compilation problems in effectively unused code trivial_assigner_run_impl is
        // empty in this case.
        assigner_detail::trivial_assigner_run_impl(e1, e2, is_convertible());
    }

    /***********************
     * Strided assign loop *
     ***********************/

    namespace strided_assign_detail
    {
        template <class T>
        void next_idx(T& outer_index, T& outer_shape)
        {
            std::size_t i = outer_index.size();
            for (; i > 0; --i)
            {
                if (ptrdiff_t(outer_index[i - 1]) >= ptrdiff_t(outer_shape[i - 1]) - 1)
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

        template <class S1, class S2>
        std::size_t check_strides(S1& s1, S2& s2)
        {
            // Indices are faster than reverse iterators
            std::size_t s1_index = s1.size();
            std::size_t s2_index = s2.size();

            for (; s2_index != 0; --s1_index, --s2_index)
            {
                if (s1[s1_index - 1] != s2[s2_index - 1])
                {
                    break;
                }
            }
            return s1_index;
        }

        struct check_strides_functor
        {
            template <class T>
            void operator()(const T& el)
            {
                auto var = check_strides(max_strides, el.strides());
                if (var > cut)
                {
                    cut = var;
                }
            }

            template <class T>
            void operator()(const xt::xscalar<T>& /*el*/)
            {
            }

            template <class F, class R, class... CT>
            void operator()(const xt::xfunction<F, R, CT...>& xf)
            {
                xt::for_each(*this, xf.arguments());
            }

            std::size_t cut;
            xt::dynamic_shape<std::size_t> max_strides;
        };

        template <class E1, class E2>
        auto get_loop_sizes(const E1& e1, const E2& e2)
        {
            auto s_fct = check_strides_functor{};
            xt::resize_container(s_fct.max_strides, e1.strides().size());
            std::copy(e1.strides().begin(), e1.strides().end(), s_fct.max_strides.begin());
            s_fct(e2);

            std::size_t cut = s_fct.cut;
            std::size_t inner_loop_size = 1;
            std::size_t outer_loop_size = 1;
            std::size_t i = 0;

            for (; i < cut; ++i)
            {
                outer_loop_size *= e1.shape()[i];
            }
            for (; i < e1.dimension(); ++i)
            {
                inner_loop_size *= e1.shape()[i];
            }

            return std::make_tuple(inner_loop_size, outer_loop_size, cut);
        }
    }


    template <class E1, class E2>
    void strided_assign(E1& e1, const E2& e2, std::true_type)
    {
        std::size_t inner_loop_size, outer_loop_size, cut;
        std::tie(inner_loop_size, outer_loop_size, cut) = strided_assign_detail::get_loop_sizes(e1, e2);

        if (cut == e1.dimension())
        {
            // last dimensions is != 1, fall back to data_assigner
            data_assigner<E1, E2, default_assignable_layout(E1::static_layout)> assigner(e1, e2);
            assigner.run();
            return;
        }

        dynamic_shape<std::size_t> idx(cut);
        dynamic_shape<std::size_t> max(e1.shape().begin(), e1.shape().begin() + cut);

        using simd_type = xsimd::simd_type<typename E1::value_type>;

        std::size_t simd_size = inner_loop_size / simd_type::size;
        std::size_t simd_rest = inner_loop_size % simd_type::size;

        auto fct_stepper = e2.stepper_begin(e1.shape());
        auto res_stepper = e1.stepper_begin(e1.shape());
    
        for (std::size_t ox = 0; ox < outer_loop_size; ++ox)
        {
            for (std::size_t i = 0; i < simd_size; i++)
            {
                res_stepper.template store_simd<simd_type>(fct_stepper.template step_simd<simd_type>());
            }
            for (std::size_t i = 0; i < simd_rest; ++i)
            {
                *(res_stepper) = *(fct_stepper);
                res_stepper.step_leading();
                fct_stepper.step_leading();
            }
            strided_assign_detail::next_idx(idx, max);
            fct_stepper.to_begin();
            for (std::size_t i = 0; i < idx.size(); ++i)
            {
                fct_stepper.step(i, idx[i]);
            }
        }
    }

    template <class E1, class E2>
    void strided_assign(xexpression<E1>& e1, const xexpression<E2>& e2, std::false_type)
    {
    }
}

#endif
