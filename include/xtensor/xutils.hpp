/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_UTILS_HPP
#define XTENSOR_UTILS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <xtl/xfunctional.hpp>
#include <xtl/xsequence.hpp>
#include <xtl/xmeta_utils.hpp>
#include <xtl/xtype_traits.hpp>

#include "xtensor_config.hpp"

#if (_MSC_VER >= 1910)
    #define NOEXCEPT(T)
#else
    #define NOEXCEPT(T) noexcept(T)
#endif

namespace xt
{
    /****************
     * declarations *
     ****************/

    template <class T>
    struct remove_class;

    /*template <class F, class... T>
    void for_each(F&& f, std::tuple<T...>& t) noexcept(implementation_dependent);*/

    /*template <class F, class R, class... T>
    R accumulate(F&& f, R init, const std::tuple<T...>& t) noexcept(implementation_dependent);*/

    template <std::size_t I, class... Args>
    constexpr decltype(auto) argument(Args&&... args) noexcept;

    template <class R, class F, class... S>
    R apply(std::size_t index, F&& func, const std::tuple<S...>& s) NOEXCEPT(noexcept(func(std::get<0>(s))));

    template <class T, class S>
    void nested_copy(T&& iter, const S& s);

    template <class T, class S>
    void nested_copy(T&& iter, std::initializer_list<S> s);

    template <class C>
    bool resize_container(C& c, typename C::size_type size);

    template <class T, std::size_t N>
    bool resize_container(std::array<T, N>& a, typename std::array<T, N>::size_type size);

    template <std::size_t... I>
    class fixed_shape;

    template <std::size_t... I>
    bool resize_container(fixed_shape<I...>& a, std::size_t size);

    template <class X, class C>
    struct rebind_container;

    template <class X, class C>
    using rebind_container_t = typename rebind_container<X, C>::type;

    std::size_t normalize_axis(std::size_t dim, std::ptrdiff_t axis);

    // gcc 4.9 is affected by C++14 defect CGW 1558
    // see http://open-std.org/JTC1/SC22/WG21/docs/cwg_defects.html#1558
    template <class... T>
    struct make_void
    {
        using type = void;
    };

    template <class... T>
    using void_t = typename make_void<T...>::type;

    // This is used for non existent types (e.g. storage for some expressions
    // like generators)
    struct invalid_type
    {
    };

    template <class... T>
    struct make_invalid_type
    {
        using type = invalid_type;
    };

    template <class T, class R>
    using disable_integral_t = std::enable_if_t<!std::is_integral<T>::value, R>;

    /********************************
     * meta identity implementation *
     ********************************/

    template <class T>
    struct meta_identity
    {
        using type = T;
    };

    /*******************************
     * remove_class implementation *
     *******************************/

    template <class T>
    struct remove_class
    {
    };

    template <class C, class R, class... Args>
    struct remove_class<R (C::*)(Args...)>
    {
        typedef R type(Args...);
    };

    template <class C, class R, class... Args>
    struct remove_class<R (C::*)(Args...) const>
    {
        typedef R type(Args...);
    };

    template <class T>
    using remove_class_t = typename remove_class<T>::type;

    /***************************
     * for_each implementation *
     ***************************/

    namespace detail
    {
        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I == sizeof...(T), void>::type
        for_each_impl(F&& /*f*/, std::tuple<T...>& /*t*/) noexcept
        {
        }

        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I < sizeof...(T), void>::type
        for_each_impl(F&& f, std::tuple<T...>& t)
            noexcept(noexcept(f(std::get<I>(t))))
        {
            f(std::get<I>(t));
            for_each_impl<I + 1, F, T...>(std::forward<F>(f), t);
        }
    }

    template <class F, class... T>
    inline void for_each(F&& f, std::tuple<T...>& t)
        noexcept(noexcept(detail::for_each_impl<0, F, T...>(std::forward<F>(f), t)))
    {
        detail::for_each_impl<0, F, T...>(std::forward<F>(f), t);
    }

    namespace detail
    {
        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I == sizeof...(T), void>::type
        for_each_impl(F&& /*f*/, const std::tuple<T...>& /*t*/) noexcept
        {
        }

        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I < sizeof...(T), void>::type
        for_each_impl(F&& f, const std::tuple<T...>& t)
            noexcept(noexcept(f(std::get<I>(t))))
        {
            f(std::get<I>(t));
            for_each_impl<I + 1, F, T...>(std::forward<F>(f), t);
        }
    }

    template <class F, class... T>
    inline void for_each(F&& f, const std::tuple<T...>& t)
        noexcept(noexcept(detail::for_each_impl<0, F, T...>(std::forward<F>(f), t)))
    {
        detail::for_each_impl<0, F, T...>(std::forward<F>(f), t);
    }

    /*****************************
     * accumulate implementation *
     *****************************/

    /// @cond DOXYGEN_INCLUDE_NOEXCEPT

    namespace detail
    {
        template <std::size_t I, class F, class R, class... T>
        inline std::enable_if_t<I == sizeof...(T), R>
        accumulate_impl(F&& /*f*/, R init, const std::tuple<T...>& /*t*/) noexcept
        {
            return init;
        }

        template <std::size_t I, class F, class R, class... T>
        inline std::enable_if_t<I < sizeof...(T), R>
        accumulate_impl(F&& f, R init, const std::tuple<T...>& t)
            noexcept(noexcept(f(init, std::get<I>(t))))
        {
            R res = f(init, std::get<I>(t));
            return accumulate_impl<I + 1, F, R, T...>(std::forward<F>(f), res, t);
        }
    }

    template <class F, class R, class... T>
    inline R accumulate(F&& f, R init, const std::tuple<T...>& t)
        noexcept(noexcept(detail::accumulate_impl<0, F, R, T...>(std::forward<F>(f), init, t)))
    {
        return detail::accumulate_impl<0, F, R, T...>(std::forward<F>(f), init, t);
    }

    /// @endcond

    /***************************
     * argument implementation *
     ***************************/

    namespace detail
    {
        template <std::size_t I>
        struct getter
        {
            template <class Arg, class... Args>
            static constexpr decltype(auto) get(Arg&& /*arg*/, Args&&... args) noexcept
            {
                return getter<I - 1>::get(std::forward<Args>(args)...);
            }
        };

        template <>
        struct getter<0>
        {
            template <class Arg, class... Args>
            static constexpr Arg&& get(Arg&& arg, Args&&... /*args*/) noexcept
            {
                return std::forward<Arg>(arg);
            }
        };
    }

    template <std::size_t I, class... Args>
    constexpr decltype(auto) argument(Args&&... args) noexcept
    {
        static_assert(I < sizeof...(Args), "I should be lesser than sizeof...(Args)");
        return detail::getter<I>::get(std::forward<Args>(args)...);
    }

    /************************
     * apply implementation *
     ************************/

    namespace detail
    {
        template <class R, class F, std::size_t I, class... S>
        R apply_one(F&& func, const std::tuple<S...>& s) NOEXCEPT(noexcept(func(std::get<I>(s))))
        {
            return static_cast<R>(func(std::get<I>(s)));
        }

        template <class R, class F, std::size_t I, class S>
        R apply_one(F&& func, const std::vector<S>& s) NOEXCEPT(noexcept(func(s[I])))
        {
            return static_cast<R>(func(s[I]));
        }

        template <class R, class F, std::size_t... I, class... S>
        R apply(std::size_t index, F&& func, std::index_sequence<I...> /*seq*/, const std::tuple<S...>& s)
            NOEXCEPT(noexcept(func(std::get<0>(s))))
        {
            using FT = std::add_pointer_t<R(F&&, const std::tuple<S...>&)>;
            static const std::array<FT, sizeof...(I)> ar = {{&apply_one<R, F, I, S...>...}};
            return ar[index](std::forward<F>(func), s);
        }

        template <class R, class F, std::size_t... I, class S>
        R apply(std::size_t index, F&& func, std::index_sequence<I...> /*seq*/, const std::vector<S>& s)
            NOEXCEPT(noexcept(func(s[0])))
        {
            using FT = std::add_pointer_t<R(F&&, const std::vector<S>&)>;
            static const std::array<FT, sizeof...(I)> ar = {{&apply_one<R, F, I, S>...}};
            return ar[index](std::forward<F>(func), s);
        }
    }

    template <class R, class F, class... S>
    inline R apply(std::size_t index, F&& func, const std::tuple<S...>& s) NOEXCEPT(noexcept(func(std::get<0>(s))))
    {
        return detail::apply<R>(index, std::forward<F>(func), std::make_index_sequence<sizeof...(S)>(), s);
    }

    template <class R, class F, class S>
    inline R apply(std::size_t index, F&& func, const std::vector<S>& s) NOEXCEPT(noexcept(func(s[0])))
    {
        return detail::apply<R>(index, std::forward<F>(func), std::make_index_sequence<sizeof(S)>(), s);
    }

    /***************************
     * nested_initializer_list *
     ***************************/

    template <class T, std::size_t I>
    struct nested_initializer_list
    {
        using type = std::initializer_list<typename nested_initializer_list<T, I - 1>::type>;
    };

    template <class T>
    struct nested_initializer_list<T, 0>
    {
        using type = T;
    };

    template <class T, std::size_t I>
    using nested_initializer_list_t = typename nested_initializer_list<T, I>::type;

    /******************************
     * nested_copy implementation *
     ******************************/

    template <class T, class S>
    inline void nested_copy(T&& iter, const S& s)
    {
        *iter++ = s;
    }

    template <class T, class S>
    inline void nested_copy(T&& iter, std::initializer_list<S> s)
    {
        for (auto it = s.begin(); it != s.end(); ++it)
        {
            nested_copy(std::forward<T>(iter), *it);
        }
    }

    /***********************************
     * resize_container implementation *
     ***********************************/

    template <class C>
    inline bool resize_container(C& c, typename C::size_type size)
    {
        c.resize(size);
        return true;
    }

    template <class T, std::size_t N>
    inline bool resize_container(std::array<T, N>& /*a*/, typename std::array<T, N>::size_type size)
    {
        return size == N;
    }

    template <std::size_t... I>
    inline bool resize_container(xt::fixed_shape<I...>&, std::size_t size)
    {
        return sizeof...(I) == size;
    }

    /*********************************
     * normalize_axis implementation *
     *********************************/

    // scalar normalize axis
    inline std::size_t normalize_axis(std::size_t dim, std::ptrdiff_t axis)
    {
        return axis < 0 ? static_cast<std::size_t>(static_cast<std::ptrdiff_t>(dim) + axis) : static_cast<std::size_t>(axis);
    }

    template <class E, class C>
    inline std::enable_if_t<!std::is_integral<std::decay_t<C>>::value &&
                     std::is_signed<typename std::decay_t<C>::value_type>::value,
                     rebind_container_t<std::size_t, std::decay_t<C>>>
    normalize_axis(E& expr, C&& axes)
    {
        rebind_container_t<std::size_t, std::decay_t<C>> res;
        resize_container(res, axes.size());

        for (std::size_t i = 0; i < axes.size(); ++i)
        {
            res[i] = normalize_axis(expr.dimension(), axes[i]);
        }

        XTENSOR_ASSERT(std::all_of(res.begin(), res.end(), [&expr](auto ax_el) { return ax_el < expr.dimension(); }));

        return res;
    }

    template <class C, class E>
    inline std::enable_if_t<!std::is_integral<std::decay_t<C>>::value && std::is_unsigned<typename std::decay_t<C>::value_type>::value, C&&>
    normalize_axis(E& expr, C&& axes)
    {
        static_cast<void>(expr);
        XTENSOR_ASSERT(std::all_of(axes.begin(), axes.end(), [&expr](auto ax_el) { return ax_el < expr.dimension(); }));
        return std::forward<C>(axes);
    }

    template <class R, class E, class C>
    inline auto forward_normalize(E& expr, C&& axes)
        -> std::enable_if_t<std::is_signed<std::decay_t<decltype(*std::begin(axes))>>::value, R>
    {
        R res;
        xt::resize_container(res, xtl::sequence_size(axes));
        auto dim = expr.dimension();
        std::transform(std::begin(axes), std::end(axes), std::begin(res), [&dim](auto ax_el) {
            return normalize_axis(dim, ax_el);
        });

        XTENSOR_ASSERT(std::all_of(res.begin(), res.end(), [&expr](auto ax_el) { return ax_el < expr.dimension(); }));

        return res;
    }

    template <class R, class E, class C>
    inline auto forward_normalize(E& expr, C&& axes)
        -> std::enable_if_t<!std::is_signed<std::decay_t<decltype(*std::begin(axes))>>::value && !std::is_same<R, std::decay_t<C>>::value, R>
    {
        static_cast<void>(expr);

        R res;
        xt::resize_container(res, xtl::sequence_size(axes));
        std::copy(std::begin(axes), std::end(axes), std::begin(res));
        XTENSOR_ASSERT(std::all_of(res.begin(), res.end(), [&expr](auto ax_el) { return ax_el < expr.dimension(); }));
        return res;
    }

    template <class R, class E, class C>
    inline auto forward_normalize(E& expr, C&& axes)
        -> std::enable_if_t<!std::is_signed<std::decay_t<decltype(*std::begin(axes))>>::value && std::is_same<R, std::decay_t<C>>::value, R&&>
    {
        static_cast<void>(expr);
        XTENSOR_ASSERT(std::all_of(std::begin(axes), std::end(axes), [&expr](auto ax_el) { return ax_el < expr.dimension(); }));
        return std::move(axes);
    }

    /******************
     * get_value_type *
     ******************/

    template <class T, class = void_t<>>
    struct get_value_type
    {
        using type = T;
    };

    template <class T>
    struct get_value_type<T, void_t<typename T::value_type>>
    {
        using type = typename T::value_type;
    };

    template <class T>
    using get_value_type_t = typename get_value_type<T>::type;

    /**********************
     * get implementation *
     **********************/

    // When subclassing from std::tuple not all compilers are able to correctly instantiate get
    // See here: https://stackoverflow.com/a/37188019/2528668
    template <std::size_t I, template <typename... Args> class T, typename ...Args>
    decltype(auto) get(T<Args...>&& v)
    {
      return std::get<I>(static_cast<std::tuple<Args...>&&>(v));
    }

    template <std::size_t I, template <typename... Args> class T, typename ...Args>
    decltype(auto) get(T<Args...>& v)
    {
      return std::get<I>(static_cast<std::tuple<Args...>&>(v));
    }

    template <std::size_t I, template <typename... Args> class T, typename ...Args>
    decltype(auto) get(const T<Args...> & v)
    {
      return std::get<I>(static_cast<const std::tuple<Args...> &>(v));
    }

    /***************************
     * apply_cv implementation *
     ***************************/

    namespace detail
    {
        template <class T, class U, bool = std::is_const<std::remove_reference_t<T>>::value,
                  bool = std::is_volatile<std::remove_reference_t<T>>::value>
        struct apply_cv_impl
        {
            using type = U;
        };

        template <class T, class U>
        struct apply_cv_impl<T, U, true, false>
        {
            using type = const U;
        };

        template <class T, class U>
        struct apply_cv_impl<T, U, false, true>
        {
            using type = volatile U;
        };

        template <class T, class U>
        struct apply_cv_impl<T, U, true, true>
        {
            using type = const volatile U;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, false, false>
        {
            using type = U&;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, true, false>
        {
            using type = const U&;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, false, true>
        {
            using type = volatile U&;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, true, true>
        {
            using type = const volatile U&;
        };
    }

    template <class T, class U>
    struct apply_cv
    {
        using type = typename detail::apply_cv_impl<T, U>::type;
    };

    template <class T, class U>
    using apply_cv_t = typename apply_cv<T, U>::type;


    /**************************
    * to_array implementation *
    ***************************/

    namespace detail
    {
        template <class T, std::size_t N, std::size_t... I>
        constexpr std::array<std::remove_cv_t<T>, N> to_array_impl(T (&a)[N], std::index_sequence<I...>)
        {
            return {{a[I]...}};
        }
    }

    template <class T, std::size_t N>
    constexpr std::array<std::remove_cv_t<T>, N> to_array(T (&a)[N])
    {
        return detail::to_array_impl(a, std::make_index_sequence<N>{});
    }

    /********************************
     * sequence_size implementation *
     ********************************/

    // equivalent to std::size(c) in c++17
    template <class C>
    constexpr auto sequence_size(const C& c) -> decltype(c.size())
    {
        return c.size();
    }

    // equivalent to std::size(a) in c++17
    template <class T, std::size_t N>
    constexpr std::size_t sequence_size(const T (&)[N])
    {
        return N;
    }

    /*************************************
     * has_data_interface implementation *
     *************************************/

    template <class E, class = void>
    struct has_data_interface : std::false_type
    {
    };

    template <class E>
    struct has_data_interface<E, void_t<decltype(std::declval<E>().data())>>
        : std::true_type
    {
    };

    template <class E, class = void>
    struct has_strides : std::false_type
    {
    };

    template <class E>
    struct has_strides<E, void_t<decltype(std::declval<E>().strides())>>
        : std::true_type
    {
    };

    template <class E, class = void>
    struct has_iterator_interface : std::false_type
    {
    };

    template <class E>
    struct has_iterator_interface<E, void_t<decltype(std::declval<E>().begin())>>
        : std::true_type
    {
    };

    /******************************
     * is_iterator implementation *
     ******************************/

    template <class E, class = void>
    struct is_iterator : std::false_type
    {
    };

    template <class E>
    struct is_iterator<E, void_t<decltype(
        *std::declval<const E>(),
        std::declval<const E>() == std::declval<const E>(),
        std::declval<const E>() != std::declval<const E>(),
        ++ (*std::declval<E*>()),
        (*std::declval<E*>()) ++,
        std::true_type())>>
        : std::true_type
    {
    };

    /********************************************
     * xtrivial_default_construct implemenation *
     ********************************************/

#if defined(_GLIBCXX_USE_CXX11_ABI)
#if _GLIBCXX_USE_CXX11_ABI || (defined(_GLIBCXX_USE_DUAL_ABI) && !_GLIBCXX_USE_DUAL_ABI)
#define XTENSOR_GLIBCXX_USE_CXX11_ABI 1
#endif
#endif

#if !defined(__GNUG__) || defined(_LIBCPP_VERSION) || defined(XTENSOR_GLIBCXX_USE_CXX11_ABI)

    template <class T>
    using xtrivially_default_constructible = std::is_trivially_default_constructible<T>;

#else

    template <class T>
    using xtrivially_default_constructible = std::has_trivial_default_constructor<T>;

#endif
#undef XTENSOR_GLIBCXX_USE_CXX11_ABI

    /*************************
     * conditional type cast *
     *************************/

    template <bool condition, class T>
    struct conditional_cast_functor;

    template <class T>
    struct conditional_cast_functor<false, T> : public xtl::identity
    {
    };

    template <class T>
    struct conditional_cast_functor<true, T>
    {
        template <class U>
        inline auto operator()(U&& u) const
        {
            return static_cast<T>(std::forward<U>(u));
        }
    };

    /**
     * @brief Perform a type cast when a condition is true.
     * If <tt>condition</tt> is true, return <tt>static_cast<T>(u)</tt>,
     * otherwise return <tt>u</tt> unchanged. This is useful when an unconditional
     * static_cast would force undesired type conversions in some situations where
     * an error or warning would be desired. The condition determines when the
     * explicit cast is ok.
     */
    template <bool condition, class T, class U>
    inline auto conditional_cast(U&& u)
    {
        return conditional_cast_functor<condition, T>()(std::forward<U>(u));
    }

    /**********************
     * tracking allocator *
     **********************/

    namespace alloc_tracking
    {
        inline bool& enabled()
        {
            static bool enabled;
            return enabled;
        }

        inline void enable()
        {
            enabled() = true;
        }

        inline void disable()
        {
            enabled() = false;
        }

        enum policy
        {
            print,
            assert
        };
    }

    template <class T, class A, alloc_tracking::policy P>
    struct tracking_allocator
        : private A
    {
        using base_type = A;
        using value_type = typename A::value_type;
        using reference = typename A::reference;
        using const_reference = typename A::const_reference;
        using pointer = typename A::pointer;
        using const_pointer = typename A::const_pointer;
        using size_type = typename A::size_type;
        using difference_type = typename A::difference_type;

        tracking_allocator() = default;

        T* allocate(std::size_t n)
        {
            if (alloc_tracking::enabled())
            {
                if (P == alloc_tracking::print)
                {
                    std::cout << "xtensor allocating: " << n << "" << std::endl;
                }
                else if (P == alloc_tracking::assert)
                {
                    XTENSOR_THROW(std::runtime_error,
                                  "xtensor allocation of " + std::to_string(n) +
                                  " elements detected");
                }
            }
            return base_type::allocate(n);
        }

        using base_type::deallocate;
        using base_type::construct;
        using base_type::destroy;

        template <class U>
        struct rebind
        {
            using traits = std::allocator_traits<A>;
            using other = tracking_allocator<U, typename traits::template rebind_alloc<U>, P>;
        };
    };

    template <class T, class AT, alloc_tracking::policy PT, class U, class AU, alloc_tracking::policy PU>
    inline bool operator==(const tracking_allocator<T, AT, PT>&, const tracking_allocator<U, AU, PU>&)
    {
      return std::is_same<AT, AU>::value;
    }

    template <class T, class AT, alloc_tracking::policy PT, class U, class AU, alloc_tracking::policy PU>
    inline bool operator!=(const tracking_allocator<T, AT, PT>& a, const tracking_allocator<U, AU, PU>& b)
    {
      return !(a == b);
    }

    /*****************
     * has_assign_to *
     *****************/

    template <class E1, class E2, class = void>
    struct has_assign_to : std::false_type
    {
    };

    template <class E1, class E2>
    struct has_assign_to<E1, E2, void_t<decltype(std::declval<const E2&>().assign_to(std::declval<E1&>()))>>
        : std::true_type
    {
    };

    /********************
     * rebind_container *
     ********************/

    template <class X, template <class, class> class C, class T, class A>
    struct rebind_container<X, C<T, A>>
    {
        using traits = std::allocator_traits<A>;
        using allocator = typename traits::template rebind_alloc<X>;
        using type = C<X, allocator>;
    };

#if defined(__GNUC__) && __GNUC__ > 6 && !defined(__clang__) && __cplusplus >= 201703L
    template <class X, class T, std::size_t N>
    struct rebind_container<X, std::array<T, N>>
    {
        using type = std::array<X, N>;
    };
#else
    template <class X, template <class, std::size_t> class C, class T, std::size_t N>
    struct rebind_container<X, C<T, N>>
    {
        using type = C<X, N>;
    };
#endif

    /********************
     * get_strides_type *
     ********************/

    template <class S>
    struct get_strides_type
    {
        using type = typename rebind_container<std::ptrdiff_t, S>::type;
    };

    template <std::size_t... I>
    struct get_strides_type<fixed_shape<I...>>
    {
        // TODO we could compute the strides statically here.
        //  But we'll need full constexpr support to have a
        //  homogenous ``compute_strides`` method
        using type = std::array<std::ptrdiff_t, sizeof...(I)>;
    };

    template <class C>
    using get_strides_t = typename get_strides_type<C>::type;

    /*******************
     * inner_reference *
     *******************/

    template <class ST>
    struct inner_reference
    {
        using storage_type = std::decay_t<ST>;
        using type = std::conditional_t<std::is_const<std::remove_reference_t<ST>>::value,
                                        typename storage_type::const_reference,
                                        typename storage_type::reference>;
    };

    template <class ST>
    using inner_reference_t = typename inner_reference<ST>::type;
}

#endif
