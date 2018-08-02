/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_SLICE_HPP
#define XTENSOR_SLICE_HPP

#include <cstddef>
#include <type_traits>
#include <utility>
#include <map>

#include <xtl/xtype_traits.hpp>

#include "xutils.hpp"


#ifndef XTENSOR_CONSTEXPR
    #if(defined(_MSC_VER) || __GNUC__ < 8)
        #define XTENSOR_CONSTEXPR inline
        #define XTENSOR_GLOBAL_CONSTEXPR static const
    #else
        #define XTENSOR_CONSTEXPR constexpr
        #define XTENSOR_GLOBAL_CONSTEXPR constexpr
    #endif
#endif

namespace xt
{
    struct xall_tag {};
    struct xnewaxis_tag {};
    struct xellipsis_tag {};

    template <class A, class B, class C>
    struct xrange_adaptor;

    namespace placeholders
    {
        // xtensor universal placeholder
        struct xtuph {};

        template <class... Args>
        struct rangemaker
        {
            ptrdiff_t rng[3]; // = { 0, 0, 0 };
        };

        XTENSOR_CONSTEXPR xtuph get_tuph_or_val(std::ptrdiff_t /*val*/, std::true_type)
        {
            return xtuph();
        }

        XTENSOR_CONSTEXPR std::ptrdiff_t get_tuph_or_val(std::ptrdiff_t val, std::false_type)
        {
            return val;
        }

        template <class A, class B, class C>
        struct rangemaker<A, B, C>
        {
            XTENSOR_CONSTEXPR operator xrange_adaptor<A, B, C>()
            {
                return {
                    get_tuph_or_val(rng[0], std::is_same<A, xtuph>()),
                    get_tuph_or_val(rng[1], std::is_same<B, xtuph>()),
                    get_tuph_or_val(rng[2], std::is_same<C, xtuph>())
                };
            }

            ptrdiff_t rng[3];// = { 0, 0, 0 };
        };

        template <class A, class B>
        struct rangemaker<A, B>
        {
            XTENSOR_CONSTEXPR operator xrange_adaptor<A, B, xt::placeholders::xtuph>()
            {
                return {
                    get_tuph_or_val(rng[0], std::is_same<A, xtuph>()),
                    get_tuph_or_val(rng[1], std::is_same<B, xtuph>()),
                    xtuph()
                };
            }

            ptrdiff_t rng[3];  // = { 0, 0, 0 };
        };

        template <class... OA>
        XTENSOR_CONSTEXPR auto operator|(const rangemaker<OA...>& rng, const std::ptrdiff_t& t)
        {
            auto nrng = rangemaker<OA..., ptrdiff_t>{rng.rng[0], rng.rng[1], rng.rng[2]};
            nrng.rng[sizeof...(OA)] = t;
            return nrng;
        }

        template <class... OA>
        XTENSOR_CONSTEXPR auto operator|(const rangemaker<OA...>& rng, const xt::placeholders::xtuph& /*t*/)
        {
            auto nrng = rangemaker<OA..., xt::placeholders::xtuph>{rng.rng[0], rng.rng[1], rng.rng[2]};
            return nrng;
        }


        XTENSOR_GLOBAL_CONSTEXPR xtuph _{};
        XTENSOR_GLOBAL_CONSTEXPR rangemaker<> _r{0, 0, 0};
        XTENSOR_GLOBAL_CONSTEXPR xall_tag _a{};
        XTENSOR_GLOBAL_CONSTEXPR xnewaxis_tag _n{};
        XTENSOR_GLOBAL_CONSTEXPR xellipsis_tag _e{};
    }

    inline auto xnone()
    {
        return placeholders::xtuph();
    }

    /**********************
     * xslice declaration *
     **********************/

    template <class D>
    class xslice
    {
    public:

        using derived_type = D;

        derived_type& derived_cast() noexcept;
        const derived_type& derived_cast() const noexcept;

    protected:

        xslice() = default;
        ~xslice() = default;

        xslice(const xslice&) = default;
        xslice& operator=(const xslice&) = default;

        xslice(xslice&&) = default;
        xslice& operator=(xslice&&) = default;
    };

    template <class S>
    using is_xslice = std::is_base_of<xslice<S>, S>;

    template <class E, class R = void>
    using disable_xslice = typename std::enable_if<!is_xslice<E>::value, R>::type;

    template <class... E>
    using has_xslice = xtl::disjunction<is_xslice<E>...>;

    /**********************
     * xrange declaration *
     **********************/

    template <class T>
    class xrange : public xslice<xrange<T>>
    {
    public:

        using size_type = T;
        using self_type = xrange<T>;

        xrange() = default;
        xrange(size_type start_val, size_type stop_val) noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;
        size_type step_size(std::size_t i, std::size_t n = 1) const noexcept;
        size_type revert_index(std::size_t i) const noexcept;

        bool contains(size_type i) const noexcept;

        bool operator==(const self_type& rhs) const noexcept;
        bool operator!=(const self_type& rhs) const noexcept;

    private:

        size_type m_start;
        size_type m_size;
    };

    /******************************
     * xstepped_range declaration *
     ******************************/

    template <class T>
    class xstepped_range : public xslice<xstepped_range<T>>
    {
    public:

        using size_type = T;
        using self_type = xstepped_range<T>;

        xstepped_range() = default;
        xstepped_range(size_type start_val, size_type stop_val, size_type step) noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;
        size_type step_size(std::size_t i, std::size_t n = 1) const noexcept;
        size_type revert_index(std::size_t i) const noexcept;

        bool contains(size_type i) const noexcept;

        bool operator==(const self_type& rhs) const noexcept;
        bool operator!=(const self_type& rhs) const noexcept;

    private:

        size_type m_start;
        size_type m_size;
        size_type m_step;
    };

    /********************
     * xall declaration *
     ********************/

    template <class T>
    class xall : public xslice<xall<T>>
    {
    public:

        using size_type = T;
        using self_type = xall<T>;

        xall() = default;
        explicit xall(size_type size) noexcept;

        constexpr size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;
        size_type step_size(std::size_t i, std::size_t n = 1) const noexcept;
        size_type revert_index(std::size_t i) const noexcept;

        bool contains(size_type i) const noexcept;

        bool operator==(const self_type& rhs) const noexcept;
        bool operator!=(const self_type& rhs) const noexcept;

    private:

        size_type m_size;
    };

    /**
     * Returns a slice representing a full dimension,
     * to be used as an argument of view function.
     * @sa view, strided_view
     */
    inline auto all() noexcept
    {
        return xall_tag();
    }

    /**
     * Returns a slice representing all remaining dimensions,
     * and selecting all in these dimensions. Ellipsis will expand
     * to a series of `all()` slices, until the number of slices is
     * equal to the number of dimensions of the source array.
     *
     * Note: ellipsis can only be used in strided_view!
     *
     * \code{.cpp}
     * xarray<double> a = xarray<double>::from_shape({5, 5, 1, 1, 5});
     * auto v = xt::strided_view(a, {2, xt::ellipsis(), 2});
     * // equivalent to using {2, xt::all(), xt::all(), xt::all(), 2};
     * \endcode
     *
     * @sa strided_view
     */
    inline auto ellipsis() noexcept
    {
        return xellipsis_tag();
    }

    /************************
     * xnewaxis declaration *
     ************************/

    template <class T>
    class xnewaxis : public xslice<xnewaxis<T>>
    {
    public:

        using size_type = T;

        xnewaxis() = default;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;
        size_type step_size(std::size_t i, std::size_t n = 1) const noexcept;
        size_type revert_index(std::size_t i) const noexcept;

        bool contains(size_type i) const noexcept;
    };

    /**
     * Returns a slice representing a new axis of length one,
     * to be used as an argument of view function.
     * @sa view, strided_view
     */
    inline auto newaxis() noexcept
    {
        return xnewaxis_tag();
    }

    /***************************
     * xkeep_slice declaration *
     ***************************/

    template <class T>
    class xkeep_slice : public xslice<xkeep_slice<T>>
    {
    public:

        using container_type = svector<T>;
        using size_type = typename container_type::value_type;

        template <class C>
        explicit xkeep_slice(const C& cont);
        template <class C>
        explicit xkeep_slice(C& cont);
        explicit xkeep_slice(container_type&& cont);

        template <class S>
        xkeep_slice(std::initializer_list<S> t);

        size_type operator()(size_type i) const noexcept;
        size_type size() const noexcept;

        void normalize(std::size_t s);

        size_type step_size(std::size_t i, std::size_t n = 1) const noexcept;
        size_type revert_index(std::size_t i) const;

        bool contains(size_type i) const noexcept;

    private:

        container_type m_indices;
        container_type m_raw_indices;
    };

    namespace detail
    {
        template <class T>
        using disable_integral_keep = std::enable_if_t<!std::is_integral<std::decay_t<T>>::value,
            xkeep_slice<typename std::decay_t<T>::value_type>>;

        template <class T, class R>
        using enable_integral_keep = std::enable_if_t<std::is_integral<T>::value, xkeep_slice<R>>;
    }

    /**
     * Create a non-contigous slice from a container of indices to keep.
     * Note: this slice cannot be used in the xstrided_view!
     *
     * \code{.cpp}
     * xt::xarray<double> a = xt::arange(9);
     * a.reshape({3, 3});
     * xt::view(a, xt::keep(0, 2); // => {{0, 1, 2}, {6, 7, 8}}
     * xt::view(a, xt::keep(1, 1, 1); // => {{3, 4, 5}, {3, 4, 5}, {3, 4, 5}}
     * \endcode
     *
     * @param indices The indices container
     * @return instance of xkeep_slice
     */
    template <class T>
    inline detail::disable_integral_keep<T> keep(T&& indices)
    {
        return xkeep_slice<typename std::decay_t<T>::value_type>(std::forward<T>(indices));
    }

    template <class R = std::ptrdiff_t, class T>
    inline detail::enable_integral_keep<T, R> keep(T i)
    {
        using slice_type = xkeep_slice<R>;
        using container_type = typename slice_type::container_type;
        container_type tmp = { static_cast<R>(i) };
        return slice_type(std::move(tmp));
    }

    template <class R = std::ptrdiff_t, class Arg0, class Arg1, class... Args>
    inline xkeep_slice<R> keep(Arg0 i0, Arg1 i1, Args... args)
    {
        using slice_type = xkeep_slice<R>;
        using container_type = typename slice_type::container_type;
        container_type tmp = { static_cast<R>(i0), static_cast<R>(i1), static_cast<R>(args)... };
        return slice_type(std::move(tmp));
    }

    /***************************
     * xdrop_slice declaration *
     ***************************/


    template <class T>
    class xdrop_slice : public xslice<xdrop_slice<T>>
    {
    public:

        using container_type = svector<T>;
        using size_type = typename container_type::value_type;

        template <class C>
        explicit xdrop_slice(const C& cont);
        template <class C>
        explicit xdrop_slice(C& cont);
        explicit xdrop_slice(container_type&& cont);

        template <class S>
        xdrop_slice(std::initializer_list<S> t);

        size_type operator()(size_type i) const noexcept;
        size_type size() const noexcept;

        void normalize(std::size_t s);

        size_type step_size(std::size_t i, std::size_t n = 1) const noexcept;
        size_type revert_index(std::size_t i) const;

        bool contains(size_type i) const noexcept;

    private:

        container_type m_indices;
        container_type m_raw_indices;
        std::map<size_type, size_type> m_inc;
        size_type m_size;
    };

    namespace detail
    {
        template <class T>
        using disable_integral_drop = std::enable_if_t<!std::is_integral<std::decay_t<T>>::value,
                                     xdrop_slice<typename std::decay_t<T>::value_type>>;

        template <class T, class R>
        using enable_integral_drop = std::enable_if_t<std::is_integral<T>::value, xdrop_slice<R>>;
    }

    /**
     * Create a non-contigous slice from a container of indices to drop.
     * Note: this slice cannot be used in the xstrided_view!
     *
     * \code{.cpp}
     * xt::xarray<double> a = xt::arange(9);
     * a.reshape({3, 3});
     * xt::view(a, xt::drop(0, 2); // => {{3, 4, 5}}
     * \endcode
     *
     * @param indices The container of indices to drop
     * @return instance of xdrop_slice
     */
    template <class T>
    inline detail::disable_integral_drop<T> drop(T&& indices)
    {
        return xdrop_slice<typename std::decay_t<T>::value_type>(std::forward<T>(indices));
    }

    template <class R = std::ptrdiff_t, class T>
    inline detail::enable_integral_drop<T, R> drop(T i)
    {
        using slice_type = xdrop_slice<R>;
        using container_type = typename slice_type::container_type;
        container_type tmp = { static_cast<R>(i) };
        return slice_type(std::move(tmp));
    }

    template <class R = std::ptrdiff_t, class Arg0, class Arg1, class... Args>
    inline xdrop_slice<R> drop(Arg0 i0, Arg1 i1, Args... args)
    {
        using slice_type = xdrop_slice<R>;
        using container_type = typename slice_type::container_type;
        container_type tmp = { static_cast<R>(i0), static_cast<R>(i1), static_cast<R>(args)... };
        return slice_type(std::move(tmp));
    }

    /******************
     * xrange_adaptor *
     ******************/

    template <class A, class B, class C>
    struct xrange_adaptor
    {
        xrange_adaptor(A start_val, B stop_val, C step)
            : m_start(start_val), m_stop(stop_val), m_step(step)
        {
        }

        auto normalize(std::ptrdiff_t val, std::size_t ssize) const
        {
            std::ptrdiff_t size = static_cast<std::ptrdiff_t>(ssize);
            val = (val >= 0) ? val : val + size;
            return std::max(std::ptrdiff_t(0), std::min(size, val));
        }

        auto get_stepped_range(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step, std::size_t ssize) const
        {
            std::ptrdiff_t size = static_cast<std::ptrdiff_t>(ssize);
            start = (start >= 0) ? start : start + size;
            stop = (stop >= 0) ? stop : stop + size;

            if(step > 0)
            {
                start = std::max(std::ptrdiff_t(0), std::min(size, start));
                stop  = std::max(std::ptrdiff_t(0), std::min(size, stop));
            }
            else
            {
                start = std::max(std::ptrdiff_t(-1), std::min(size - 1, start));
                stop  = std::max(std::ptrdiff_t(-1), std::min(size - 1, stop));
            }

            return xstepped_range<std::ptrdiff_t>(start, stop, step);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<std::is_integral<MI>::value &&
                                std::is_integral<MA>::value &&
                                std::is_integral<STEP>::value,
                                xstepped_range<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return get_stepped_range(m_start, m_stop, m_step, size);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<!std::is_integral<MI>::value &&
                                std::is_integral<MA>::value &&
                                std::is_integral<STEP>::value,
                                xstepped_range<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return get_stepped_range(m_step > 0 ? 0 : static_cast<std::ptrdiff_t>(size) - 1, m_stop, m_step, size);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<std::is_integral<MI>::value &&
                                !std::is_integral<MA>::value &&
                                std::is_integral<STEP>::value,
                                xstepped_range<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            auto sz = static_cast<std::ptrdiff_t>(size);
            return get_stepped_range(m_start, m_step > 0 ? sz : -(sz + 1), m_step, size);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<std::is_integral<MI>::value &&
                                std::is_integral<MA>::value &&
                                !std::is_integral<STEP>::value,
                                xrange<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return xrange<std::ptrdiff_t>(normalize(m_start, size), normalize(m_stop, size));
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<!std::is_integral<MI>::value &&
                                !std::is_integral<MA>::value &&
                                std::is_integral<STEP>::value,
                                xstepped_range<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            std::ptrdiff_t start = m_step >= 0 ? 0 : static_cast<std::ptrdiff_t>(size) - 1;
            std::ptrdiff_t stop = m_step >= 0 ? static_cast<std::ptrdiff_t>(size) : -1;
            return xstepped_range<std::ptrdiff_t>(start, stop, m_step);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<std::is_integral<MI>::value &&
                                !std::is_integral<MA>::value &&
                                !std::is_integral<STEP>::value,
                                xrange<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return xrange<std::ptrdiff_t>(normalize(m_start, size), static_cast<std::ptrdiff_t>(size));
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<!std::is_integral<MI>::value &&
                                std::is_integral<MA>::value &&
                                !std::is_integral<STEP>::value,
                                xrange<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return xrange<std::ptrdiff_t>(0, normalize(m_stop, size));
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<!std::is_integral<MI>::value &&
                                !std::is_integral<MA>::value &&
                                !std::is_integral<STEP>::value,
                                xall<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return xall<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(size));
        }

    private:

        A m_start;
        B m_stop;
        C m_step;
    };

    namespace detail
    {
        template <class T, class E = void>
        struct cast_if_integer
        {
            using type = T;

            type operator()(T t)
            {
                return t;
            }
        };

        template <class T>
        struct cast_if_integer<T, std::enable_if_t<std::is_integral<T>::value>>
        {
            using type = std::ptrdiff_t;

            type operator()(T t)
            {
                return static_cast<type>(t);
            }
        };

        template <class T>
        using cast_if_integer_t = typename cast_if_integer<T>::type;
    }

    /**
     * Select a range from start_val to stop_val.
     * You can use the shorthand `_` syntax to select from the start or until the end.
     *
     * \code{.cpp}
     * using namespace xt::placeholders;  // to enable _ syntax
     *
     * range(3, _)  // select from index 3 to the end
     * range(_, 5)  // select from index o to 5
     * range(_, _)  // equivalent to `all()`
     * \endcode
     *
     * @sa view, strided_view
     */
    template <class A, class B>
    inline auto range(A start_val, B stop_val)
    {
        return xrange_adaptor<detail::cast_if_integer_t<A>, detail::cast_if_integer_t<B>, placeholders::xtuph>(
            detail::cast_if_integer<A>{}(start_val), detail::cast_if_integer<B>{}(stop_val), placeholders::xtuph());
    }

    /**
     * Select a range from start_val to stop_val with step
     * You can use the shorthand `_` syntax to select from the start or until the end.
     *
     * \code{.cpp}
     * using namespace xt::placeholders;  // to enable _ syntax
     * range(3, _, 5)  // select from index 3 to the end with stepsize 5
     * \endcode
     *
     * @sa view, strided_view
     */
    template <class A, class B, class C>
    inline auto range(A start_val, B stop_val, C step)
    {
        return xrange_adaptor<detail::cast_if_integer_t<A>, detail::cast_if_integer_t<B>, detail::cast_if_integer_t<C>>(
            detail::cast_if_integer<A>{}(start_val), detail::cast_if_integer<B>{}(stop_val), detail::cast_if_integer<C>{}(step));
    }


    /******************************************************
     * homogeneous get_size for integral types and slices *
     ******************************************************/

    template <class S>
    inline disable_xslice<S, std::size_t> get_size(const S&) noexcept
    {
        return 1;
    }

    template <class S>
    inline auto get_size(const xslice<S>& slice) noexcept
    {
        return slice.derived_cast().size();
    }

    /*******************************************************
     * homogeneous step_size for integral types and slices *
     *******************************************************/

    template <class S>
    inline disable_xslice<S, std::size_t> step_size(const S&, std::size_t) noexcept
    {
        return 0;
    }

    template <class S>
    inline disable_xslice<S, std::size_t> step_size(const S&, std::size_t, std::size_t) noexcept
    {
        return 0;
    }

    template <class S>
    inline auto step_size(const xslice<S>& slice, std::size_t idx) noexcept
    {
        return slice.derived_cast().step_size(idx);
    }

    template <class S>
    inline auto step_size(const xslice<S>& slice, std::size_t idx, std::size_t n) noexcept
    {
        return slice.derived_cast().step_size(idx, n);
    }

    /*********************************************
     * homogeneous value for integral and slices *
     *********************************************/

    template <class S, class I>
    inline disable_xslice<S, std::size_t> value(const S& s, I) noexcept
    {
        return static_cast<std::size_t>(s);
    }

    template <class S, class I>
    inline auto value(const xslice<S>& slice, I i) noexcept
    {
        using ST = typename S::size_type;
        return slice.derived_cast()(static_cast<ST>(i));
    }

    /****************************************
     * homogeneous get_slice_implementation *
     ****************************************/

    template <class E, class SL>
    inline auto get_slice_implementation(E& /*e*/, SL&& slice, std::size_t /*index*/)
    {
        return std::forward<SL>(slice);
    }

    template <class E, class T>
    inline auto get_slice_implementation(E& e, xkeep_slice<T>&& slice, std::size_t index)
    {
        slice.normalize(e.shape()[index]);
        return slice;
    }

    template <class E, class T>
    inline auto get_slice_implementation(E& e, xdrop_slice<T>&& slice, std::size_t index)
    {
        slice.normalize(e.shape()[index]);
        return slice;
    }

    template <class E>
    inline auto get_slice_implementation(E& e, xall_tag&&, std::size_t index)
    {
        return xall<typename E::size_type>(e.shape()[index]);
    }

    template <class E>
    inline auto get_slice_implementation(E& /*e*/, xnewaxis_tag&&, std::size_t /*index*/)
    {
        return xnewaxis<typename E::size_type>();
    }

    template <class E, class A, class B, class C>
    inline auto get_slice_implementation(E& e, xrange_adaptor<A, B, C>&& adaptor, std::size_t index)
    {
        return adaptor.get(e.shape()[index]);
    }

    /******************************
     * homogeneous get_slice_type *
     ******************************/

    namespace detail
    {
        template <class E, class SL>
        struct get_slice_type_impl
        {
            using type = SL;
        };

        template <class E>
        struct get_slice_type_impl<E, xall_tag>
        {
            using type = xall<typename E::size_type>;
        };

        template <class E>
        struct get_slice_type_impl<E, xnewaxis_tag>
        {
            using type = xnewaxis<typename E::size_type>;
        };

        template <class E, class A, class B, class C>
        struct get_slice_type_impl<E, xrange_adaptor<A, B, C>>
        {
            using type = decltype(xrange_adaptor<A, B, C>(A(), B(), C()).get(0));
        };
    }

    template <class E, class SL>
    using get_slice_type = typename detail::get_slice_type_impl<E, std::remove_reference_t<SL>>::type;

    /*************************
     * xslice implementation *
     *************************/

    template <class D>
    inline auto xslice<D>::derived_cast() noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xslice<D>::derived_cast() const noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    /*************************
     * xrange implementation *
     *************************/

    template <class T>
    inline xrange<T>::xrange(size_type start_val, size_type stop_val) noexcept
        : m_start(start_val), m_size(stop_val > start_val ? stop_val - start_val: 0)
    {
    }

    template <class T>
    inline auto xrange<T>::operator()(size_type i) const noexcept -> size_type
    {
        return m_start + i;
    }

    template <class T>
    inline auto xrange<T>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class T>
    inline auto xrange<T>::step_size() const noexcept -> size_type
    {
        return 1;
    }

    template <class T>
    inline auto xrange<T>::step_size(std::size_t /*i*/, std::size_t n) const noexcept -> size_type
    {
        return static_cast<size_type>(n);
    }

    template <class T>
    inline auto xrange<T>::revert_index(std::size_t i) const noexcept -> size_type
    {
        return i - m_start;
    }

    template <class T>
    inline bool xrange<T>::contains(size_type i) const noexcept
    {
        return i >= m_start && i < m_start + m_size;
    }

    template <class T>
    inline bool xrange<T>::operator==(const self_type& rhs) const noexcept
    {
        return (m_start == rhs.m_start) && (m_size == rhs.m_size);
    }

    template <class T>
    inline bool xrange<T>::operator!=(const self_type& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    /********************************
     * xtepped_range implementation *
     ********************************/

    template <class T>
    inline xstepped_range<T>::xstepped_range(size_type start_val, size_type stop_val, size_type step) noexcept
        : m_start(start_val), m_size(size_type(std::ceil(double(stop_val - start_val) / double(step)))), m_step(step)
    {
    }


    template <class T>
    inline auto xstepped_range<T>::operator()(size_type i) const noexcept -> size_type
    {
        return m_start + i * m_step;
    }

    template <class T>
    inline auto xstepped_range<T>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class T>
    inline auto xstepped_range<T>::step_size() const noexcept -> size_type
    {
        return m_step;
    }

    template <class T>
    inline auto xstepped_range<T>::step_size(std::size_t /*i*/, std::size_t n) const noexcept -> size_type
    {
        return m_step * static_cast<size_type>(n);
    }

    template <class T>
    inline auto xstepped_range<T>::revert_index(std::size_t i) const noexcept -> size_type
    {
        return (i - m_start) / m_step;
    }

    template <class T>
    inline bool xstepped_range<T>::contains(size_type i) const noexcept
    {
        return i >= m_start && i < m_start + m_size * m_step && ((i - m_start) % m_step == 0);
    }

    template <class T>
    inline bool xstepped_range<T>::operator==(const self_type& rhs) const noexcept
    {
        return (m_start == rhs.m_start) && (m_size == rhs.m_size) && (m_step == rhs.m_step);
    }

    template <class T>
    inline bool xstepped_range<T>::operator!=(const self_type& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    /***********************
     * xall implementation *
     ***********************/

    template <class T>
    inline xall<T>::xall(size_type size) noexcept
        : m_size(size)
    {
    }

    template <class T>
    constexpr auto xall<T>::operator()(size_type i) const noexcept -> size_type
    {
        return i;
    }

    template <class T>
    inline auto xall<T>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class T>
    inline auto xall<T>::step_size() const noexcept -> size_type
    {
        return 1;
    }

    template <class T>
    inline auto xall<T>::step_size(std::size_t /*i*/, std::size_t n) const noexcept -> size_type
    {
        return static_cast<size_type>(n);
    }

    template <class T>
    inline auto xall<T>::revert_index(std::size_t i) const noexcept -> size_type
    {
        return i;
    }

    template <class T>
    inline bool xall<T>::contains(size_type i) const noexcept
    {
        return i < m_size;
    }

    template <class T>
    inline bool xall<T>::operator==(const self_type& rhs) const noexcept
    {
        return m_size == rhs.m_size;
    }

    template <class T>
    inline bool xall<T>::operator!=(const self_type& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    /***************************
     * xnewaxis implementation *
     ***************************/

    template <class T>
    inline auto xnewaxis<T>::operator()(size_type) const noexcept -> size_type
    {
        return 0;
    }

    template <class T>
    inline auto xnewaxis<T>::size() const noexcept -> size_type
    {
        return 1;
    }

    template <class T>
    inline auto xnewaxis<T>::step_size() const noexcept -> size_type
    {
        return 0;
    }

    template <class T>
    inline auto xnewaxis<T>::step_size(std::size_t /*i*/, std::size_t /*n*/) const noexcept -> size_type
    {
        return 0;
    }

    template <class T>
    inline auto xnewaxis<T>::revert_index(std::size_t i) const noexcept -> size_type
    {
        return i;
    }

    template <class T>
    inline bool xnewaxis<T>::contains(size_type i) const noexcept
    {
        return i == 0;
    }

    /******************************
     * xkeep_slice implementation *
     ******************************/

    template <class T>
    template <class C>
    inline xkeep_slice<T>::xkeep_slice(const C& cont)
        : m_raw_indices(cont.begin(), cont.end())
    {
    }

    template <class T>
    template <class C>
    inline xkeep_slice<T>::xkeep_slice(C& cont)
        : m_raw_indices(cont.begin(), cont.end())
    {
    }

    template <class T>
    inline xkeep_slice<T>::xkeep_slice(container_type&& cont)
        : m_raw_indices(std::move(cont))
    {
    }

    template <class T>
    template <class S>
    inline xkeep_slice<T>::xkeep_slice(std::initializer_list<S> t)
        : m_raw_indices(t.size())
    {
        std::transform(t.begin(), t.end(), m_raw_indices.begin(),
            [](auto t) { return static_cast<size_type>(t); });
    }

    template <class T>
    inline void xkeep_slice<T>::normalize(std::size_t shape)
    {
        m_indices.resize(m_raw_indices.size());
        std::size_t sz = m_indices.size();
        for (std::size_t i = 0; i < sz; ++i)
        {
            m_indices[i] = m_raw_indices[i] < 0 ? static_cast<std::ptrdiff_t>(shape) + m_raw_indices[i] : m_raw_indices[i];
        }
    }

    template <class T>
    inline auto xkeep_slice<T>::operator()(size_type i) const noexcept -> size_type
    {
        return m_indices[static_cast<std::size_t>(i)];
    }

    template <class T>
    inline auto xkeep_slice<T>::size() const noexcept -> size_type
    {
        return static_cast<size_type>(m_raw_indices.size());
    }

    template <class T>
    inline auto xkeep_slice<T>::step_size(std::size_t i, std::size_t n) const noexcept -> size_type
    {
        // special case one-past-end step (should be removed soon)
        if (i == m_indices.size())
        {
            return 1;
        }
        else
        {
            --i;
            return m_indices[i + n] - m_indices[i];
        }
    }

    template <class T>
    inline auto xkeep_slice<T>::revert_index(std::size_t i) const -> size_type
    {
        auto it = std::find(m_indices.begin(), m_indices.end(), i);
        if (it != m_indices.end())
        {
            return std::distance(m_indices.begin(), it);
        }
        else
        {
            throw std::runtime_error("Index i (" + std::to_string(i) + ") not in indices of islice.");
        }
    }

    template <class T>
    inline bool xkeep_slice<T>::contains(size_type i) const noexcept
    {
        return (std::find(m_indices.begin(), m_indices.end(), i) == m_indices.end()) ? false : true;
    }

    /******************************
     * xdrop_slice implementation *
     ******************************/

    template <class T>
    template <class C>
    inline xdrop_slice<T>::xdrop_slice(const C& cont)
        : m_raw_indices(cont.begin(), cont.end())
    {
    }

    template <class T>
    template <class C>
    inline xdrop_slice<T>::xdrop_slice(C& cont)
        : m_raw_indices(cont.begin(), cont.end())
    {
    }

    template <class T>
    inline xdrop_slice<T>::xdrop_slice(container_type&& cont)
        : m_raw_indices(std::move(cont))
    {
    }

    template <class T>
    template <class S>
    inline xdrop_slice<T>::xdrop_slice(std::initializer_list<S> t)
        : m_raw_indices(t.size())
    {
        std::transform(t.begin(), t.end(), m_raw_indices.begin(),
            [](auto t) { return static_cast<size_type>(t); });
    }

    template <class T>
    inline void xdrop_slice<T>::normalize(std::size_t shape)
    {
        m_size = static_cast<size_type>(shape - m_raw_indices.size());

        m_indices.resize(m_raw_indices.size());
        std::size_t sz = m_indices.size();
        for (std::size_t i = 0; i < sz; ++i)
        {
            m_indices[i] = m_raw_indices[i] < 0 ? static_cast<std::ptrdiff_t>(shape) + m_raw_indices[i] : m_raw_indices[i];
        }
        size_type cum = size_type(0);
        size_type prev_cum = cum;
        for (std::size_t i = 0; i < sz; ++i)
        {
            std::size_t ind = i;
            size_type d = m_indices[i];
            while (i + 1 < sz && m_indices[i + 1] == m_indices[i] + 1)
            {
                ++i;
            }
            cum += (static_cast<size_type>(i) - static_cast<size_type>(ind)) + 1;
            m_inc[d - prev_cum] = cum;
            prev_cum = cum;
        }
    }

    template <class T>
    inline auto xdrop_slice<T>::operator()(size_type i) const noexcept -> size_type
    {
        if (i < m_inc.begin()->first)
        {
            return i;
        }
        else
        {
            auto iter = --m_inc.upper_bound(i);
            return i + iter->second;
        }
    }

    template <class T>
    inline auto xdrop_slice<T>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class T>
    inline auto xdrop_slice<T>::step_size(std::size_t i, std::size_t n) const noexcept -> size_type
    {
        // special case one-past-end step (should be removed soon)
        if (i == static_cast<size_type>(m_indices.size()))
        {
            return 1;
        }
        else
        {
            return (*this)(i + n) - (*this)(i);
        }
    }

    template <class T>
    inline auto xdrop_slice<T>::revert_index(std::size_t i) const -> size_type
    {
        if (i < m_inc.begin()->first)
        {
            return i;
        }
        else
        {
            auto iter = --m_inc.lower_bound(i);
            auto check = iter->first + iter->second;
            if (check > i)
                --iter;
            return i - iter->second;
        }
    }

    template <class T>
    inline bool xdrop_slice<T>::contains(size_type i) const noexcept
    {
        return (std::find(m_indices.begin(), m_indices.end(), i) == m_indices.end()) ? true : false;
    }
}

#undef XTENSOR_CONSTEXPR

#endif
