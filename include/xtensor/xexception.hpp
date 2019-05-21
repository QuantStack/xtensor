/****************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_EXCEPTION_HPP
#define XTENSOR_EXCEPTION_HPP

#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

namespace xt
{
    /*******************
     * broadcast_error *
     *******************/

    class broadcast_error : public std::runtime_error
    {
    public:

        explicit broadcast_error(const char* msg)
            : std::runtime_error(msg)
        {
        }
    };

    template <class S1, class S2>
    [[noreturn]] void throw_broadcast_error(const S1& lhs, const S2& rhs);

    /**********************************
     * broadcast_error implementation *
     **********************************/

#ifdef NDEBUG
    // Do not inline this function
    template <class S1, class S2>
    [[noreturn]] void throw_broadcast_error(const S1&, const S2&)
    {
        throw broadcast_error("Incompatible dimension of arrays, compile in DEBUG for more info");
    }
#else
    template <class S1, class S2>
    [[noreturn]] void throw_broadcast_error(const S1& lhs, const S2& rhs)
    {
        std::ostringstream buf("Incompatible dimension of arrays:", std::ios_base::ate);

        buf << "\n LHS shape = (";
        using size_type1 = typename S1::value_type;
        std::ostream_iterator<size_type1> iter1(buf, ", ");
        std::copy(lhs.cbegin(), lhs.cend(), iter1);

        buf << ")\n RHS shape = (";
        using size_type2 = typename S2::value_type;
        std::ostream_iterator<size_type2> iter2(buf, ", ");
        std::copy(rhs.cbegin(), rhs.cend(), iter2);
        buf << ")";

        throw broadcast_error(buf.str().c_str());
    }
#endif

    /*******************
     * transpose_error *
     *******************/

    class transpose_error : public std::runtime_error
    {
    public:

        explicit transpose_error(const char* msg)
            : std::runtime_error(msg)
        {
        }
    };

    /***************
     * check_index *
     ***************/

    template <class S, class... Args>
    void check_index(const S& shape, Args... args);

    template <class S, class It>
    void check_element_index(const S& shape, It first, It last);

    namespace detail
    {
        template <class S, std::size_t dim>
        inline void check_index_impl(const S&)
        {
        }

        template <class S, std::size_t dim, class T, class... Args>
        inline void check_index_impl(const S& shape, T arg, Args... args)
        {
            if (sizeof...(Args) + 1 > shape.size())
            {
                check_index_impl<S, dim>(shape, args...);
            }
            else
            {
                if (std::size_t(arg) >= std::size_t(shape[dim]) && shape[dim] != 1)
                {
                    throw std::out_of_range("index " + std::to_string(arg)
                                            + " is out of bounds for axis " + std::to_string(dim)
                                            + " with size " + std::to_string(shape[dim]));
                }
                check_index_impl<S, dim + 1>(shape, args...);
            }
        }
    }

    template <class S, class... Args>
    inline void check_index(const S& shape, Args... args)
    {
        using value_type = typename S::value_type;
        detail::check_index_impl<S, 0>(shape, static_cast<value_type>(args)...);
    }

    template <class S, class It>
    inline void check_element_index(const S& shape, It first, It last)
    {
        using value_type = typename std::iterator_traits<It>::value_type;
        using size_type = typename S::size_type;
        auto dst = static_cast<size_type>(last - first);
        It efirst = last - static_cast<std::ptrdiff_t>((std::min)(shape.size(), dst));
        std::size_t axis = 0;

        if (shape.empty())
        {
            if (first != last && *(--last) != value_type(0))
            {
                throw std::out_of_range("index out of bound (empty array)");
            }
            else
            {
                return;
            }
        }
        else
        {
            while (efirst != last)
            {
                if (*efirst >= value_type(shape[axis]) && shape[axis] != 1)
                {
                    throw std::out_of_range("index " + std::to_string(*efirst)
                                            + " is out of bounds for axis " + std::to_string(axis)
                                            + " with size " + std::to_string(shape[axis]));
                }
                ++efirst, ++axis;
            }
        }
    }

    /*******************
     * check_dimension *
     *******************/

    template <class S, class... Args>
    inline void check_dimension(const S& shape, Args...)
    {
        if (sizeof...(Args) > shape.size())
        {
            throw std::out_of_range("Number of arguments (" + std::to_string(sizeof...(Args))
                                    + ") us greater " + "than the number of dimensions ("
                                    + std::to_string(shape.size()) + ")");
        }
    }

    /****************
     * check_access *
     ****************/

    template <class S, class... Args>
    inline void check_access(const S& shape, Args... args)
    {
        check_dimension(shape, args...);
        check_index(shape, args...);
    }

#ifdef XTENSOR_ENABLE_ASSERT
#define XTENSOR_TRY(expr) XTENSOR_TRY_IMPL(expr, __FILE__, __LINE__)
#define XTENSOR_TRY_IMPL(expr, file, line)                                                         \
    try                                                                                            \
    {                                                                                              \
        expr;                                                                                      \
    }                                                                                              \
    catch (std::exception & e)                                                                     \
    {                                                                                              \
        throw std::runtime_error(std::string(file) + ':' + std::to_string(line)                    \
                                 + ": check failed\n\t" + std::string(e.what()));                  \
    }
#else
#define XTENSOR_TRY(expr)
#endif

#ifdef XTENSOR_ENABLE_ASSERT
#define XTENSOR_ASSERT(expr) XTENSOR_ASSERT_IMPL(expr, __FILE__, __LINE__)
#define XTENSOR_ASSERT_IMPL(expr, file, line)                                                      \
    if (!(expr))                                                                                   \
    {                                                                                              \
        throw std::runtime_error(std::string(file) + ':' + std::to_string(line)                    \
                                 + ": assertion failed (" #expr ") \n\t");                         \
    }
#else
#define XTENSOR_ASSERT(expr)
#endif

#ifdef XTENSOR_ENABLE_CHECK_DIMENSION
#define XTENSOR_CHECK_DIMENSION(S, ARGS) XTENSOR_TRY(check_dimension(S, ARGS))
#else
#define XTENSOR_CHECK_DIMENSION(S, ARGS)
#endif

#ifdef XTENSOR_ENABLE_ASSERT
#define XTENSOR_ASSERT_MSG(expr, msg)                                                              \
    if (!(expr))                                                                                   \
    {                                                                                              \
        throw std::runtime_error(std::string("Assertion error!\n") + msg + "\n  " + __FILE__ + '(' \
                                 + std::to_string(__LINE__) + ")\n");                              \
    }
#else
#define XTENSOR_ASSERT_MSG(expr, msg)
#endif

#define XTENSOR_PRECONDITION(expr, msg)                                                            \
    if (!(expr))                                                                                   \
    {                                                                                              \
        throw std::runtime_error(std::string("Precondition violation!\n") + msg + "\n  "           \
                                 + __FILE__ + '(' + std::to_string(__LINE__) + ")\n");             \
    }
}
#endif  // XEXCEPTION_HPP
