/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xutils.hpp"

#include <initializer_list>
#include <type_traits>
#include <tuple>
#include <complex>

#include "gtest/gtest.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xshape.hpp"
#include "xtensor/xview.hpp"
#include "test_common_macros.hpp"


namespace xt
{
    using std::size_t;

    struct for_each_fn
    {
        short a;
        int b;
        float c;
        double d;

        template <class T>
        void operator()(T t)
        {
            if (std::is_same<T, short>::value)
            {
                a = static_cast<short>(t);
            }
            else if (std::is_same<T, int>::value)
            {
                b = static_cast<int>(t);
            }
            else if (std::is_same<T, float>::value)
            {
                c = static_cast<float>(t);
            }
            else if (std::is_same<T, double>::value)
            {
                d = static_cast<double>(t);
            }
        }
    };

    TEST(utils, for_each)
    {
        for_each_fn fn;
        short a = 1;
        int b = 4;
        float c = float(1.2);
        double d = 2.3;
        auto t = std::make_tuple(a, b, c, d);
        for_each(fn, t);
        ASSERT_TRUE(a == fn.a && b == fn.b && c == fn.c && d == fn.d);
        EXPECT_FALSE(noexcept(for_each(fn, t)));
    }

    TEST(utils, accumulate)
    {
        auto func = [](int i, int j) { return i + j; };
        const std::tuple<int, int, int> t(3, 4, 1);
        EXPECT_EQ(8, accumulate(func, 0, t));
        EXPECT_FALSE(noexcept(accumulate(func, 0, t)));

        auto func_ne = [](int i, int j) noexcept { return i + j; };
        EXPECT_EQ(8, accumulate(func_ne, 0, t));
        EXPECT_TRUE(noexcept(accumulate(func_ne, 0, t)));
    }

    template <class... T>
    auto foo(const std::tuple<T...>& t)
    {
        auto func = [](int i) { return i; };
        return apply<int>(1, func, t);
    }

    int fun(int i) noexcept{ return 2 * i; }
    TEST(utils, apply)
    {
        ASSERT_TRUE(foo(std::make_tuple(1, 2, 3)) == 2);
        EXPECT_FALSE(noexcept(foo(std::make_tuple(1, 2, 3))));
        auto func_ne = [](int i) noexcept { return i; };
        auto t = std::make_tuple(1, 2, 3);
#if (_MSC_VER >= 1910)
        EXPECT_FALSE(noexcept(apply<int>(1, func_ne, t)));
#else
        EXPECT_TRUE(noexcept(apply<int>(1, func_ne, t)));
#endif
    }

    TEST(utils, conditional_cast)
    {
        EXPECT_TRUE((std::is_same<decltype(conditional_cast<false, double>(1)), int>::value));
        EXPECT_TRUE((std::is_same<decltype(conditional_cast<true, double>(1)), double>::value));
    }

    TEST(utils, has_data_interface)
    {
        bool b = has_data_interface<xarray<int>>::value;
        EXPECT_TRUE(b);
        b = has_data_interface<const xarray<int>>::value;
        EXPECT_TRUE(b);
        b = has_data_interface<const xtensor<double, 2>>::value;
        EXPECT_TRUE(b);
        b = has_data_interface<const xtensor_fixed<double, xshape<3, 4>>>::value;
        EXPECT_TRUE(b);

        xarray<int> a = xarray<int>::from_shape({3, 4, 5});
        auto f = a + a - 23;
        auto v2 = strided_view(a, {all(), 1, all()});
        auto vv2 = strided_view(v2, {all(), 2});
        auto v3 = strided_view(f, {all(), 2});

        b = has_data_interface<decltype(v2)>::value;
        EXPECT_TRUE(b);
        b = has_data_interface<decltype(vv2)>::value;
        EXPECT_TRUE(b);
        b = has_data_interface<decltype(v3)>::value;
        EXPECT_FALSE(b);
    }

    TEST(utils, has_strides)
    {
        bool b = has_strides<xarray<int>>::value;
        EXPECT_TRUE(b);
        b = has_strides<const xarray<int>>::value;
        EXPECT_TRUE(b);
        b = has_strides<const xtensor<double, 2>>::value;
        EXPECT_TRUE(b);
        b = has_strides<const xtensor_fixed<double, xshape<3, 4>>>::value;
        EXPECT_TRUE(b);

        xarray<int> a = xarray<int>::from_shape({3, 4, 5});
        auto f = a + a - 23;
        auto v2 = strided_view(a, {all(), 1, all()});
        auto vv2 = strided_view(v2, {all(), 2});
        auto v3 = strided_view(f, {all(), 2});

        b = has_strides<decltype(v2)>::value;
        EXPECT_TRUE(b);
        b = has_strides<decltype(vv2)>::value;
        EXPECT_TRUE(b);

#ifndef _MSC_VER
        // TODO fix this test for MSVC 2015!
        b = has_strides<decltype(v3)>::value;
        EXPECT_TRUE(b);
#endif
    }

    TEST(utils, has_simd_interface)
    {
        bool b = has_simd_interface<xarray<int>>::value;

        EXPECT_TRUE(b);
        b = has_simd_interface<const xarray<int>>::value;
        EXPECT_TRUE(b);
        b = has_simd_interface<const xtensor<double, 2>>::value;
        EXPECT_TRUE(b);
        b = has_simd_interface<const xtensor_fixed<double, xshape<3, 4>>>::value;
        EXPECT_TRUE(b);

        xarray<int> a = xarray<int>::from_shape({3, 4, 5});
        auto f = a + a - 23;
        auto v2 = strided_view(a, {all(), 1, all()});
        auto vv2 = strided_view(v2, {all(), 2});
        auto v3 = strided_view(f, {all(), 2});

        b = has_simd_interface<decltype(v2)>::value;
        EXPECT_FALSE(b);
        b = has_simd_interface<decltype(vv2)>::value;
        EXPECT_FALSE(b);
        b = has_simd_interface<decltype(v3)>::value;
        EXPECT_FALSE(b);

        auto xv = xt::view(a, 1);
        b = has_simd_interface<decltype(xv)>::value;
        EXPECT_TRUE(b);
    }

    TEST(utils, allocation_tracking)
    {
        using arr_t = xarray<double, layout_type::row_major,
                             tracking_allocator<double, std::allocator<double>, alloc_tracking::policy::assert>>;

        arr_t a = {{1,2,3}, {5,6,7}};

        alloc_tracking::enable();
        EXPECT_THROW(arr_t b = a + 123, std::runtime_error);
        EXPECT_NO_THROW(a.resize({2, 3}));
        EXPECT_NO_THROW(a.resize({3, 2}));
        EXPECT_THROW(a.resize({3, 15}), std::runtime_error);
        alloc_tracking::disable();
        EXPECT_NO_THROW(arr_t c = a);
    }

    TEST(utils, static_dimension)
    {
        std::ptrdiff_t sdim = static_dimension<std::vector<int>>::value;
        EXPECT_EQ(sdim, -1);
        sdim = static_dimension<std::array<int, 4>>::value;
        EXPECT_EQ(sdim, 4);
        sdim = static_dimension<xt::const_array<char, 12>>::value;
        EXPECT_EQ(sdim, 12);
        sdim = static_dimension<xt::fixed_shape<4, 1, 2, 3>>::value;
        EXPECT_EQ(sdim, 4);
        sdim = static_dimension<xt::sequence_view<std::array<std::ptrdiff_t, 2>, 1, 2>>::value;
        EXPECT_EQ(sdim, 1);
        sdim = static_dimension<xt::sequence_view<std::array<std::ptrdiff_t, 2>, 1, -1>>::value;
        EXPECT_EQ(sdim, -1);
        sdim = static_dimension<xt::sequence_view<xt::fixed_shape<4, 1, 2, 3>, 1, 4>>::value;
        EXPECT_EQ(sdim, 3);
    }
}
