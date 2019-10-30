/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xexpression.hpp"

#include <sstream>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xio.hpp"

namespace xt
{
    auto fun()
    {
        auto sa = make_xshared(xarray<double>({{1,2,3,4}, {5,6,7,8}}));
        return sa + sa * sa - sa;
    }

    TEST(xexpression, shared_basic)
    {
        xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        xarray<double> ca = {{1,2,3,4}, {5,6,7,8}};

        auto sa = make_xshared(std::move(a));

        EXPECT_EQ(sa.dimension(), std::size_t(2));
        EXPECT_EQ(sa.shape(), ca.shape());
        EXPECT_EQ(sa.strides(), ca.strides());
        EXPECT_EQ(sa(1, 3), ca(1, 3));
        EXPECT_EQ(sa.storage(), ca.storage());
        EXPECT_EQ(sa.data_offset(), ca.data_offset());
        EXPECT_EQ(sa.data()[0], ca.data()[0]);
        layout_type L = decltype(sa)::static_layout;
        bool contig = decltype(sa)::contiguous_layout;
        EXPECT_EQ(L, XTENSOR_DEFAULT_LAYOUT);
        EXPECT_EQ(contig, true);

        EXPECT_EQ(sa.use_count(), 2);
        auto cpysa = sa;
        EXPECT_EQ(sa.use_count(), 3);
        
        std::stringstream buffer;
        buffer << sa;
        EXPECT_EQ(buffer.str(), "{{ 1.,  2.,  3.,  4.},\n { 5.,  6.,  7.,  8.}}");
    }

    TEST(xexpression, shared_iterator)
    {
        xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        xarray<double> ca = {{1,2,3,4}, {5,6,7,8}};

        auto sa = make_xshared(std::move(a));

        EXPECT_EQ(*(sa.begin()), *(ca.begin()));
        EXPECT_EQ(*(sa.cbegin()), *(ca.cbegin()));
        EXPECT_EQ(*(sa.rbegin()), *(ca.rbegin()));
        EXPECT_EQ(*(sa.crbegin()), *(ca.crbegin()));

        auto it = sa.begin() + 8;
        EXPECT_EQ(it, sa.end());
        auto cit = sa.cbegin() + 8;
        EXPECT_EQ(cit, sa.cend());

        auto rit = sa.rbegin() + 8;
        EXPECT_EQ(rit, sa.rend());
        auto crit = sa.crbegin() + 8;
        EXPECT_EQ(crit, sa.crend());
    }

    template <class E>
    auto test_sum(E&& e)
    {
        return share(e) + share(e);
    }

    TEST(xexpression, shared_xfunctions)
    {
        xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        xarray<double> ca = {{1,2,3,4}, {5,6,7,8}};
        xarray<double> acopy(a);

        auto sa = make_xshared(std::move(a));

        auto expr1 = sa + sa;
        auto expr2 = a + a;
        auto expr3 = test_sum(std::move(acopy));

        EXPECT_EQ(sa.use_count(), 4);
        EXPECT_TRUE(all(equal(expr1, expr2)));
        EXPECT_TRUE(all(equal(expr1, expr3)));
        std::stringstream buffer;
        buffer << expr1;
        EXPECT_EQ(buffer.str(), "{{  2.,   4.,   6.,   8.},\n { 10.,  12.,  14.,  16.}}");

        // Compilation test
        auto sexpr1 = make_xshared(std::move(expr1));
        using expr_type = decltype(sexpr1);
        using strides_type = typename expr_type::strides_type;
        using inner_strides_type = typename expr_type::inner_strides_type;
        using backstrides_type = typename expr_type::backstrides_type;
        using inner_strides_tybackstrides_typepe = typename expr_type::inner_backstrides_type;
    }

    TEST(xexpression, shared_expr_return)
    {
        auto expr = fun();
        xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        EXPECT_EQ(expr, a * a);
    }

    TEST(xexpression, temporary_type)
    {
        using dyn_shape = xt::svector<std::size_t, 4, std::allocator<std::size_t>, true>;
        using dyn_tmp = xt::detail::xtype_for_shape<dyn_shape>::type<int, XTENSOR_DEFAULT_LAYOUT>;
        using dyn_exp = xt::xarray<int>;
        constexpr bool dyn_res = std::is_same<dyn_tmp, dyn_exp>::value;
        EXPECT_TRUE(dyn_res);
        
        using sta_shape = std::array<std::size_t, 4>;
        using sta_tmp = xt::detail::xtype_for_shape<sta_shape>::type<int, XTENSOR_DEFAULT_LAYOUT>;
        using sta_exp = xt::xtensor<int, 4>;
        constexpr bool sta_res = std::is_same<sta_tmp, sta_exp>::value;
        EXPECT_TRUE(sta_res);
    }

}  // namespace xt
