/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xio.hpp"
#include "test_common.hpp"
#include <type_traits>

namespace xt
{
    using xarray_dynamic = xarray<int, layout_type::dynamic>;

    TEST(xarray, shaped_constructor)
    {
        {
            SCOPED_TRACE("row_major constructor");
            row_major_result<> rm;
            xarray_dynamic ra(rm.m_shape, layout_type::row_major);
            compare_shape(ra, rm);
        }

        {
            SCOPED_TRACE("column_major constructor");
            column_major_result<> cm;
            xarray<int, layout_type::column_major> ca(cm.m_shape);
            compare_shape(ca, cm);
        }

        {
            SCOPED_TRACE("from shape");
            std::array<std::size_t, 3> shp = {5, 4, 2};
            std::vector<std::size_t> shp_as_vec = {5, 4, 2};
            auto ca = xarray<int, layout_type::column_major>::from_shape({3, 2, 1});
            auto cb = xarray<int, layout_type::column_major>::from_shape(shp);
            std::vector<std::size_t> expected_shape = {3, 2, 1};
            EXPECT_EQ(expected_shape, ca.shape());
            EXPECT_EQ(shp_as_vec, cb.shape());
        }
    }

    TEST(xarray, strided_constructor)
    {
        central_major_result<> cmr;
        xarray<int, layout_type::dynamic> cma(cmr.m_shape, cmr.m_strides);
        compare_shape(cma, cmr);
    }

    TEST(xarray, valued_constructor)
    {
        {
            SCOPED_TRACE("row_major valued constructor");
            row_major_result<> rm;
            int value = 2;
            xarray_dynamic ra(rm.m_shape, value, layout_type::row_major);
            compare_shape(ra, rm);
            xarray_dynamic::storage_type vec(ra.size(), value);
            EXPECT_EQ(ra.storage(), vec);
        }

        {
            SCOPED_TRACE("column_major valued constructor");
            column_major_result<> cm;
            int value = 2;
            xarray<int, layout_type::column_major> ca(cm.m_shape, value);
            compare_shape(ca, cm);
            xarray_dynamic::storage_type vec(ca.size(), value);
            EXPECT_EQ(ca.storage(), vec);
        }
    }

    TEST(xarray, strided_valued_constructor)
    {
        central_major_result<> cmr;
        int value = 2;
        xarray<int, layout_type::dynamic> cma(cmr.m_shape, cmr.m_strides, value);
        compare_shape(cma, cmr);
        xarray_dynamic::storage_type vec(cma.size(), value);
        EXPECT_EQ(cma.storage(), vec);
    }

    TEST(xarray, xscalar_constructor)
    {
        xscalar<int> xs(2);
        xarray<int> a(xs);
        EXPECT_EQ(a(), xs());
    }

    TEST(xarray, copy_semantic)
    {
        central_major_result<> res;
        int value = 2;
        xarray_dynamic a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("copy constructor");
            xarray_dynamic b(a);
            compare_shape(a, b);
            EXPECT_EQ(a.storage(), b.storage());
        }

        {
            SCOPED_TRACE("assignment operator");
            row_major_result<> r;
            xarray_dynamic c(r.m_shape, 0);
            EXPECT_NE(a.storage(), c.storage());
            c = a;
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
        }
    }

    TEST(xarray, move_semantic)
    {
        central_major_result<> res;
        int value = 2;
        xarray_dynamic a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("move constructor");
            xarray_dynamic tmp(a);
            xarray_dynamic b(std::move(tmp));
            compare_shape(a, b);
            EXPECT_EQ(a.storage(), b.storage());
        }

        {
            SCOPED_TRACE("move assignment");
            row_major_result<> r;
            xarray_dynamic c(r.m_shape, 0);
            EXPECT_NE(a.storage(), c.storage());
            xarray_dynamic tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
        }
    }

    TEST(xarray, resize)
    {
        xarray_dynamic a;
        std::vector<size_t> shape = {2, 2};
        xarray_dynamic::strides_type strides = {2, 1};
        test_resize(a);
#ifdef XTENSOR_ENABLE_ASSERT
        EXPECT_NO_THROW(a.resize(shape));
        EXPECT_NO_THROW(a.resize(shape, layout_type::row_major));
        EXPECT_NO_THROW(a.resize(shape, strides));
#endif
    }

    TEST(xarray, reshape)
    {
        xarray_dynamic a;
        test_reshape(a);
        test_throwing_reshape(a);
    }

    TEST(xarray, transpose)
    {
        xarray_dynamic a;
        test_transpose(a);
    }

    TEST(xarray, transpose_row)
    {
        xarray<float> a = {{0, 1, 1, 1}};
        xarray<float> res = xt::transpose(a);

        xarray<float>::shape_type sh = {4, 1};
        EXPECT_EQ(res.shape(), sh);
        EXPECT_EQ(res(0, 0), 0.f);
        EXPECT_EQ(res(1, 0), 1.f);
        EXPECT_EQ(res(2, 0), 1.f);
        EXPECT_EQ(res(3, 0), 1.f);
    }

#if !(defined(XTENSOR_ENABLE_ASSERT) && defined(XTENSOR_DISABLE_EXCEPTIONS))
    TEST(xarray, access)
    {
        xarray_dynamic a;
        test_access(a);
    }
#endif

    TEST(xarray, unchecked)
    {
        xarray_dynamic a;
        test_unchecked(a);
    }

    TEST(xarray, at)
    {
        xarray_dynamic a;
        test_at(a);
    }

#if !(defined(XTENSOR_ENABLE_ASSERT) && defined(XTENSOR_DISABLE_EXCEPTIONS))
    TEST(xarray, element)
    {
        xarray_dynamic a;
        test_element(a);
    }
#endif

    TEST(xarray, indexed_access)
    {
        xarray_dynamic a;
        test_indexed_access(a);
    }

    TEST(xarray, broadcast_shape)
    {
        xarray_dynamic a;
        test_broadcast(a);
        test_broadcast2(a);
    }

    TEST(xarray, iterator)
    {
        xarray<int, layout_type::row_major> arm;
        xarray<int, layout_type::column_major> acm;
        test_iterator(arm, acm);
    }

    TEST(xarray, fill)
    {
        xarray_dynamic a;
        test_fill(a);
    }

    TEST(xarray, initializer_list)
    {
        xarray_dynamic a0(1);
        xarray_dynamic a1({1, 2});
        xarray_dynamic a2({{1, 2}, {2, 4}, {5, 6}});
        EXPECT_EQ(1, a0());
        EXPECT_EQ(2, a1(1));
        EXPECT_EQ(4, a2(1, 1));
    }

    TEST(xarray, zerod)
    {
        xarray_dynamic a;
        EXPECT_EQ(0u, a.dimension());
        EXPECT_EQ(0, a());

        xarray_dynamic b = {1, 2, 3};
        xarray_dynamic c(2 + xt::sum(b));
        EXPECT_EQ(8, c());

        EXPECT_EQ(8, c(1, 2));
        xindex idx = { 1, 2 };
        EXPECT_EQ(8, c.element(idx.cbegin(), idx.cend()));
    }

    TEST(xarray, xiterator)
    {
        xarray_dynamic a;
        test_xiterator(a);
    }

    TEST(xarray, reverse_xiterator)
    {
        xarray_dynamic a;
        test_reverse_xiterator(a);
    }

    TEST(xarray, cross_layout_assign)
    {
        xarray<int, layout_type::row_major> a = {{1, 2, 3, 4},
                                                 {5, 6, 7, 8}};
        xarray<int, layout_type::column_major> b = {{1, 2, 3, 4},
                                                    {5, 6, 7, 8}};

        xarray<int, layout_type::column_major> ra = a;
        EXPECT_EQ(b, ra);

        xarray<int, layout_type::row_major> rb = b;
        EXPECT_EQ(a, rb);
    }

    TEST(xarray, end_optimized_stride)
    {
        xarray_dynamic a = {1};
        EXPECT_FALSE((a.begin() == a.end()));
    }

    TEST(xarray, move_from_xtensor)
    {
        xtensor<double, 3> a = {{{1,2,3}, {4,5,6}}, {{10, 10, 10}, {1,5,10}}};
        xtensor<double, 3> a1 = a;
        xarray<double> b(std::move(a1));
        EXPECT_EQ(a, b);
        EXPECT_TRUE(std::equal(a.strides().begin(), a.strides().end(), b.strides().begin()) && a.strides().size() == b.strides().size());
        EXPECT_TRUE(std::equal(a.backstrides().begin(), a.backstrides().end(), b.backstrides().begin()) && a.backstrides().size() == b.backstrides().size());
        EXPECT_EQ(a.layout(), b.layout());

        xtensor<double, 3> a2 = a;
        xarray<double> c;
        c = std::move(a2);

        EXPECT_EQ(a, c);
        EXPECT_TRUE(std::equal(a.strides().begin(), a.strides().end(), c.strides().begin()) && a.strides().size() == c.strides().size());
        EXPECT_TRUE(std::equal(a.backstrides().begin(), a.backstrides().end(), c.backstrides().begin()) && a.backstrides().size() == c.backstrides().size());
        EXPECT_EQ(a.layout(), c.layout());
    }

    TEST(xarray, operator_brace)
    {
        xt::xarray<size_t> a = {{0,1,2}, {3,4,5}};
        xt::xarray<size_t> b = {{0,1,2}, {30,40,50}};
        a(1,3) = 30;
        a(1,4) = 40;
        a(1,5) = 50;
        EXPECT_EQ(a, b);
        EXPECT_THROW(a(2, 0));
        EXPECT_THROW(a(0, 3));
    }

    TEST(xarray, periodic)
    {
        xt::xarray<size_t> a = {{0,1,2}, {3,4,5}};
        xt::xarray<size_t> b = {{0,1,2}, {30,40,50}};
        a.periodic(-1,3) = 30;
        a.periodic(-1,4) = 40;
        a.periodic(-1,5) = 50;
        EXPECT_EQ(a, b);
    }

    TEST(xarray, front)
    {
        xt::xarray<size_t> a = {{1,2,3}, {4,5,6}};
        EXPECT_EQ(a.front(), 1);
    }

    TEST(xarray, back)
    {
        xt::xarray<size_t> a = {{1,2,3}, {4,5,6}};
        EXPECT_EQ(a.back(), 6);
    }

    TEST(xarray, flat)
    {
        {
            xt::xarray<size_t, xt::layout_type::row_major> a = {{0,1,2}, {3,4,5}};
            xt::xarray<size_t, xt::layout_type::row_major> b = {{0,1,2}, {30,40,50}};
            a.flat(3) = 30;
            a.flat(4) = 40;
            a.flat(5) = 50;
            EXPECT_EQ(a, b);
            EXPECT_THROW(a.flat(6));
        }
        {
            xt::xarray<size_t, xt::layout_type::column_major> a = {{0,1,2}, {3,4,5}};
            xt::xarray<size_t, xt::layout_type::column_major> b = {{0,1,2}, {30,40,50}};
            a.flat(1) = 30;
            a.flat(3) = 40;
            a.flat(5) = 50;
            EXPECT_EQ(a, b);
            EXPECT_THROW(a.flat(6));
        }
    }

    TEST(xarray, in_bounds)
    {
        xt::xarray<size_t> a = {{0,1,2}, {3,4,5}};
        EXPECT_TRUE(a.in_bounds(0,0) == true);
        EXPECT_TRUE(a.in_bounds(2,0) == false);
    }

    TEST(xarray, iterator_types)
    {
        using array_type = xarray<int>;
        test_iterator_types<array_type, int*, const int*>();
    }

    auto test_reshape_compile() {
        xt::xarray<double> a = xt::zeros<double>({5, 5});
        return a.reshape({1, 25});
    }

    TEST(xarray, reshape_return)
    {
        auto a = test_reshape_compile();
        EXPECT_EQ(a.shape(), std::vector<std::size_t>({1, 25}));
    }

    TEST(xarray, type_traits)
    {
        using array_type = xt::xarray<double>;
        EXPECT_TRUE(std::is_constructible<array_type>::value);

        EXPECT_TRUE(std::is_default_constructible<array_type>::value);

        EXPECT_TRUE(std::is_copy_constructible<array_type>::value);

        EXPECT_TRUE(std::is_move_constructible<array_type>::value);
        EXPECT_TRUE(std::is_nothrow_move_constructible<array_type>::value);

        EXPECT_TRUE(std::is_copy_assignable<array_type>::value);

        EXPECT_TRUE(std::is_move_assignable<array_type>::value);
        EXPECT_TRUE(std::is_nothrow_move_assignable<array_type>::value);

        EXPECT_TRUE(std::is_destructible<array_type>::value);
        EXPECT_TRUE(std::is_nothrow_destructible<array_type>::value);
    }

    TEST(xarray, bool_container)
    {
        xt::xarray<int> a{1, 0, 1, 0}, b{1, 1, 0, 0};

        xt::xarray<bool> c = a & b;
        EXPECT_TRUE(c(0));
        EXPECT_FALSE(c(1));
        EXPECT_FALSE(c(2));
        EXPECT_FALSE(c(3));

        xt::xarray<bool> d = a | b;
        EXPECT_TRUE(d(0));
        EXPECT_TRUE(d(1));
        EXPECT_TRUE(d(2));
        EXPECT_FALSE(d(3));
    }
}
