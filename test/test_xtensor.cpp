/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "test_common.hpp"
#include <type_traits>

#include "xtensor/xio.hpp"

namespace xt
{
    using storage_type = std::array<std::size_t, 3>;
    using xtensor_dynamic = xtensor<int, 3, layout_type::dynamic>;

    TEST(xtensor, default_constructor)
    {
        xtensor_dynamic a = {};
        a.size();
        EXPECT_EQ(a.size(), size_t(0));
        EXPECT_EQ(a.dimension(), size_t(3));
    }

    TEST(xtensor, initializer_constructor)
    {
        xtensor_dynamic t
          {{{0, 1, 2},
            {3, 4, 5},
            {6, 7, 8}},
           {{9, 10, 11},
            {12, 13, 14},
            {15, 16, 17}}};
        EXPECT_EQ(t.dimension(), size_t(3));
        EXPECT_EQ(t(0, 0, 1), 1);
        EXPECT_EQ(t.shape()[0], size_t(2));
    }

    TEST(xtensor, shaped_constructor)
    {
        {
            SCOPED_TRACE("row_major constructor");
            row_major_result<storage_type> rm;
            xtensor_dynamic ra(rm.m_shape, layout_type::row_major);
            compare_shape(ra, rm);
        }

        {
            SCOPED_TRACE("column_major constructor");
            column_major_result<storage_type> cm;
            xtensor_dynamic ca(cm.m_shape, layout_type::column_major);
            compare_shape(ca, cm);
        }

        {
            SCOPED_TRACE("from shape");
            std::array<std::size_t, 3> shp = {5, 4, 2};
            std::vector<std::size_t> shp_as_vec = {5, 4, 2};
            auto ca = xtensor<int, 3>::from_shape({3, 2, 1});
            auto cb = xtensor<int, 3>::from_shape(shp_as_vec);
            std::vector<std::size_t> expected_shape = {3, 2, 1};
            EXPECT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), ca.shape().begin()));
            EXPECT_TRUE(std::equal(shp.begin(), shp.end(), cb.shape().begin()));
        }
    }

    TEST(xtensor, strided_constructor)
    {
        central_major_result<storage_type> cmr;
        xtensor_dynamic cma(cmr.m_shape, cmr.m_strides);
        compare_shape(cma, cmr);
    }

    TEST(xtensor, valued_constructor)
    {
        {
            SCOPED_TRACE("row_major valued constructor");
            row_major_result<storage_type> rm;
            int value = 2;
            xtensor_dynamic ra(rm.m_shape, value, layout_type::row_major);
            compare_shape(ra, rm);
            xtensor_dynamic::storage_type vec(ra.size(), value);
            EXPECT_EQ(ra.storage(), vec);
        }

        {
            SCOPED_TRACE("column_major valued constructor");
            column_major_result<storage_type> cm;
            int value = 2;
            xtensor_dynamic ca(cm.m_shape, value, layout_type::column_major);
            compare_shape(ca, cm);
            xtensor_dynamic::storage_type vec(ca.size(), value);
            EXPECT_EQ(ca.storage(), vec);
        }
    }

    TEST(xtensor, strided_valued_constructor)
    {
        central_major_result<storage_type> cmr;
        int value = 2;
        xtensor_dynamic cma(cmr.m_shape, cmr.m_strides, value);
        compare_shape(cma, cmr);
        xtensor_dynamic::storage_type vec(cma.size(), value);
        EXPECT_EQ(cma.storage(), vec);
    }

    TEST(xtensor, xscalar_constructor)
    {
        xscalar<int> xs(2);
        xtensor<int, 0> a(xs);
        EXPECT_EQ(a(), xs());
    }

    TEST(xtensor, copy_semantic)
    {
        central_major_result<storage_type> res;
        int value = 2;
        xtensor_dynamic a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("copy constructor");
            xtensor_dynamic b(a);
            compare_shape(a, b);
            EXPECT_EQ(a.storage(), b.storage());
        }

        {
            SCOPED_TRACE("assignment operator");
            row_major_result<storage_type> r;
            xtensor_dynamic c(r.m_shape, 0);
            EXPECT_NE(a.storage(), c.storage());
            c = a;
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
        }
    }

    TEST(xtensor, move_semantic)
    {
        central_major_result<storage_type> res;
        int value = 2;
        xtensor_dynamic a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("move constructor");
            xtensor_dynamic tmp(a);
            xtensor_dynamic b(std::move(tmp));
            compare_shape(a, b);
            EXPECT_EQ(a.storage(), b.storage());
        }

        {
            SCOPED_TRACE("move assignment");
            row_major_result<storage_type> r;
            xtensor_dynamic c(r.m_shape, 0);
            EXPECT_NE(a.storage(), c.storage());
            xtensor_dynamic tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
        }
    }

    TEST(xtensor, resize)
    {
        xtensor_dynamic a;
        test_resize<xtensor_dynamic, storage_type>(a);

#ifdef XTENSOR_ENABLE_ASSERT
        xtensor_dynamic b;
        std::vector<size_t> s = { 2u, 2u };
        xtensor_dynamic::strides_type strides = {2u, 1u};
        EXPECT_THROW(b.resize(s), std::runtime_error);
        EXPECT_THROW(b.resize(s, layout_type::dynamic), std::runtime_error);
        EXPECT_THROW(b.resize(s, strides), std::runtime_error);
#endif
    }

    TEST(xtensor, reshape)
    {
        xtensor_dynamic a;
        test_reshape<xtensor_dynamic, storage_type>(a);

        xtensor<int, 1> b;
        test_throwing_reshape(b);
    }

    TEST(xtensor, transpose)
    {
        xtensor_dynamic a;
        test_transpose<xtensor_dynamic, storage_type>(a);
    }

#if !(defined(XTENSOR_ENABLE_ASSERT) && defined(XTENSOR_DISABLE_EXCEPTIONS))
    TEST(xtensor, access)
    {
        xtensor_dynamic a;
        test_access<xtensor_dynamic, storage_type>(a);
    }
#endif

    TEST(xtensor, unchecked)
    {
        xtensor_dynamic a;
        test_unchecked<xtensor_dynamic, storage_type>(a);
    }

    TEST(xtensor, at)
    {
        xtensor_dynamic a;
        test_at<xtensor_dynamic, storage_type>(a);
    }

#if !(defined(XTENSOR_ENABLE_ASSERT) && defined(XTENSOR_DISABLE_EXCEPTIONS))
    TEST(xtensor, element)
    {
        xtensor_dynamic a;
        test_element<xtensor_dynamic, storage_type>(a);
    }
#endif

    TEST(xtensor, indexed_access)
    {
        xtensor_dynamic a;
        test_indexed_access<xtensor_dynamic, storage_type>(a);
    }

    TEST(xtensor, broadcast_shape)
    {
        xtensor<int, 4> a;
        test_broadcast(a);
    }

    TEST(xtensor, iterator)
    {
        using xtensor_rm = xtensor<int, 3, layout_type::row_major>;
        using xtensor_cm = xtensor<int, 3, layout_type::column_major>;
        xtensor_rm arm;
        xtensor_cm acm;
        test_iterator<xtensor_rm, xtensor_cm, storage_type>(arm, acm);
    }

    TEST(xtensor, fill)
    {
        xtensor<int, 2> a;
        test_fill(a);
    }

    TEST(xtensor, zerod)
    {
        xtensor<int, 0> b;
        EXPECT_EQ(b(), 0);
    }

    TEST(xtensor, xiterator)
    {
        xtensor_dynamic a;
        test_xiterator<xtensor_dynamic, storage_type>(a);
    }

    TEST(xtensor, reverse_xiterator)
    {
        xtensor_dynamic a;
        test_reverse_xiterator<xtensor_dynamic, storage_type>(a);
    }

    TEST(xtensor, single_element)
    {
        xtensor<int, 1> a = {1};
        xtensor<int, 1> res = 2 * a;
        EXPECT_EQ(2, res(0));
        EXPECT_EQ(2, res(1));

        xtensor<int, 1> b = { 1, 2 };
        xt::xarray<double> ca = a * b;
    }

    TEST(xtensor, move_from_xarray)
    {
        xarray<double> a = {{{1,2,3}, {4,5,6}}, {{10, 10, 10}, {1,5,10}}};
        xarray<double> a1 = a;
        xtensor<double, 3> b(std::move(a1));
        EXPECT_EQ(a, b);
        EXPECT_TRUE(std::equal(a.strides().begin(), a.strides().end(), b.strides().begin()) && a.strides().size() == b.strides().size());
        EXPECT_TRUE(std::equal(a.backstrides().begin(), a.backstrides().end(), b.backstrides().begin()) && a.backstrides().size() == b.backstrides().size());
        EXPECT_EQ(a.layout(), b.layout());

        xarray<double> a2 = a;
        xtensor<double, 3> c;
        c = std::move(a2);

        EXPECT_EQ(a, c);
        EXPECT_TRUE(std::equal(a.strides().begin(), a.strides().end(), c.strides().begin()) && a.strides().size() == c.strides().size());
        EXPECT_TRUE(std::equal(a.backstrides().begin(), a.backstrides().end(), c.backstrides().begin()) && a.backstrides().size() == c.backstrides().size());
        EXPECT_EQ(a.layout(), c.layout());
    }

    TEST(xtensor, from_indices)
    {
        xt::xtensor<int, 2> a = {{1,0,0},{1,1,0},{1,1,1}};
        xt::xtensor<size_t, 2> jdx = {{0,0},{1,0},{1,1},{2,0},{2,1},{2,2}};
        xt::xtensor<size_t, 2> idx = xt::from_indices(xt::argwhere(xt::equal(a,1)));
        EXPECT_TRUE(idx==jdx);
    }

    TEST(xtensor, flatten_indices)
    {
        xt::xtensor<int, 1> a = {1,0,0,1,1,0,1,1,1};
        xt::xtensor<size_t, 1> jdx = {0,3,4,6,7,8};
        xt::xtensor<size_t, 1> idx = xt::flatten_indices(xt::argwhere(xt::equal(a,1)));
        EXPECT_TRUE(idx==jdx);
    }

    TEST(xtensor, ravel_indices)
    {
        xt::xtensor<int, 2> a = {{1,0,0},{1,1,0},{1,1,1}};
        xt::xtensor<size_t, 1> jdx = {0,3,4,6,7,8};
        xt::xtensor<size_t, 1> idx = xt::ravel_indices(xt::argwhere(xt::equal(a,1)), a.shape());
        EXPECT_TRUE(idx==jdx);
    }

    TEST(xtensor, ravel_indices_column_major)
    {
        xt::xtensor<int, 2, xt::layout_type::column_major> a = {{1,0,0},{1,1,0},{1,1,1}};
        xt::xtensor<size_t, 1> jdx = {0,1,4,2,5,8};
        xt::xtensor<size_t, 1> idx = xt::ravel_indices(xt::argwhere(xt::equal(a,1)), a.shape(), xt::layout_type::column_major);
        EXPECT_TRUE(idx==jdx);
    }

    TEST(xarray, operator_brace)
    {
        xt::xtensor<size_t,2> a = {{0,1,2}, {3,4,5}};
        xt::xtensor<size_t,2> b = {{0,1,2}, {30,40,50}};
        a(1,3) = 30;
        a(1,4) = 40;
        a(1,5) = 50;
        EXPECT_EQ(a, b);
        EXPECT_THROW(a(2, 0));
        EXPECT_THROW(a(0, 3));
    }

    TEST(xtensor, periodic)
    {
        xt::xtensor<size_t,2> a = {{0,1,2}, {3,4,5}};
        xt::xtensor<size_t,2> b = {{0,1,2}, {30,40,50}};
        a.periodic(-1,3) = 30;
        a.periodic(-1,4) = 40;
        a.periodic(-1,5) = 50;
        EXPECT_EQ(a, b);
    }

    TEST(xtensor, flat)
    {
        {
            xt::xtensor<size_t, 2, xt::layout_type::row_major> a = {{0,1,2}, {3,4,5}};
            xt::xtensor<size_t, 2, xt::layout_type::row_major> b = {{0,1,2}, {30,40,50}};
            a.flat(3) = 30;
            a.flat(4) = 40;
            a.flat(5) = 50;
            EXPECT_EQ(a, b);
            EXPECT_THROW(a.flat(6));
        }
        {
            xt::xtensor<size_t, 2, xt::layout_type::column_major> a = {{0,1,2}, {3,4,5}};
            xt::xtensor<size_t, 2, xt::layout_type::column_major> b = {{0,1,2}, {30,40,50}};
            a.flat(1) = 30;
            a.flat(3) = 40;
            a.flat(5) = 50;
            EXPECT_EQ(a, b);
            EXPECT_THROW(a.flat(6));
        }
    }

    TEST(xtensor, in_bounds)
    {
        xt::xtensor<size_t,2> a = {{0,1,2}, {3,4,5}};
        EXPECT_TRUE(a.in_bounds(0,0) == true);
        EXPECT_TRUE(a.in_bounds(2,0) == false);
    }

    TEST(xtensor, iterator_types)
    {
        using tensor_type = xtensor<int, 2>;
        test_iterator_types<tensor_type, int*, const int*>();
    }

    TEST(xtensor, type_traits)
    {
        using tensor_type = xt::xtensor<double, 3>;
        EXPECT_TRUE(std::is_constructible<tensor_type>::value);
        EXPECT_TRUE(std::is_constructible<tensor_type>::value);

        EXPECT_TRUE(std::is_default_constructible<tensor_type>::value);

        EXPECT_TRUE(std::is_copy_constructible<tensor_type>::value);

        EXPECT_TRUE(std::is_move_constructible<tensor_type>::value);
        EXPECT_TRUE(std::is_nothrow_move_constructible<tensor_type>::value);

        EXPECT_TRUE(std::is_copy_assignable<tensor_type>::value);

        EXPECT_TRUE(std::is_move_assignable<tensor_type>::value);
        EXPECT_TRUE(std::is_nothrow_move_assignable<tensor_type>::value);

        EXPECT_TRUE(std::is_destructible<tensor_type>::value);
        EXPECT_TRUE(std::is_nothrow_destructible<tensor_type>::value);
    }

    TEST(xtensor, bool_container)
    {
        xt::xtensor<int, 1> a{1, 0, 1, 0}, b{1, 1, 0, 0};

        xt::xtensor<bool, 1> c = a & b;
        EXPECT_TRUE(c(0));
        EXPECT_FALSE(c(1));
        EXPECT_FALSE(c(2));
        EXPECT_FALSE(c(3));

        xt::xtensor<bool, 1> d = a | b;
        EXPECT_TRUE(d(0));
        EXPECT_TRUE(d(1));
        EXPECT_TRUE(d(2));
        EXPECT_FALSE(d(3));
    }
}
