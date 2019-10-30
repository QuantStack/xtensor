/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xoptional_assembly.hpp"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

#include "test_common.hpp"
#include "test_common_macros.hpp"

namespace xt
{
    using array_type = xarray<int, layout_type::dynamic>;
    using flag_array_type = xarray<bool, layout_type::dynamic>;
    using adaptor_type = xoptional_assembly_adaptor<array_type&, flag_array_type&>;

    TEST(xoptional_assembly_adaptor, constructor)
    {
        array_type v = {{1, 2, 3}, {4, 5, 6}};
        flag_array_type hv = {{true, false, true}, {false, true, false}};
        adaptor_type a(v, hv);
        compare_shape(a, v);
    }

    TEST(xoptional_assembly_adaptor, copy_semantic)
    {
        array_type v = {{1, 2, 3}, {4, 5, 6}};
        flag_array_type hv = {{true, false, true}, {false, true, false}};
        adaptor_type a(v, hv);

        {
            SCOPED_TRACE("copy constructor");
            adaptor_type b(a);
            compare_shape(a, b);
            EXPECT_EQ(a.value().storage(), b.value().storage());
            EXPECT_EQ(a.has_value().storage(), b.has_value().storage());
        }

        {
            SCOPED_TRACE("assignment operator");
            array_type v2 = {{1, 2, 13}, {14, 15, 16}};
            flag_array_type hv2 = {{false, true, true}, {false, true, false}};
            adaptor_type c(v2, hv2);
            EXPECT_NE(a.value().storage(), c.value().storage());
            EXPECT_NE(a.has_value().storage(), c.has_value().storage());
            c = a;
            compare_shape(a, c);
            EXPECT_EQ(a.value().storage(), c.value().storage());
            EXPECT_EQ(a.has_value().storage(), c.has_value().storage());
        }
    }

    TEST(xoptional_assembly_adaptor, move_semantic)
    {
        array_type v = {{1, 2, 3}, {4, 5, 6}};
        flag_array_type hv = {{true, false, true}, {false, true, false}};
        adaptor_type a(v, hv);

        {
            SCOPED_TRACE("copy constructor");
            adaptor_type tmp(a);
            adaptor_type b(std::move(tmp));
            compare_shape(a, b);
            EXPECT_EQ(a.value().storage(), b.value().storage());
            EXPECT_EQ(a.has_value().storage(), b.has_value().storage());
        }

        {
            SCOPED_TRACE("assignment operator");
            array_type v2 = {{1, 2, 13}, {14, 15, 16}};
            flag_array_type hv2 = {{false, true, true}, {false, true, false}};
            adaptor_type c(v2, hv2);
            EXPECT_NE(a.value().storage(), c.value().storage());
            EXPECT_NE(a.has_value().storage(), c.has_value().storage());
            adaptor_type tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.value().storage(), c.value().storage());
            EXPECT_EQ(a.has_value().storage(), c.has_value().storage());
        }
    }

    TEST(xoptional_assembly_adaptor, resize)
    {
        array_type v = {{1, 2, 3}, {4, 5, 6}};
        flag_array_type hv = {{true, false, true}, {false, true, false}};
        adaptor_type a(v, hv);
        test_resize(a);
        compare_shape(a.value(), a.has_value());
    }

    TEST(xoptional_assembly_adaptor, reshape)
    {
        array_type v = {{1, 2, 3}, {4, 5, 6}};
        flag_array_type hv = {{true, false, true}, {false, true, false}};
        adaptor_type a(v, hv);
        test_reshape(a);
        compare_shape(a.value(), a.has_value());
    }

    TEST(xoptional_assembly_adaptor, access)
    {
        using opt = xtl::xoptional<int>;
        array_type v = {{1, 2, 3}, {4, 5, 6}};
        flag_array_type hv = {{true, false, true}, {false, true, false}};
        adaptor_type a(v, hv);
        EXPECT_EQ(a(0, 0), opt(1, true));
        EXPECT_EQ(a(0, 1), opt(2, false));
        EXPECT_EQ(a(0, 2), opt(3, true));
        EXPECT_EQ(a(1, 0), opt(4, false));
        EXPECT_EQ(a(1, 1), opt(5, true));
        EXPECT_EQ(a(1, 2), opt(6, false));
    }

    TEST(xoptional_assembly_adaptor, at)
    {
        using opt = xtl::xoptional<int>;
        array_type v = {{1, 2, 3}, {4, 5, 6}};
        flag_array_type hv = {{true, false, true}, {false, true, false}};
        adaptor_type a(v, hv);
        EXPECT_EQ(a.at(0, 0), opt(1, true));
        EXPECT_EQ(a.at(0, 1), opt(2, false));
        EXPECT_EQ(a.at(0, 2), opt(3, true));
        EXPECT_EQ(a.at(1, 0), opt(4, false));
        EXPECT_EQ(a.at(1, 1), opt(5, true));
        EXPECT_EQ(a.at(1, 2), opt(6, false));
    }

    TEST(xoptional_assembly_adaptor, element)
    {
        using opt = xtl::xoptional<int>;
        array_type v = {{1, 2, 3}, {4, 5, 6}};
        flag_array_type hv = {{true, false, true}, {false, true, false}};
        adaptor_type a(v, hv);
        std::vector<std::size_t> v00({0, 0}), v01({0, 1}), v02({0, 2}),
            v10({1, 0}), v11({1, 1}), v12({1, 2});

        EXPECT_EQ(a.element(v00.begin(), v00.end()), opt(1, true));
        EXPECT_EQ(a.element(v01.begin(), v01.end()), opt(2, false));
        EXPECT_EQ(a.element(v02.begin(), v02.end()), opt(3, true));
        EXPECT_EQ(a.element(v10.begin(), v10.end()), opt(4, false));
        EXPECT_EQ(a.element(v11.begin(), v11.end()), opt(5, true));
        EXPECT_EQ(a.element(v12.begin(), v12.end()), opt(6, false));
    }

    TEST(xoptional_assembly_adaptor, indexed_access)
    {
        using opt = xtl::xoptional<int>;
        array_type v = {{1, 2, 3}, {4, 5, 6}};
        flag_array_type hv = {{true, false, true}, {false, true, false}};
        adaptor_type a(v, hv);
        xindex i00({0, 0}), i01({0, 1}), i02({0, 2}),
            i10({1, 0}), i11({1, 1}), i12({1, 2});

        EXPECT_EQ(a[i00], opt(1, true));
        EXPECT_EQ((a[{0, 0}]), opt(1, true));
        EXPECT_EQ(a[i01], opt(2, false));
        EXPECT_EQ((a[{0, 1}]), opt(2, false));
        EXPECT_EQ(a[i02], opt(3, true));
        EXPECT_EQ((a[{0, 2}]), opt(3, true));
        EXPECT_EQ(a[i10], opt(4, false));
        EXPECT_EQ((a[{1, 0}]), opt(4, false));
        EXPECT_EQ(a[i11], opt(5, true));
        EXPECT_EQ((a[{1, 1}]), opt(5, true));
        EXPECT_EQ(a[i12], opt(6, false));
        EXPECT_EQ((a[{1, 2}]), opt(6, false));
    }

    TEST(xoptional_assembly_adaptor, broadcast_shape)
    {
        using shape_type = adaptor_type::shape_type;
        shape_type s = {3, 1, 4, 2};
        array_type v(s);
        flag_array_type hv(s);
        adaptor_type a(v, hv);

        {
            SCOPED_TRACE("same shape");
            shape_type s1 = s;
            bool res = a.broadcast_shape(s1);
            EXPECT_EQ(s1, s);
            EXPECT_TRUE(res);
        }

        {
            SCOPED_TRACE("different shape");
            shape_type s2 = {3, 5, 1, 2};
            shape_type s2r = {3, 5, 4, 2};
            bool res = a.broadcast_shape(s2);
            EXPECT_EQ(s2, s2r);
            EXPECT_FALSE(res);
        }

        {
            SCOPED_TRACE("incompatible shapes");
            shape_type s4 = {2, 1, 3, 2};
            EXPECT_THROW(a.broadcast_shape(s4), broadcast_error);
        }

        {
            shape_type s2 = {3, 1, 4, 2};
            a.resize(s2);
            SCOPED_TRACE("different dimensions");
            shape_type s3 = {5, 3, 1, 4, 2};
            shape_type s3r = s3;
            bool res = a.broadcast_shape(s3);
            EXPECT_EQ(s3, s3r);
            EXPECT_FALSE(res);
        }
    }

    TEST(xoptional_assembly_adaptor, iterator)
    {
        using opt = xtl::xoptional<int>;
        std::vector<opt> vec = {opt(1), opt(2, false), opt(3, false), opt(4)};

        {
            SCOPED_TRACE("row_major storage iterator");
            xarray<int, layout_type::row_major> v;
            xarray<bool, layout_type::row_major> hv;
            xoptional_assembly_adaptor<decltype(v)&, decltype(hv)&> rma(v, hv);
            rma.resize({2, 2});
            std::copy(vec.cbegin(), vec.cend(), rma.begin<layout_type::row_major>());
            EXPECT_EQ(vec[0], rma(0, 0));
            EXPECT_EQ(vec[1], rma(0, 1));
            EXPECT_EQ(vec[2], rma(1, 0));
            EXPECT_EQ(vec[3], rma(1, 1));
            EXPECT_EQ(vec.size(), std::size_t(std::distance(rma.begin<layout_type::row_major>(), rma.end<layout_type::row_major>())));
        }

        {
            SCOPED_TRACE("column_major storage iterator");
            xarray<int, layout_type::row_major> v;
            xarray<bool, layout_type::row_major> hv;
            xoptional_assembly_adaptor<decltype(v)&, decltype(hv)&> cma(v, hv);
            cma.resize({2, 2});
            std::copy(vec.cbegin(), vec.cend(), cma.begin<layout_type::column_major>());
            EXPECT_EQ(vec[0], cma(0, 0));
            EXPECT_EQ(vec[1], cma(1, 0));
            EXPECT_EQ(vec[2], cma(0, 1));
            EXPECT_EQ(vec[3], cma(1, 1));
            EXPECT_EQ(vec.size(), std::size_t(std::distance(cma.begin<layout_type::column_major>(), cma.end<layout_type::column_major>())));
        }
    }

    TEST(xoptional_assembly_adaptor, xiterator)
    {
        row_major_result<> rm;
        array_type a;
        a.resize(rm.m_shape, layout_type::row_major);
        a.fill(0);
        a(1, 1, 0) = rm.m_assigner[1][1][0];
        a[0] = 4;
        flag_array_type fa(rm.m_shape, true);
        size_t nb_iter = a.size() / 2;
        using shape_type = std::vector<size_t>;
        adaptor_type vec(a, fa);

        // broadcast_iterator
        {
            auto iter = vec.begin<layout_type::row_major>();
            auto iter_end = vec.end<layout_type::row_major>();
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(vec.value().storage()[nb_iter], *iter);
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }

        // shaped_xiterator
        {
            shape_type shape(rm.m_shape.size() + 1);
            std::copy(rm.m_shape.begin(), rm.m_shape.end(), shape.begin() + 1);
            shape[0] = 2;
            auto iter = vec.begin<layout_type::row_major>(shape);
            auto iter_end = vec.end<layout_type::row_major>(shape);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(vec(0, 0), *iter);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }

        // column broadcast_iterator
        {
            auto iter = vec.begin<layout_type::column_major>();
            auto iter_end = vec.end<layout_type::column_major>();
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(vec(0, 0, 2), *iter);
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }

        // column shaped_xiterator
        {
            shape_type shape(rm.m_shape.size() + 1);
            std::copy(rm.m_shape.begin(), rm.m_shape.end(), shape.begin() + 1);
            shape[0] = 2;
            auto iter = vec.begin<layout_type::column_major>(shape);
            auto iter_end = vec.end<layout_type::column_major>(shape);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(vec(0, 0, 2), *iter);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }
    }

    TEST(xoptional_assembly_adaptor, reverse_xiterator)
    {
        row_major_result<> rm;
        array_type a;
        a.resize(rm.m_shape, layout_type::row_major);
        a(1, 0, 3) = rm.m_assigner[1][0][3];
        a(2, 1, 3) = 2;
        flag_array_type fa(rm.m_shape, true);
        size_t nb_iter = a.size() / 2;
        using shape_type = std::vector<size_t>;
        adaptor_type vec(a, fa);

        // broadcast_iterator
        {
            auto iter = vec.rbegin<layout_type::row_major>();
            auto iter_end = vec.rend<layout_type::row_major>();
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(vec.value().storage()[nb_iter - 1], *iter);
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }

        // shaped_xiterator
        {
            shape_type shape(rm.m_shape.size() + 1);
            std::copy(rm.m_shape.begin(), rm.m_shape.end(), shape.begin() + 1);
            shape[0] = 2;
            auto iter = vec.rbegin<layout_type::row_major>(shape);
            auto iter_end = vec.rend<layout_type::row_major>(shape);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(vec.value().storage()[2 * nb_iter - 1], *iter);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }
    }

    TEST(xoptional_assembly_adaptor, semantic)
    {
        array_type v = {{1, 2}, {3, 4}};
        flag_array_type hv = {{true, false}, {false, true}};

        adaptor_type a(v, hv);
        adaptor_type b(a);

        array_type vres;
        flag_array_type hvres;
        adaptor_type res(vres, hvres);

        res = a + b;
        EXPECT_EQ(res(0, 0), a(0, 0) + b(0, 0));
        EXPECT_EQ(res(0, 1), a(0, 1) + b(0, 1));
        EXPECT_EQ(res(1, 0), a(1, 0) + b(1, 0));
        EXPECT_EQ(res(1, 1), a(1, 1) + b(1, 1));

        res = a;
        res += b;
        EXPECT_EQ(res(0, 0), a(0, 0) + b(0, 0));
        EXPECT_EQ(res(0, 1), a(0, 1) + b(0, 1));
        EXPECT_EQ(res(1, 0), a(1, 0) + b(1, 0));
        EXPECT_EQ(res(1, 1), a(1, 1) + b(1, 1));
    }
}
