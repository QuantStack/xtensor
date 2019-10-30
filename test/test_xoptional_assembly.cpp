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
    using array_type = xarray<int>;
    using flag_array_type = xarray<bool>;
    using opt_ass_type = xoptional_assembly<array_type, flag_array_type>;
    using cm_opt_ass_type = xoptional_assembly<xarray<int, layout_type::column_major>, xarray<bool, layout_type::column_major>>;
    using dyn_opt_ass_type = xoptional_assembly<xarray<int, layout_type::dynamic>, xarray<bool, layout_type::dynamic>>;

    TEST(xoptional_assembly, shaped_constructor)
    {
        {
            SCOPED_TRACE("row_major constructor");
            row_major_result<> rm;
            dyn_opt_ass_type ra(rm.m_shape, layout_type::row_major);
            compare_shape(ra, rm);
        }

        {
            SCOPED_TRACE("column_major constructor");
            column_major_result<> cm;
            cm_opt_ass_type ca(cm.m_shape);
            compare_shape(ca, cm);
        }

        {
            SCOPED_TRACE("from shape");
            std::array<std::size_t, 3> shp = {5, 4, 2};
            std::vector<std::size_t> shp_as_vec = {5, 4, 2};
            auto ca = cm_opt_ass_type::from_shape({3, 2, 1});
            auto cb = cm_opt_ass_type::from_shape(shp);
            std::vector<std::size_t> expected_shape = {3, 2, 1};
            EXPECT_EQ(expected_shape, ca.shape());
            EXPECT_EQ(shp_as_vec, cb.shape());
        }
    }

    TEST(xoptional_assembly, strided_constructor)
    {
        central_major_result<> cmr;
        dyn_opt_ass_type cma(cmr.m_shape, cmr.m_strides);
        compare_shape(cma, cmr);
    }

    TEST(xoptional_assembly, valued_constructor)
    {
        {
            SCOPED_TRACE("row_major valued constructor");
            row_major_result<> rm;
            int value = 2;
            dyn_opt_ass_type ra(rm.m_shape, value, layout_type::row_major);
            compare_shape(ra, rm);
            dyn_opt_ass_type::raw_value_expression::storage_type vec(ra.size(), value);
            EXPECT_EQ(ra.value().storage(), vec);
        }

        {
            SCOPED_TRACE("column_major valued constructor");
            column_major_result<> cm;
            int value = 2;
            cm_opt_ass_type ca(cm.m_shape, value);
            compare_shape(ca, cm);
            cm_opt_ass_type::raw_value_expression::storage_type vec(ca.size(), value);
            EXPECT_EQ(ca.value().storage(), vec);
        }
    }

    TEST(xoptional_assembly, strided_valued_constructor)
    {
        central_major_result<> cmr;
        int value = 2;
        dyn_opt_ass_type cma(cmr.m_shape, cmr.m_strides, value);
        compare_shape(cma, cmr);
        dyn_opt_ass_type::raw_value_expression::storage_type vec(cma.size(), value);
        EXPECT_EQ(cma.value().storage(), vec);
    }

    TEST(xoptional_assembly, xscalar_constructor)
    {
        xscalar<int> xs(2);
        opt_ass_type a(xs);
        EXPECT_EQ(a(), xs());
    }

    TEST(xoptional_assembly, initializer_list)
    {
        using opt = xtl::xoptional<int>;
        opt_ass_type a0(opt(1));
        opt_ass_type a1({opt(1), opt(2, false)});
        opt_ass_type a2({{opt(1, true), opt(2, false)}, {opt(2), opt(4, true)}, {opt(5), opt(6)}});
        EXPECT_EQ(opt(1, true), a0());
        EXPECT_EQ(opt(2, false), a1(1));
        EXPECT_EQ(opt(4), a2(1, 1));
    }

    TEST(xoptional_assembly, expression_constructor)
    {
        using opt = xtl::xoptional<int>;
        array_type value = {{1, 2}, {3, 4}};
        flag_array_type flag = {{true, false}, {false, true}};

        opt_ass_type a(std::move(value), flag);
        EXPECT_EQ(a(0, 0), opt(1, true));
        EXPECT_EQ(a(0, 1), opt(2, false));
        EXPECT_EQ(a(1, 0), opt(3, false));
        EXPECT_EQ(a(1, 1), opt(4, true));

        opt_ass_type a2(a.value());
        EXPECT_EQ(a2(0, 0), opt(1, true));
        EXPECT_EQ(a2(0, 1), opt(2, true));
        EXPECT_EQ(a2(1, 0), opt(3, true));
        EXPECT_EQ(a2(1, 1), opt(4, true));

        array_type value2 = { {1, 2}, {3, 4} };
        opt_ass_type a3(std::move(value2));
        EXPECT_EQ(a3(0, 0), opt(1, true));
        EXPECT_EQ(a3(0, 1), opt(2, true));
        EXPECT_EQ(a3(1, 0), opt(3, true));
        EXPECT_EQ(a3(1, 1), opt(4, true));
    }

    TEST(xoptional_assembly, copy_semantic)
    {
        central_major_result<> res;
        int value = 2;
        dyn_opt_ass_type a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("copy constructor");
            dyn_opt_ass_type b(a);
            compare_shape(a, b);
            EXPECT_EQ(a.storage(), b.storage());
        }

        {
            SCOPED_TRACE("assignment operator");
            row_major_result<> r;
            dyn_opt_ass_type c(r.m_shape, dyn_opt_ass_type::value_type(0, false));
            EXPECT_NE(a.storage(), c.storage());
            c = a;
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
        }
    }

    TEST(xoptional_assembly, move_semantic)
    {
        central_major_result<> res;
        int value = 2;
        dyn_opt_ass_type a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("move constructor");
            dyn_opt_ass_type tmp(a);
            dyn_opt_ass_type b(std::move(tmp));
            compare_shape(a, b);
            EXPECT_EQ(a.storage(), b.storage());
        }

        {
            SCOPED_TRACE("move assignment");
            row_major_result<> r;
            dyn_opt_ass_type c(r.m_shape, dyn_opt_ass_type::value_type(0, false));
            EXPECT_NE(a.value().storage(), c.value().storage());
            EXPECT_NE(a.has_value().storage(), c.has_value().storage());
            dyn_opt_ass_type tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
        }
    }

    TEST(xoptional_assembly, resize)
    {
        dyn_opt_ass_type a;
        test_resize(a);
    }

    TEST(xoptional_assembly, reshape)
    {
        dyn_opt_ass_type a;
        test_reshape(a);
    }

    TEST(xoptional_assembly, access)
    {
        using opt = xtl::xoptional<int>;
        opt_ass_type a = {{opt(1), opt(2, false)}, {opt(3, false), opt(4)}};
        EXPECT_EQ(a(0, 0), opt(1, true));
        EXPECT_EQ(a(0, 1), opt(2, false));
        EXPECT_EQ(a(1, 0), opt(3, false));
        EXPECT_EQ(a(1, 1), opt(4, true));
    }

    TEST(xoptional_assembly, fill)
    {
        using opt = xtl::xoptional<int>;
        opt_ass_type a = {{opt(1), opt(2, false)}, {opt(3, false), opt(4)}};

        a.fill(opt(5, false));
        EXPECT_EQ(a(0, 0), opt(5, false));
        EXPECT_EQ(a(0, 1), opt(5, false));
        EXPECT_EQ(a(1, 0), opt(5, false));
        EXPECT_EQ(a(1, 1), opt(5, false));

        a.fill(3);
        EXPECT_EQ(a(0, 0), opt(3, true));
        EXPECT_EQ(a(0, 1), opt(3, true));
        EXPECT_EQ(a(1, 0), opt(3, true));
        EXPECT_EQ(a(1, 1), opt(3, true));
    }

    TEST(xoptional_assembly, unchecked)
    {
        using opt = xtl::xoptional<int>;
        opt_ass_type a = { { opt(1), opt(2, false) },{ opt(3, false), opt(4) } };
        EXPECT_EQ(a.unchecked(0, 0), opt(1, true));
        EXPECT_EQ(a.unchecked(0, 1), opt(2, false));
        EXPECT_EQ(a.unchecked(1, 0), opt(3, false));
        EXPECT_EQ(a.unchecked(1, 1), opt(4, true));
    }

    TEST(xoptional_assembly, at)
    {
        using opt = xtl::xoptional<int>;
        opt_ass_type a = {{opt(1), opt(2, false)}, {opt(3, false), opt(4)}};
        EXPECT_EQ(a.at(0, 0), opt(1, true));
        EXPECT_EQ(a.at(0, 1), opt(2, false));
        EXPECT_EQ(a.at(1, 0), opt(3, false));
        EXPECT_EQ(a.at(1, 1), opt(4, true));
        EXPECT_ANY_THROW(a.at(2, 2));
    }

    TEST(xoptional_assembly, element)
    {
        using opt = xtl::xoptional<int>;
        opt_ass_type a = {{opt(1), opt(2, false)}, {opt(3, false), opt(4)}};
        std::vector<std::size_t> v0({0, 0}), v1({0, 1}), v2({1, 0}), v3({1, 1});
        EXPECT_EQ(a.element(v0.begin(), v0.end()), opt(1, true));
        EXPECT_EQ(a.element(v1.begin(), v1.end()), opt(2, false));
        EXPECT_EQ(a.element(v2.begin(), v2.end()), opt(3, false));
        EXPECT_EQ(a.element(v3.begin(), v3.end()), opt(4, true));
    }

    TEST(xoptional_assembly, indexed_access)
    {
        using opt = xtl::xoptional<int>;
        opt_ass_type a = {{opt(1), opt(2, false)}, {opt(3, false), opt(4)}};
        xindex i0({0, 0}), i1({0, 1}), i2({1, 0}), i3({1, 1});
        EXPECT_EQ(a[i0], opt(1, true));
        EXPECT_EQ((a[{0, 0}]), opt(1, true));
        EXPECT_EQ(a[i1], opt(2, false));
        EXPECT_EQ((a[{0, 1}]), opt(2, false));
        EXPECT_EQ(a[i2], opt(3, false));
        EXPECT_EQ((a[{1, 0}]), opt(3, false));
        EXPECT_EQ(a[i3], opt(4, true));
        EXPECT_EQ((a[{1, 1}]), opt(4, true));
    }

    TEST(xoptional_assembly, broadcast_shape)
    {
        using shape_type = typename opt_ass_type::shape_type;

        shape_type s = {3, 1, 4, 2};
        opt_ass_type vec(s);

        {
            SCOPED_TRACE("same shape");
            shape_type s1 = s;
            bool res = vec.broadcast_shape(s1);
            EXPECT_EQ(s1, s);
            EXPECT_TRUE(res);
        }

        {
            SCOPED_TRACE("different shape");
            shape_type s2 = {3, 5, 1, 2};
            shape_type s2r = {3, 5, 4, 2};
            bool res = vec.broadcast_shape(s2);
            EXPECT_EQ(s2, s2r);
            EXPECT_FALSE(res);
        }

        {
            SCOPED_TRACE("incompatible shapes");
            shape_type s4 = {2, 1, 3, 2};
            EXPECT_THROW(vec.broadcast_shape(s4), broadcast_error);
        }

        {
            shape_type s2 = {3, 1, 4, 2};
            vec.resize(s2);
            SCOPED_TRACE("different dimensions");
            shape_type s3 = {5, 3, 1, 4, 2};
            shape_type s3r = s3;
            bool res = vec.broadcast_shape(s3);
            EXPECT_EQ(s3, s3r);
            EXPECT_FALSE(res);
        }
    }

    TEST(xoptional_assembly, iterator)
    {
        using opt = xtl::xoptional<int>;
        std::vector<opt> vec = {opt(1), opt(2, false), opt(3, false), opt(4)};

        {
            SCOPED_TRACE("row_major storage iterator");
            opt_ass_type rma(opt_ass_type::shape_type({2, 2}));
            std::copy(vec.cbegin(), vec.cend(), rma.begin<layout_type::row_major>());
            EXPECT_EQ(vec[0], rma(0, 0));
            EXPECT_EQ(vec[1], rma(0, 1));
            EXPECT_EQ(vec[2], rma(1, 0));
            EXPECT_EQ(vec[3], rma(1, 1));
            EXPECT_EQ(vec.size(), std::size_t(std::distance(rma.begin<layout_type::row_major>(), rma.end<layout_type::row_major>())));
        }

        {
            SCOPED_TRACE("column_major storage iterator");
            cm_opt_ass_type cma(opt_ass_type::shape_type({2, 2}));
            std::copy(vec.cbegin(), vec.cend(), cma.begin<layout_type::column_major>());
            EXPECT_EQ(vec[0], cma(0, 0));
            EXPECT_EQ(vec[1], cma(1, 0));
            EXPECT_EQ(vec[2], cma(0, 1));
            EXPECT_EQ(vec[3], cma(1, 1));
            EXPECT_EQ(vec.size(), std::size_t(std::distance(cma.begin<layout_type::column_major>(), cma.end<layout_type::column_major>())));
        }
    }

    TEST(xoptional_assembly, xiterator)
    {
        row_major_result<> rm;
        dyn_opt_ass_type vec;
        vec.resize(rm.m_shape, layout_type::row_major);
        vec.fill(123);
        vec(1, 1, 0) = rm.m_assigner[1][1][0];
        vec.value()[0] = 4;
        size_t nb_iter = vec.size() / 2;
        using shape_type = std::vector<size_t>;

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

    TEST(xoptional_assembly, reverse_xiterator)
    {
        row_major_result<> rm;
        dyn_opt_ass_type vec;
        vec.resize(rm.m_shape, layout_type::row_major);
        vec(1, 0, 3) = rm.m_assigner[1][0][3];
        vec(2, 1, 3) = 2;
        size_t nb_iter = vec.size() / 2;
        using shape_type = std::vector<size_t>;

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

    TEST(xoptional_assembly, semantic)
    {
        using opt = xtl::xoptional<int>;
        dyn_opt_ass_type a = {{opt(1), opt(2, false)}, {opt(3, false), opt(4)}};
        dyn_opt_ass_type b(a);

        dyn_opt_ass_type c = a + b;
        EXPECT_EQ(c(0, 0), a(0, 0) + b(0, 0));
        EXPECT_EQ(c(0, 1), a(0, 1) + b(0, 1));
        EXPECT_EQ(c(1, 0), a(1, 0) + b(1, 0));
        EXPECT_EQ(c(1, 1), a(1, 1) + b(1, 1));

        dyn_opt_ass_type d(a);
        d += b;
        EXPECT_EQ(d(0, 0), a(0, 0) + b(0, 0));
        EXPECT_EQ(d(0, 1), a(0, 1) + b(0, 1));
        EXPECT_EQ(d(1, 0), a(1, 0) + b(1, 0));
        EXPECT_EQ(d(1, 1), a(1, 1) + b(1, 1));
    }

    TEST(xoptional_assembly, mixed_semantic)
    {
        using d_opt_ass_type = xoptional_assembly<xarray<double, layout_type::row_major>, xarray<bool, layout_type::row_major>>;
        using opt = xtl::xoptional<double>;
        d_opt_ass_type a = {{opt(1.), opt(2., false), opt(3., false), opt(4.)},
                            {opt(5., false), opt(6.), opt(7.), opt(8., false)}};
        xarray_optional<double> b = {{opt(1.), opt(2.), opt(3., false), opt(4., false)},
                                     {opt(5., false), opt(6.), opt(7.), opt(8.)}};

        d_opt_ass_type res = a + b;
        EXPECT_EQ(res(0, 0), opt(2.));
        EXPECT_EQ(res(0, 1), opt(4., false));
        EXPECT_EQ(res(0, 2), opt(6., false));
        EXPECT_EQ(res(0, 3), opt(8., false));
        EXPECT_EQ(res(1, 0), opt(10., false));
        EXPECT_EQ(res(1, 1), opt(12.));
        EXPECT_EQ(res(1, 2), opt(14.));
        EXPECT_EQ(res(1, 3), opt(16., false));
    }

    TEST(xoptional_assembly, mixed_expression)
    {
        using opt = xtl::xoptional<int>;
        dyn_opt_ass_type a = { { opt(1), opt(2, false), opt(3, false), opt(4) },
                               { opt(5, false), opt(6), opt(7), opt(8, false) } };
        xarray<int> b = { { 1, 2, 3, 4}, { 5, 6, 7, 8} };

        dyn_opt_ass_type c = a + b;
        dyn_opt_ass_type res = { { opt(2), opt(4, false), opt(6, false), opt(8) },
                                 { opt(10, false), opt(12), opt(14), opt(16, false) } };
        EXPECT_EQ(res, c);

        dyn_opt_ass_type d = 2 * a;
        EXPECT_EQ(res, d);

        opt e = opt(2, true);
        dyn_opt_ass_type f = e * a;
        EXPECT_EQ(res, f);

        dyn_opt_ass_type g = opt(2, true) * a;
        EXPECT_EQ(res, f);
    }
}
