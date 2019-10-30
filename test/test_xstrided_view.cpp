/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xstrided_view.hpp"

#include "gtest/gtest.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xview.hpp"
#include "test_common_macros.hpp"


namespace xt
{
    using std::size_t;
    using shape_t = std::vector<std::size_t>;
    using view_shape_type = dynamic_shape<size_t>;

    bool operator==(const view_shape_type& lhs, const dynamic_shape<ptrdiff_t>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    bool operator==(const dynamic_shape<ptrdiff_t>& lhs, const dynamic_shape<size_t>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    TEST(xstrided_view, simple)
    {
        view_shape_type shape = { 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.begin());

        auto view1 = strided_view(a, xstrided_slice_vector({ 1, range(1, 4) }));
        EXPECT_EQ(a(1, 1), view1(0));
        EXPECT_EQ(a(1, 2), view1(1));
        EXPECT_EQ(size_t(1), view1.dimension());
        EXPECT_EQ(view1(), view1(0));
        EXPECT_EQ(view1(1, 0), view1(0));
        EXPECT_EQ(view1.unchecked(0), view1(0));

        auto view0 = strided_view(a, xstrided_slice_vector({ 0, range(0, 3) }));
        EXPECT_EQ(a(0, 0), view0(0));
        EXPECT_EQ(a(0, 1), view0(1));
        EXPECT_EQ(size_t(1), view0.dimension());
        EXPECT_EQ(size_t(3), view0.shape()[0]);

        auto view2 = strided_view(a, xstrided_slice_vector({ range(0, 2), 2 }));
        EXPECT_EQ(a(0, 2), view2(0));
        EXPECT_EQ(a(1, 2), view2(1));
        EXPECT_EQ(size_t(1), view2.dimension());
        EXPECT_EQ(size_t(2), view2.shape()[0]);

        auto view4 = strided_view(a, { 1 });
        EXPECT_EQ(size_t(1), view4.dimension());
        EXPECT_EQ(size_t(4), view4.shape()[0]);

        auto view5 = strided_view(view4, { 1 });
        EXPECT_EQ(size_t(0), view5.dimension());
        EXPECT_EQ(size_t(0), view5.shape().size());

        auto view6 = strided_view(a, xstrided_slice_vector({ 1, all() }));
        EXPECT_EQ(a(1, 0), view6(0));
        EXPECT_EQ(a(1, 1), view6(1));
        EXPECT_EQ(a(1, 2), view6(2));
        EXPECT_EQ(a(1, 3), view6(3));

        auto view7 = strided_view(a, xstrided_slice_vector({ all(), 2 }));
        EXPECT_EQ(a(0, 2), view7(0));
        EXPECT_EQ(a(1, 2), view7(1));
        EXPECT_EQ(a(2, 2), view7(2));

        EXPECT_THROW(strided_view(a, { all(), all(), 1 }), std::runtime_error);
        EXPECT_THROW(strided_view(a, { all(), all(), all() }), std::runtime_error);
        EXPECT_NO_THROW(strided_view(a, { all(), newaxis(), all() }));
        EXPECT_NO_THROW(strided_view(a, { 3, newaxis(), 1 }));
        EXPECT_NO_THROW(strided_view(a, { 3, 1, newaxis() }));
    }

    TEST(xstrided_view, assign)
    {
        xt::xarray<double> arr = { 5., 6. };
        xt::strided_view(arr, { 0 }) = xt::strided_view(arr, { 1 });
    }

    TEST(xstrided_view, three_dimensional)
    {
        view_shape_type shape = { 3, 4, 2 };
        std::vector<double> data = {
            1, 2,
            3, 4,
            5, 6,
            7, 8,

            9, 10,
            11, 12,
            21, 22,
            23, 24,

            25, 26,
            27, 28,
            29, 210,
            211, 212
        };

        xarray<double> a(shape);
        std::copy(data.cbegin(), data.cend(), a.begin());

        auto view1 = strided_view(a, { 1 });
        EXPECT_EQ(size_t(2), view1.dimension());
        view_shape_type expected_shape = { 4, 2 };
        EXPECT_EQ(expected_shape, view1.shape());
        EXPECT_EQ(a(1, 0, 0), view1(0, 0));
        EXPECT_EQ(a(1, 0, 1), view1(0, 1));
        EXPECT_EQ(a(1, 1, 0), view1(1, 0));
        EXPECT_EQ(a(1, 1, 1), view1(1, 1));

        std::array<std::size_t, 2> idx = { 1, 1 };
        EXPECT_EQ(a(1, 1, 1), view1.element(idx.cbegin(), idx.cend()));
    }

    TEST(xstrided_view, iterator)
    {
        view_shape_type shape = { 2, 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data{ 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = strided_view(a, xstrided_slice_vector({ range(0, 2), 1, range(1, 4) }));
        auto iter = view1.template begin<layout_type::row_major>();
        auto iter_end = view1.template end<layout_type::row_major>();

        EXPECT_EQ(6, *iter);
        ++iter;
        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(8, *iter);
        ++iter;
        EXPECT_EQ(18, *iter);
        ++iter;
        EXPECT_EQ(19, *iter);
        ++iter;
        EXPECT_EQ(20, *iter);
        ++iter;
        EXPECT_EQ(iter, iter_end);
        EXPECT_FALSE(iter < iter_end);

        auto view2 = strided_view(view1, xstrided_slice_vector({ range(0, 2), range(1, 3) }));
        auto iter2 = view2.template begin<layout_type::row_major>();
        auto iter_end2 = view2.template end<layout_type::row_major>();

        EXPECT_EQ(7, *iter2);
        ++iter2;
        EXPECT_EQ(8, *iter2);
        ++iter2;
        EXPECT_EQ(19, *iter2);
        ++iter2;
        EXPECT_EQ(20, *iter2);
        ++iter2;
        EXPECT_EQ(iter2, iter_end2);
        EXPECT_FALSE(iter2 < iter_end2);
    }

    TEST(xstrided_view, fill)
    {
        view_shape_type shape = { 2, 3, 4 };
        xarray<double> a(shape), res(shape);
        std::vector<double> data{ 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());
        std::vector<double> data_res = { 1, 2, 3, 4, 5, 4, 4, 4, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 4, 4, 4, 21, 22, 23, 24 };
        std::copy(data_res.cbegin(), data_res.cend(), res.template begin<layout_type::row_major>());
        auto view1 = strided_view(a, xstrided_slice_vector({ range(0, 2), 1, range(1, 4) }));
        view1.fill(4);
        EXPECT_EQ(a, res);
    }

    TEST(xstrided_view, xstrided_view_on_xfunction)
    {
        xarray<int> a = {{ 1,  2,  3,  4 },
                         { 5,  6,  7,  8 },
                         { 9, 10, 11, 12 } };
        xarray<int> b = { 1, 2, 3, 4 };

        auto sum = a + b;
        auto func = strided_view(sum, xstrided_slice_vector({ 1, range(1, 4) }));

        auto iter = func.template begin<layout_type::row_major>();
        auto iter_end = func.template end<layout_type::row_major>();

        EXPECT_EQ(8, *iter);
        ++iter;
        EXPECT_EQ(10, *iter);
        ++iter;
        EXPECT_EQ(12, *iter);
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xstrided_view, view_on_generator)
    {
        auto vgen = strided_view(eye(4), xstrided_slice_vector({ 1, range(1, 4) }));
        auto iter = vgen.cbegin();
    }

    TEST(xstrided_view, xstrided_view_on_xtensor)
    {
        xtensor<int, 2> a({ 3, 4 });
        std::vector<int> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = strided_view(a, xstrided_slice_vector({ 1, range(1, 4) }));
        EXPECT_EQ(a(1, 1), view1(0));
        EXPECT_EQ(a(1, 2), view1(1));
        EXPECT_EQ(size_t(1), view1.dimension());

        auto iter = view1.begin();
        auto iter_end = view1.end();

        EXPECT_EQ(6, *iter);
        ++iter;
        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(8, *iter);

        xarray<int> b({ 3 }, 2);
        xtensor<int, 1> res = view1 + b;
        EXPECT_EQ(8, res(0));
        EXPECT_EQ(9, res(1));
        EXPECT_EQ(10, res(2));
    }

    TEST(xstrided_view, const_view)
    {
        const xtensor<double, 3> arr{ { 1, 2, 3 }, 2.5 };
        xtensor<double, 2> arr2{ { 2, 3 }, 0.0 };
        xtensor<double, 2> ref{ { 2, 3 }, 2.5 };
        arr2 = strided_view(arr, { 0 });
        EXPECT_EQ(ref, arr2);
        // check that the following compiles
        auto v = strided_view(arr, { 0 });
        double acc = v(0);
        EXPECT_EQ(acc, 2.5);
        auto iter = v.begin();
    }

    TEST(xstrided_view, newaxis)
    {
        view_shape_type shape = { 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.begin());

        auto view1 = strided_view(a, xstrided_slice_vector({ all(), newaxis(), all() }));
        EXPECT_EQ(a(1, 1), view1(1, 0, 1));
        EXPECT_EQ(a(1, 2), view1(1, 0, 2));
        EXPECT_EQ(size_t(3), view1.dimension());
        EXPECT_EQ(size_t(3), view1.shape()[0]);
        EXPECT_EQ(size_t(1), view1.shape()[1]);
        EXPECT_EQ(size_t(4), view1.shape()[2]);
        EXPECT_EQ(view1(0, 1), view1(0, 0, 1));
        EXPECT_EQ(view1(2, 1, 0, 1), view1(1, 0, 1));

        auto view2 = strided_view(a, xstrided_slice_vector({ all(), all(), newaxis() }));
        EXPECT_EQ(a(1, 1), view2(1, 1, 0));
        EXPECT_EQ(a(1, 2), view2(1, 2, 0));
        EXPECT_EQ(size_t(3), view2.dimension());
        EXPECT_EQ(size_t(3), view2.shape()[0]);
        EXPECT_EQ(size_t(4), view2.shape()[1]);
        EXPECT_EQ(size_t(1), view2.shape()[2]);

        auto view3 = strided_view(a, xstrided_slice_vector({ 1, newaxis(), all() }));
        EXPECT_EQ(a(1, 1), view3(0, 1));
        EXPECT_EQ(a(1, 2), view3(0, 2));
        EXPECT_EQ(size_t(2), view3.dimension());

        auto view4 = strided_view(a, xstrided_slice_vector({ 1, all(), newaxis() }));
        EXPECT_EQ(a(1, 1), view4(1, 0));
        EXPECT_EQ(a(1, 2), view4(2, 0));
        EXPECT_EQ(size_t(2), view4.dimension());

        auto view5 = strided_view(view1, { 1 });
        EXPECT_EQ(a(1, 1), view5(0, 1));
        EXPECT_EQ(a(1, 2), view5(0, 2));
        EXPECT_EQ(size_t(2), view5.dimension());

        auto view6 = strided_view(view2, { 1 });
        EXPECT_EQ(a(1, 1), view6(1, 0));
        EXPECT_EQ(a(1, 2), view6(2, 0));
        EXPECT_EQ(size_t(2), view6.dimension());

        std::array<std::size_t, 3> idx1 = { 1, 0, 2 };
        EXPECT_EQ(a(1, 2), view1.element(idx1.begin(), idx1.end()));

        std::array<std::size_t, 3> idx2 = { 1, 2, 0 };
        EXPECT_EQ(a(1, 2), view2.element(idx2.begin(), idx2.end()));
    }

    TEST(xstrided_view, newaxis_iterating)
    {
        view_shape_type shape = { 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = strided_view(a, xstrided_slice_vector({ all(), all(), newaxis() }));
        auto iter1 = view1.template begin<layout_type::row_major>();
        auto iter1_end = view1.template end<layout_type::row_major>();

        EXPECT_EQ(a(0, 0), *iter1);
        ++iter1;
        EXPECT_EQ(a(0, 1), *iter1);
        ++iter1;
        EXPECT_EQ(a(0, 2), *iter1);
        ++iter1;
        EXPECT_EQ(a(0, 3), *iter1);
        ++iter1;
        EXPECT_EQ(a(1, 0), *iter1);
        ++iter1;
        EXPECT_EQ(a(1, 1), *iter1);
        ++iter1;
        EXPECT_EQ(a(1, 2), *iter1);
        ++iter1;
        EXPECT_EQ(a(1, 3), *iter1);
        ++iter1;
        EXPECT_EQ(a(2, 0), *iter1);
        ++iter1;
        EXPECT_EQ(a(2, 1), *iter1);
        ++iter1;
        EXPECT_EQ(a(2, 2), *iter1);
        ++iter1;
        EXPECT_EQ(a(2, 3), *iter1);
        ++iter1;
        EXPECT_EQ(iter1_end, iter1);

        auto view2 = strided_view(a, xstrided_slice_vector({ all(), newaxis(), all() }));
        auto iter2 = view2.template begin<layout_type::row_major>();
        auto iter2_end = view2.template end<layout_type::row_major>();

        EXPECT_EQ(a(0, 0), *iter2);
        ++iter2;
        EXPECT_EQ(a(0, 1), *iter2);
        ++iter2;
        EXPECT_EQ(a(0, 2), *iter2);
        ++iter2;
        EXPECT_EQ(a(0, 3), *iter2);
        ++iter2;
        EXPECT_EQ(a(1, 0), *iter2);
        ++iter2;
        EXPECT_EQ(a(1, 1), *iter2);
        ++iter2;
        EXPECT_EQ(a(1, 2), *iter2);
        ++iter2;
        EXPECT_EQ(a(1, 3), *iter2);
        ++iter2;
        EXPECT_EQ(a(2, 0), *iter2);
        ++iter2;
        EXPECT_EQ(a(2, 1), *iter2);
        ++iter2;
        EXPECT_EQ(a(2, 2), *iter2);
        ++iter2;
        EXPECT_EQ(a(2, 3), *iter2);
        ++iter2;
        EXPECT_EQ(iter2_end, iter2);
    }

    TEST(xstrided_view, newaxis_function)
    {
        view_shape_type shape = { 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        xarray<double> b(view_shape_type(1, 4));
        auto data_end = data.cbegin();
        data_end += 4;
        std::copy(data.cbegin(), data_end, b.template begin<layout_type::row_major>());

        auto v = strided_view(b, xstrided_slice_vector({ newaxis(), all() }));
        xarray<double> res = a + v;

        std::vector<double> data2{ 2, 4, 6, 8, 6, 8, 10, 12, 10, 12, 14, 16 };
        xarray<double> expected(shape);
        std::copy(data2.cbegin(), data2.cend(), expected.template begin<layout_type::row_major>());

        EXPECT_EQ(expected, res);
    }

    TEST(xstrided_view, range_adaptor)
    {
        using namespace xt::placeholders;
        using t = xarray<int>;
        t a = { 1, 2, 3, 4, 5 };

        auto n = xnone();

        auto v1 = strided_view(a, xstrided_slice_vector({ range(3, _) }));
        t v1e = { 4, 5 };
        EXPECT_TRUE(v1e == v1);

        auto v2 = strided_view(a, xstrided_slice_vector({ range(_, 2) }));
        t v2e = { 1, 2 };
        EXPECT_TRUE(v2e == v2);

        auto v3 = strided_view(a, xstrided_slice_vector({ range(n, n) }));
        t v3e = { 1, 2, 3, 4, 5 };
        EXPECT_TRUE(v3e == v3);

        auto v4 = strided_view(a, xstrided_slice_vector({ range(n, 2, -1) }));
        t v4e = { 5, 4 };
        EXPECT_TRUE(v4e == v4);

        auto v5 = strided_view(a, xstrided_slice_vector({ range(2, n, -1) }));
        t v5e = { 3, 2, 1 };
        EXPECT_TRUE(v5e == v5);

        auto v6 = strided_view(a, xstrided_slice_vector({ range(n, n, n) }));
        t v6e = { 1, 2, 3, 4, 5 };
        EXPECT_TRUE(v6e == v6);

        auto v7 = strided_view(a, xstrided_slice_vector({ range(1, n, 2) }));
        t v7e = { 2, 4 };
        EXPECT_TRUE(v7e == v7);

        auto v8 = strided_view(a, xstrided_slice_vector({ range(2, n, 2) }));
        t v8e = { 3, 5 };
        EXPECT_TRUE(v8e == v8);
    }

    TEST(xstrided_view, extended_assign)
    {
        using t = xarray<int>;
        t a = { 1, 2, 3, 4, 5 };

        auto v = strided_view(a, xstrided_slice_vector({ range(0, 2) }));
        v = 1000;
        EXPECT_EQ(v(0), 1000);
        EXPECT_EQ(a(0), 1000);
        EXPECT_EQ(a(1), 1000);

        auto v2 = strided_view(a, xstrided_slice_vector({ range(3, 5) }));
        t b = { -100, -100 };
        v2 = b;
        EXPECT_EQ(v2(1), -100);
        EXPECT_EQ(a(4), -100);
    }

    TEST(xstrided_view, ellipsis)
    {
        using t = xarray<int>;
        auto a = t::from_shape({ 5, 5, 1, 1, 1, 4 });
        std::iota(a.begin(), a.end(), 0);

        auto v1 = strided_view(a, { 1, 1, xt::ellipsis() });
        dynamic_shape<std::size_t> v1_s{ 1, 1, 1, 4 };
        EXPECT_EQ(v1.shape(), v1_s);

        EXPECT_EQ(strided_view(a, { 2, 2, xt::ellipsis() }), strided_view(a, { 2, 2, xt::all(), xt::all(), xt::all(), xt::all() }));
        EXPECT_EQ(strided_view(a, { 2, xt::ellipsis(), 0, 2 }), strided_view(a, { 2, xt::all(), xt::all(), xt::all(), 0, 2 }));

        EXPECT_THROW(strided_view(a, { xt::ellipsis(), 0, xt::ellipsis() }), std::runtime_error);

        t b = xt::ones<int>({ 5, 5, 5 });
        auto v2 = strided_view(b, { xt::ellipsis(), 1, 1, 1 });
        EXPECT_EQ(v2(), 1);
        EXPECT_EQ(v2.shape().size(), size_t(0));

        auto v3 = strided_view(b, { xt::ellipsis(), 1, xt::all(), 1 });
        dynamic_shape<std::size_t> v3_s{ 5 };
        EXPECT_EQ(v3.shape(), v3_s);


        EXPECT_THROW(strided_view(b, { xt::ellipsis(), 1, 1, 1, 1 }), std::runtime_error);
    }

    TEST(xstrided_view, incompatible_shape)
    {
        xarray<int> a = xarray<int>::from_shape({ 4, 3, 2 });
        xarray<int> b = xarray<int>::from_shape({ 2, 3, 4 });
        xstrided_slice_vector sv;
        sv.push_back(all());
        auto v = strided_view(a, sv);

        EXPECT_FALSE(broadcastable(v.shape(), b.shape()));
        EXPECT_FALSE(broadcastable(b.shape(), v.shape()));
        EXPECT_THROW(assert_compatible_shape(b, v), broadcast_error);
        EXPECT_THROW(assert_compatible_shape(v, b), broadcast_error);
        EXPECT_THROW(v = b, broadcast_error);
        EXPECT_THROW(noalias(v) = b, broadcast_error);
    }

    TEST(xstrided_view, strided_view_on_view)
    {
        xarray<int> a = xt::ones<int>({ 3, 4, 5 });
        auto v1 = view(a, 1, all(), all());
        auto vv1 = strided_view(v1, { 1, all() });
        vv1 = vv1 * 5;
        EXPECT_EQ(a(0, 0, 0), 1);
        EXPECT_EQ(a(1, 1, 0), 5);
        EXPECT_EQ(a(1, 1, 4), 5);
        EXPECT_EQ(a(1, 2, 4), 1);
        EXPECT_EQ(v1(1, 4), 5);
    }

    TEST(xstrided_view, strided_view_on_strided_view)
    {
        xarray<int> a = xt::ones<int>({ 3, 4, 5 });
        auto v1 = strided_view(a, { 1, all(), all() });
        auto vv1 = strided_view(v1, { 1, all() });
        vv1 = vv1 * 5;
        EXPECT_EQ(a(0, 0, 0), 1);
        EXPECT_EQ(a(1, 1, 0), 5);
        EXPECT_EQ(a(1, 1, 4), 5);
        EXPECT_EQ(a(1, 2, 4), 1);
        EXPECT_EQ(v1(1, 4), 5);

        a = xt::ones<int>({ 3, 4, 5 });
        auto v2 = strided_view(a, { all(), 1, all() });
        auto vv2 = strided_view(v2, { all(), 2 });
        vv2 = vv2 * 5;
        EXPECT_EQ(a(0, 0, 0), 1);
        EXPECT_EQ(a(1, 1, 2), 5);
        EXPECT_EQ(a(2, 1, 2), 5);
        EXPECT_EQ(a(0, 1, 2), 5);
        EXPECT_EQ(v2(0, 2), 5);
        EXPECT_TRUE(xt::all(equal(vv2, 5)));
    }

    TEST(xstrided_view, range_integer_casting)
    {
        // just check compilation
        auto arr = xarray<int>::from_shape({ 3, 4, 5 });
        auto a = strided_view(arr, { range(0, std::ptrdiff_t(2)), 323 });
        auto b = strided_view(arr, { range(std::size_t(0), 2), 323 });
    }

    TEST(xstrided_view, strides)
    {
        // Strides: 72/24/6/1
        xarray<int, layout_type::row_major> a = xarray<int, layout_type::row_major>::from_shape({ 5, 3, 4, 6 });

        auto s1 = strided_view(a, { 1, 1, xt::all(), xt::all() }).strides();
        std::vector<std::size_t> s1e = { 6, 1 };
        EXPECT_EQ(s1, s1e);

        auto s2 = strided_view(a, { 1, xt::all(), xt::all(), 1 }).strides();
        std::vector<std::size_t> s2e = { 24, 6 };
        EXPECT_EQ(s2, s2e);

        auto s3 = strided_view(a, { 1, xt::all(), 1, xt::newaxis(), xt::newaxis(), xt::all() }).strides();
        std::vector<std::size_t> s3e = { 24, 0, 0, 1 };
        EXPECT_EQ(s3, s3e);

        auto s4 = strided_view(a, { xt::range(0, 1, 2), 1, 0, xt::all(), xt::newaxis() }).strides();
        std::vector<std::size_t> s4e = { 0, 1, 0 };
        EXPECT_EQ(s4, s4e);

        auto s4x = strided_view(a, { xt::range(0, 5, 2), 1, 0, xt::all(), xt::newaxis() }).strides();
        std::vector<std::size_t> s4xe = { 72 * 2, 1, 0 };
        EXPECT_EQ(s4x, s4xe);

        auto s5 = strided_view(a, { xt::all(), 1 }).strides();
        std::vector<std::size_t> s5e = { 72, 6, 1 };
        EXPECT_EQ(s5, s5e);

        auto s6 = strided_view(a, { xt::all(), 1, 1, xt::newaxis(), xt::all() }).strides();
        std::vector<std::size_t> s6e = { 72, 0, 1 };
        EXPECT_EQ(s6, s6e);

        auto s7 = strided_view(a, { xt::all(), 1, xt::newaxis(), xt::all() }).strides();
        std::vector<std::size_t> s7e = { 72, 0, 6, 1 };
        EXPECT_EQ(s7, s7e);
    }

    TEST(xstrided_view, layout)
    {
        xarray<int, layout_type::row_major> a = xarray<int, layout_type::row_major>::from_shape({ 5, 3, 4, 6 });

        auto s1 = strided_view(a, { xt::all(), 1, xt::newaxis(), xt::all() }).layout();
        EXPECT_EQ(s1, layout_type::dynamic);

        auto s1x = strided_view(a, { 1, xt::all(), xt::newaxis(), xt::all() }).layout();
        EXPECT_EQ(s1x, layout_type::row_major);

        auto s2 = strided_view(a, { 1, 2, range(0, 3), xt::all() }).layout();
        EXPECT_EQ(s2, layout_type::row_major);

        auto s3 = strided_view(a, { 1 }).layout();
        EXPECT_EQ(s3, layout_type::row_major);

        xarray<int, layout_type::column_major> b = xarray<int, layout_type::column_major>::from_shape({ 5, 3, 4, 6 });

        auto s4 = strided_view(b, { 1 }).layout();
        EXPECT_EQ(s4, layout_type::dynamic);

        auto s5 = strided_view(b, { xt::all(), 1, 1, 1 }).layout();
        EXPECT_EQ(s5, layout_type::column_major);

        auto s6 = strided_view(b, { xt::all(), 1, 1, xt::range(0, 6) }).layout();
        EXPECT_EQ(s6, layout_type::dynamic);
    }

    TEST(xstrided_view, expression_adapter)
    {
        auto e = xt::arange<double>(24);
        auto sv = xstrided_slice_vector({range(2, 10, 3)});
        auto vt = strided_view(e, sv);

        EXPECT_EQ(vt(0), 2);
        EXPECT_EQ(vt(1), 5);

        xt::xarray<double> assigned = vt;
        EXPECT_EQ(assigned, vt);
        EXPECT_EQ(assigned(1), 5);
    }

    TEST(xstrided_view, reverse_iteration)
    {
        view_shape_type shape = {2, 3, 4};
        xarray<double, layout_type::row_major> a(shape);
        std::vector<double> data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = strided_view(a, {range(0, 2), 1, range(1, 4)});
        auto iter = view1.template rbegin<layout_type::row_major>();
        auto iter_end = view1.template rend<layout_type::row_major>();

        EXPECT_EQ(20, *iter);
        ++iter;
        EXPECT_EQ(19, *iter);
        ++iter;
        EXPECT_EQ(18, *iter);
        ++iter;
        EXPECT_EQ(8, *iter);
        ++iter;
        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(6, *iter);
        ++iter;
        EXPECT_EQ(iter, iter_end);

        auto view2 = strided_view(view1, {range(0, 2), range(1, 3)});
        auto iter2 = view2.template rbegin<layout_type::row_major>();
        auto iter_end2 = view2.template rend<layout_type::row_major>();

        EXPECT_EQ(20, *iter2);
        ++iter2;
        EXPECT_EQ(19, *iter2);
        ++iter2;
        EXPECT_EQ(8, *iter2);
        ++iter2;
        EXPECT_EQ(7, *iter2);
        ++iter2;
        EXPECT_EQ(iter2, iter_end2);
    }

    TEST(xstrided_view, reshape_view)
    {
        xarray<double> a = xt::arange<double>(9);
        auto av = xt::reshape_view<XTENSOR_DEFAULT_LAYOUT>(a, {3, 3});
        auto xv = xt::reshape_view<XTENSOR_DEFAULT_LAYOUT>(a, xshape<3, 3>());

        xtensor<double, 2> e = xt::reshape_view<XTENSOR_DEFAULT_LAYOUT>(xt::arange(9), {3, 3});
        xtensor<double, 2> es = xt::reshape_view<XTENSOR_DEFAULT_LAYOUT>(a, {3, 3});

        a.reshape({3, 3});
        for (std::size_t i = 0; i < 100; ++i)
        {
            xtensor<double, 2> par = xt::reshape_view<XTENSOR_DEFAULT_LAYOUT>(xt::arange(9), {3, 3});
            EXPECT_EQ(a, par);
        }
        using assign_traits = xassign_traits<xarray<double>, decltype(av)>;

#if XTENSOR_USE_XSIMD
        EXPECT_FALSE(assign_traits::simd_linear_assign());
#endif

        EXPECT_EQ(av, e);
        EXPECT_EQ(av, a);

        bool truthy;
        truthy = std::is_same<typename decltype(xv)::temporary_type, xtensor_fixed<double, xshape<3, 3>, layout_type::dynamic>>();
        EXPECT_TRUE(truthy);

#if !defined(X_OLD_CLANG)
        truthy = std::is_same<typename decltype(av)::temporary_type, xtensor<double, 2, layout_type::dynamic>>();
        EXPECT_TRUE(truthy);
        truthy = std::is_same<typename decltype(av)::shape_type, typename decltype(e)::shape_type>::value;
        EXPECT_TRUE(truthy);
#endif

        xarray<int> xa = {{1, 2, 3}, {4, 5, 6}};
        std::vector<std::size_t> new_shape = {3, 2};
        auto xrv = reshape_view(xa, new_shape);

        xarray<int> xres = {{1, 2}, {3, 4}, {5, 6}};
        EXPECT_EQ(xrv, xres);
    }

    TEST(xstrided_view, reshape_view_assign)
    {
        xarray<int, layout_type::column_major> xa = {{1, 2, 3}, {4, 5, 6}};
        xarray<int, layout_type::column_major> exp = {{1, 2},
                                                      {3, 4},
                                                      {5, 6}};
        auto v = reshape_view(xa, {3, 2});
        xarray<int, layout_type::column_major> res = v;
        EXPECT_EQ(res, exp);
    }

    TEST(xstrided_view, on_xview)
    {
        xarray<double> a = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        auto v = view(a, keep(3, 5, 6));
        auto v2 = strided_view(v, {2});
        EXPECT_EQ(v2(0), 6);
        auto v3 = strided_view(v, {all()});
        EXPECT_EQ(v3(0), 3);
        EXPECT_EQ(v3(1), 5);
    }

    TEST(xstrided_view, on_xbroadcast)
    {
        xarray<double, layout_type::column_major> a = 
          {{  0.0,  1.0,  2.0},
           { 10.0, 11.0, 12.0}};

        auto b = broadcast(a, {2, 3});
        auto v = strided_view(b, {0});
        EXPECT_EQ(v(0), 0.0);
        EXPECT_EQ(v(1), 1.0);
        EXPECT_EQ(v(2), 2.0);
    }

    TEST(xstrided_view, on_broadcasted_scalar)
    {
        xarray<double> expected = { 1, 1, 1, 1, 1, 1, 1, 1 };
        xarray<double> a = xt::squeeze(xt::ones<double>({8, 1}));
        EXPECT_EQ(a, expected);
    }

    TEST(xstrided_view, on_transpose)
    {
        xt::xarray<double> arr
          {{ 0.0,  1.0,  2.0},
           {10.0, 11.0, 12.0}};

        auto t = xt::transpose(arr + arr);
        auto v = xt::strided_view(t, {0});
        EXPECT_EQ(v(0), 0.0);
        EXPECT_EQ(v(1), 20.0);
    }

    TEST(xstrided_view, reshape_strided)
    {
        xtensor<int, 2> a = {{  1,  2,  3,  4,  5,  6,  7,  8 },
                             { 11, 12, 13, 14, 15, 16, 17, 18 }};

        auto v1 = view(a, all(), range(0, 4));
        auto rv1 = reshape_view(v1, {2, 2 ,2});
        xtensor<int, 3> expected = {{{1, 2}, {3, 4}}, {{11, 12}, {13, 14}}};
        EXPECT_EQ(expected, rv1);

        auto rv2 = reshape_view(a, {2, 2, 4});
        auto v2 = strided_view(rv2, {0, 0, all()});
        xtensor<int, 1> expected2 = {1, 2, 3 ,4};
        EXPECT_EQ(expected2, v2);
    }

    TEST(xstrided_view, zerod_view_iterator)
    {
        xt::xarray<int> a{ { { 1, 2 },{ 3, 4 },{ 5, 6 } },{ { 7, 8 },{ 9, 10 },{ 11, 12 } } };
        xt::xstrided_slice_vector sl = { 1, 0, 1 };

        auto vi = xt::strided_view(a, sl);
        auto it0 = vi.cbegin<layout_type::row_major>();
        auto it1 = it0 + 0;
        EXPECT_EQ(*it0, *it1);

        auto it2 = vi.cbegin<layout_type::column_major>();
        auto it3 = it2 + 0;
        EXPECT_EQ(*it2, *it3);
    }

    TEST(xstrided_view, view_on_const)
    {
        const xtensor<int, 1> a = {1, 2, 3, 4};
        auto v = strided_view(a, {all()});
        auto d = v.data();
        EXPECT_TRUE((std::is_same<decltype(d), const int*>::value));
        EXPECT_EQ(d[0], 1);
    }

    namespace test
    {
        template <class E1, class E2>
        void check_reshaped_1d(const E1& e1, const E2& e2)
        {
            EXPECT_EQ(e1(0, 0), e2(0, 0));
            EXPECT_EQ(e1(0, 1), e2(0, 1));
            EXPECT_EQ(e1(1, 0), e2(1, 0));
            EXPECT_EQ(e1(1, 1), e2(1, 1));
            EXPECT_EQ(e1(2, 0), e2(2, 0));
            EXPECT_EQ(e1(2, 1), e2(2, 1));

            auto iter1 = e1.template cbegin<layout_type::row_major>();
            auto iter2 = e2.template cbegin<layout_type::row_major>();
            EXPECT_EQ(*iter1, e1(0, 0));
            EXPECT_EQ(*iter1++, *iter2++);
            EXPECT_EQ(*iter1, e1(0, 1));
            EXPECT_EQ(*iter1++, *iter2++);
            EXPECT_EQ(*iter1, e1(1, 0));
            EXPECT_EQ(*iter1++, *iter2++);
            EXPECT_EQ(*iter1, e1(1, 1));
            EXPECT_EQ(*iter1++, *iter2++);
            EXPECT_EQ(*iter1, e1(2, 0));
            EXPECT_EQ(*iter1++, *iter2++);
            EXPECT_EQ(*iter1, e1(2, 1));
            EXPECT_EQ(*iter1++, *iter2++);
            EXPECT_EQ(iter1, e1.template cend<layout_type::row_major>());
            EXPECT_EQ(iter2, e2.template cend<layout_type::row_major>());
        }

        template <class E1, class E2>
        void check_reshaped_2d_rm(const E1& e1, const E2& e2)
        {
            check_reshaped_1d(e1, e2);
            EXPECT_EQ(e1(0, 0), 1);
            EXPECT_EQ(e1(0, 1), 2);
            EXPECT_EQ(e1(1, 0), 3);
            EXPECT_EQ(e1(1, 1), 4);
            EXPECT_EQ(e1(2, 0), 5);
            EXPECT_EQ(e1(2, 1), 6);
        }

        template <class E1, class E2>
        void check_reshaped_2d_cm(const E1& e1, const E2& e2)
        {
            check_reshaped_1d(e1, e2);
            EXPECT_EQ(e1(0, 0), 1);
            EXPECT_EQ(e1(0, 1), 5);
            EXPECT_EQ(e1(1, 0), 4);
            EXPECT_EQ(e1(1, 1), 3);
            EXPECT_EQ(e1(2, 0), 2);
            EXPECT_EQ(e1(2, 1), 6);
        }
    }

    TEST(xstrided_view, reshape_view_1d)
    {
        xarray<int> a = { 1, 2, 3, 4, 5, 6 };

        auto var = reshape_view<layout_type::row_major>(a, { 3, 2 });
        xarray<int> rvar = var;
        test::check_reshaped_1d(rvar, var);
        EXPECT_EQ(rvar(0, 0), 1);
        EXPECT_EQ(rvar(0, 1), 2);
        EXPECT_EQ(rvar(1, 0), 3);
        EXPECT_EQ(rvar(1, 1), 4);
        EXPECT_EQ(rvar(2, 0), 5);
        EXPECT_EQ(rvar(2, 1), 6);

        auto vac = reshape_view<layout_type::column_major>(a, { 3, 2 });
        xarray<int> rvac = vac;
        test::check_reshaped_1d(rvac, vac);
        EXPECT_EQ(rvac(0, 0), 1);
        EXPECT_EQ(rvac(0, 1), 4);
        EXPECT_EQ(rvac(1, 0), 2);
        EXPECT_EQ(rvac(1, 1), 5);
        EXPECT_EQ(rvac(2, 0), 3);
        EXPECT_EQ(rvac(2, 1), 6);
    }

    TEST(xstrided_view, reshape_view_2d)
    {
        xarray<int, layout_type::row_major> ra = { {1, 2, 3}, {4, 5, 6} };

        auto vrar = reshape_view<layout_type::row_major>(ra, { 3, 2 });
        xarray<int> rvrar = vrar;
        test::check_reshaped_2d_rm(rvrar, vrar);

        auto vrac = reshape_view<layout_type::column_major>(ra, { 3, 2 });
        xarray<int> rvrac = vrac;
        test::check_reshaped_2d_cm(rvrac, vrac);

        xarray<int, layout_type::column_major> ca = { { 1, 2, 3 },{ 4, 5, 6 } };

        auto vcar = reshape_view<layout_type::row_major>(ca, { 3, 2 });
        xarray<int> rvcar = vcar;
        test::check_reshaped_2d_rm(rvcar, vcar);

        auto vcac = reshape_view<layout_type::column_major>(ca, { 3, 2 });
        xarray<int> rvcac = vcac;
        test::check_reshaped_2d_cm(rvcac, vcac);
    }

    TEST(xstrided_view, flatten)
    {
        xt::xarray<int> x = {{1, 2, 3}, {4, 5, 6}};
        auto x_view = xt::strided_view(x, {xt::range(0, 1), xt::range(0, 1)});
        auto x_flat = xt::flatten<xt::layout_type::column_major>(x_view);
        x_flat = 10;
        xt::xarray<int> exp = {{10, 2, 3}, {4, 5, 6}};
        EXPECT_EQ(x, exp);

        auto x_view2 = xt::strided_view(x, {xt::range(0, 1), xt::range(0, 2)});
        auto x_flat2 = xt::flatten<xt::layout_type::column_major>(x_view2);
        x_flat2 = 15;
        xt::xarray<int> exp2 = {{15, 15, 3}, {4, 5, 6}};
        EXPECT_EQ(x, exp2);

        xt::xarray<int> b = {20, 25};
        x_flat2 = b;
        xt::xarray<int> exp3 = {{20, 25, 3}, {4, 5, 6}};
        EXPECT_EQ(x, exp3);
    }
}
