/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <algorithm>

#include "gtest/gtest.h"
#include "test_common_macros.hpp"

// Workaround to avoid warnings regarding initialization
// of distribution internal variables
#if (defined(__GNUC__) && !defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#include "xtensor/xgenerator.hpp"
#pragma GCC diagnostic pop
#else
#endif


#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"

namespace xt
{
    using std::size_t;
    using view_shape_type = dynamic_shape<size_t>;

    template <class A, class B, std::ptrdiff_t BB, std::ptrdiff_t BE>
    bool operator==(const A& lhs, const sequence_view<B, BB, BE>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(rhs.begin(), rhs.end(), lhs.begin());
    }

    template <class A, class B, std::ptrdiff_t BB, std::ptrdiff_t BE>
    bool operator==(const sequence_view<B, BB, BE>& lhs, const A& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(rhs.begin(), rhs.end(), lhs.begin());
    }

    TEST(xview, temporary_type)
    {
        {
            view_shape_type shape = {3, 4};
            xarray<double> a(shape);
            auto view1 = view(a, 1, range(1, 4));
            bool check = std::is_same<xarray<double>, typename xcontainer_inner_types<decltype(view1)>::temporary_type>::value;
            EXPECT_TRUE(check);
        }

        {
            xtensor<double, 2>::shape_type shape = {3, 4};
            xtensor<double, 2> a(shape);
            auto view1 = view(a, 1, range(1, 4));
            bool check1 = std::is_same<xtensor<double, 1>, typename xcontainer_inner_types<decltype(view1)>::temporary_type>::value;
            EXPECT_TRUE(check1);

            auto view2 = view(a, all(), newaxis(), range(1, 4));
            bool check2 = std::is_same<xtensor<double, 3>, typename xcontainer_inner_types<decltype(view2)>::temporary_type>::value;
            EXPECT_TRUE(check2);
        }
    }

    TEST(xview, simple)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, 1, range(1, 4));
        EXPECT_EQ(a(1, 1), view1(0));
        EXPECT_EQ(a(1, 2), view1(1));
        EXPECT_EQ(size_t(1), view1.dimension());
        XT_EXPECT_ANY_THROW(view1.at(10));
        XT_EXPECT_ANY_THROW(view1.at(0, 0));

        auto view0 = view(a, 0, range(0, 3));
        EXPECT_EQ(a(0, 0), view0(0));
        EXPECT_EQ(a(0, 1), view0(1));
        EXPECT_EQ(size_t(1), view0.dimension());
        EXPECT_EQ(size_t(3), view0.shape(0));

        auto view2 = view(a, range(0, 2), 2);
        EXPECT_EQ(a(0, 2), view2(0));
        EXPECT_EQ(a(1, 2), view2(1));
        EXPECT_EQ(size_t(1), view2.dimension());
        EXPECT_EQ(size_t(2), view2.shape(0));

        auto view4 = view(a, 1);
        EXPECT_EQ(size_t(1), view4.dimension());
        EXPECT_EQ(size_t(4), view4.shape(0));

        auto view5 = view(view4, 1);
        EXPECT_EQ(size_t(0), view5.dimension());
        EXPECT_EQ(size_t(0), view5.shape().size());

        auto view6 = view(a, 1, all());
        EXPECT_EQ(a(1, 0), view6(0));
        EXPECT_EQ(a(1, 1), view6(1));
        EXPECT_EQ(a(1, 2), view6(2));
        EXPECT_EQ(a(1, 3), view6(3));

        auto view7 = view(a, all(), 2);
        EXPECT_EQ(a(0, 2), view7(0));
        EXPECT_EQ(a(1, 2), view7(1));
        EXPECT_EQ(a(2, 2), view7(2));

        if (a.layout() == layout_type::row_major)
        {
            EXPECT_EQ(a.layout(), view1.layout());
            EXPECT_EQ(layout_type::dynamic, view2.layout());
            EXPECT_EQ(a.layout(), view4.layout());
            EXPECT_EQ(a.layout(), view5.layout());
            EXPECT_EQ(a.layout(), view6.layout());
            EXPECT_EQ(layout_type::dynamic, view7.layout());
        }
        else
        {
            EXPECT_EQ(layout_type::dynamic, view1.layout());
            EXPECT_EQ(a.layout(), view2.layout());
            EXPECT_EQ(layout_type::dynamic, view4.layout());
            // TODO ideally this would return the underlying expression's layout
            // but needs special casing 'view-on-view'
            EXPECT_EQ(layout_type::dynamic, view5.layout());
            EXPECT_EQ(layout_type::dynamic, view6.layout());
            EXPECT_EQ(a.layout(), view7.layout());
        }
    }

    TEST(xview, negative_index)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view0 = view(a, -2, range(1, 4));
        auto view1 = view(a, 1, range(1, 4));
        EXPECT_EQ(view0, view1);
    }

    TEST(xview, stored_range)
    {
        view_shape_type shape = { 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto r0 = range(1, 3);
        auto r1 = range(0, 3);
        auto view0 = view(a, r0, r1);

        view0 += xt::ones<double>({2, 3});
        EXPECT_EQ(a(1, 0), 6);
        EXPECT_EQ(a(1, 1), 7);
        EXPECT_EQ(a(1, 2), 8);
        EXPECT_EQ(a(2, 0), 10);
        EXPECT_EQ(a(2, 1), 11);
        EXPECT_EQ(a(2, 2), 12);
    }

    TEST(xview, copy_semantic)
    {
        view_shape_type shape = { 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        {
            SCOPED_TRACE("copy constructor");
            auto view1 = view(a, 1, range(1, 4));
            auto view2(view1);
            EXPECT_EQ(a(1, 1), view2(0));
            EXPECT_EQ(a(1, 2), view2(1));
            EXPECT_EQ(size_t(1), view2.dimension());
            if (a.layout() == layout_type::row_major)
            {
                EXPECT_EQ(a.layout(), view2.layout());
            }
            else
            {
                EXPECT_EQ(layout_type::dynamic, view2.layout());
            }
        }

        {
            SCOPED_TRACE("copy assignment operator");
            auto view1 = view(a, 1, range(1, 4));
            auto view2 = view(a, 2, range(0, 3));
            view2 = view1;
            EXPECT_EQ(a(2, 0), a(1, 1));
            EXPECT_EQ(a(2, 1), a(1, 2));
            EXPECT_EQ(a(2, 2), a(1, 3));
        }
    }

    TEST(xview, move_semantic)
    {
        view_shape_type shape = { 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        {
            SCOPED_TRACE("copy constructor");
            auto view1 = view(a, 1, range(1, 4));
            auto view2(std::move(view1));
            EXPECT_EQ(a(1, 1), view2(0));
            EXPECT_EQ(a(1, 2), view2(1));
            EXPECT_EQ(size_t(1), view2.dimension());
            if (a.layout() == layout_type::row_major)
            {
                EXPECT_EQ(a.layout(), view2.layout());
            }
            else
            {
                EXPECT_EQ(layout_type::dynamic, view2.layout());
            }
        }

        {
            SCOPED_TRACE("copy assignment operator");
            auto view1 = view(a, 1, range(1, 4));
            auto view2 = view(a, 2, range(0, 3));
            view2 = std::move(view1);
            EXPECT_EQ(a(2, 0), a(1, 1));
            EXPECT_EQ(a(2, 1), a(1, 2));
            EXPECT_EQ(a(2, 2), a(1, 3));
        }
    }

    TEST(xview, three_dimensional)
    {
        view_shape_type shape = {3, 4, 2};
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
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, 1);
        EXPECT_EQ(size_t(2), view1.dimension());
        EXPECT_EQ(a(1, 0, 0), view1(0, 0));
        EXPECT_EQ(a(1, 0, 1), view1(0, 1));
        EXPECT_EQ(a(1, 1, 0), view1(1, 0));
        EXPECT_EQ(a(1, 1, 1), view1(1, 1));
        XT_EXPECT_ANY_THROW(view1.at(10, 10));
        XT_EXPECT_ANY_THROW(view1.at(0, 0, 0));

        std::array<std::size_t, 2> idx = {1, 1};
        EXPECT_EQ(a(1, 1, 1), view1.element(idx.cbegin(), idx.cend()));
    }

    TEST(xview, integral_count)
    {
        size_t squeeze1 = integral_count<size_t, size_t, size_t, xrange<size_t>>();
        EXPECT_EQ(squeeze1, size_t(3));
        size_t squeeze2 = integral_count<size_t, xrange<size_t>, size_t>();
        EXPECT_EQ(squeeze2, size_t(2));
        size_t squeeze3 = integral_count_before<size_t, size_t, size_t, xrange<size_t>>(3);
        EXPECT_EQ(squeeze3, size_t(3));
        size_t squeeze4 = integral_count_before<size_t, xrange<size_t>, size_t>(2);
        EXPECT_EQ(squeeze4, size_t(1));
        size_t squeeze5 = integral_count<xnewaxis<size_t>>();
        EXPECT_EQ(squeeze5, size_t(0));
    }

    TEST(xview, integral_skip)
    {
        size_t index0 = integral_skip<size_t, xrange<size_t>, size_t, xrange<size_t>>(0);
        size_t index1 = integral_skip<size_t, xrange<size_t>, size_t, xrange<size_t>>(1);
        size_t index2 = integral_skip<size_t, xrange<size_t>, size_t, xrange<size_t>>(2);
        EXPECT_EQ(index0, size_t(1));
        EXPECT_EQ(index1, size_t(3));
        EXPECT_EQ(index2, size_t(4));
    }

    TEST(xview, single_newaxis_shape)
    {
        xarray<double> a = {1, 2, 3, 4};
        auto v = view(a, newaxis());
        view_shape_type s = {1, 4};
        EXPECT_EQ(s, v.shape());
    }

    TEST(xview, temporary_view)
    {
        xt::xarray<double> arr1
         {{1.0, 2.0, 3.0},
          {2.0, 5.0, 7.0},
          {2.0, 5.0, 7.0}};

        xt::xarray<double> arr2
         {5.0, 6.0, 7.0};

        xt::xarray<double> res = xt::view(arr1, 1) + arr2;
        EXPECT_EQ(7., res(0));
        EXPECT_EQ(11., res(1));
        EXPECT_EQ(14., res(2));
    }

    TEST(xview, access)
    {
        xt::xarray<double> arr
        {{ 1.0, 2.0, 3.0 },
         { 2.0, 5.0, 7.0 },
         { 2.0, 5.0, 7.0 }};

        auto v1 = xt::view(arr, 1, xt::range(1, 3));
        EXPECT_EQ(v1(), arr(0, 1));
        EXPECT_EQ(v1(1), arr(1, 2));
        EXPECT_EQ(v1(1, 1), arr(1, 2));

        auto v2 = xt::view(arr, all(), newaxis(), all());
        //EXPECT_EQ(v2(1), arr(0, 1));
        EXPECT_EQ(v2(1, 0, 2), arr(1, 2));
        EXPECT_EQ(v2(2, 1, 0, 2), arr(1, 2));

        auto v3 = xt::view(arr, xt::range(0, 2), xt::range(1, 3));
        //EXPECT_EQ(v3(1), arr(0, 2));
        EXPECT_EQ(v3(1, 1), arr(1, 2));
        EXPECT_EQ(v3(2, 3, 1, 1), arr(1, 2));
    }

    TEST(xview, unchecked)
    {
        xt::xarray<double> arr
        { { 1.0, 2.0, 3.0 },
        { 2.0, 5.0, 7.0 },
        { 2.0, 5.0, 7.0 } };

        auto v1 = xt::view(arr, 1, xt::range(1, 3));
        EXPECT_EQ(v1.unchecked(1), arr(1, 2));

        auto v2 = xt::view(arr, all(), newaxis(), all());
        EXPECT_EQ(v2.unchecked(1, 0, 2), arr(1, 2));

        auto v3 = xt::view(arr, xt::range(0, 2), xt::range(1, 3));
        EXPECT_EQ(v3.unchecked(1, 1), arr(1, 2));
    }

    TEST(xview, iterator)
    {
        view_shape_type shape = {2, 3, 4};
        xarray<double, layout_type::row_major> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, range(0, 2), 1, range(1, 4));
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

        auto view2 = view(view1, range(0, 2), range(1, 3));
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
    }

    TEST(xview, fill)
    {
        view_shape_type shape = { 2, 3, 4 };
        xarray<double, layout_type::row_major> a(shape), res(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());
        std::vector<double> data_res = { 1, 2, 3, 4, 5, 4, 4, 4, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 4, 4, 4, 21, 22, 23, 24 };
        std::copy(data_res.cbegin(), data_res.cend(), res.template begin<layout_type::row_major>());
        auto view1 = view(a, range(0, 2), 1, range(1, 4));
        view1.fill(4);
        EXPECT_EQ(a, res);
    }

    TEST(xview, reverse_iterator)
    {
        view_shape_type shape = {2, 3, 4};
        xarray<double, layout_type::row_major> a(shape);
        std::vector<double> data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, range(0, 2), 1, range(1, 4));
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

        auto view2 = view(view1, range(0, 2), range(1, 3));
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

    TEST(xview, xview_on_xfunction)
    {
        view_shape_type shape = {3, 4};
        xarray<int> a(shape);
        std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        view_shape_type shape2 = {4};
        xarray<int> b(shape2);
        std::vector<int> data2 = {1, 2, 3, 4};
        std::copy(data2.cbegin(), data2.cend(), b.template begin<layout_type::row_major>());

        auto v = view(a + b, 1, range(1, 4));
        auto iter = v.begin();
        auto iter_end = v.end();

        EXPECT_EQ(8, *iter);
        ++iter;
        EXPECT_EQ(10, *iter);
        ++iter;
        EXPECT_EQ(12, *iter);
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xview, xview_on_xtensor)
    {
        xtensor<int, 2> a({3, 4});
        std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, 1, range(1, 4));
        EXPECT_EQ(a(1, 1), view1(0));
        EXPECT_EQ(a(1, 2), view1(1));
        EXPECT_EQ(size_t(1), view1.dimension());

        auto iter = view1.template begin<layout_type::row_major>();

        EXPECT_EQ(6, *iter);
        ++iter;
        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(8, *iter);

        xarray<int> b({3}, 2);
        xtensor<int, 1> res = view1 + b;
        EXPECT_EQ(8, res(0));
        EXPECT_EQ(9, res(1));
        EXPECT_EQ(10, res(2));
    }

    TEST(xview, on_const_array)
    {
        const xt::xarray<int> a1{{0, 1}, {2, 3}};
        auto a2 = xt::view(a1, 1, xt::range(1, 2));
        int v2 = a2(0);
        EXPECT_EQ(v2, 3);

        auto it = a2.begin();
        EXPECT_EQ(*it, v2);
    }

    TEST(xview, trivial_iterating)
    {
        using tensor_type = xtensor<double, 1>;
        using shape_type = tensor_type::shape_type;
        tensor_type arr1{shape_type{2}};
        std::fill(arr1.begin(), arr1.end(), 6);
        auto view = xt::view(arr1, 0);
        auto iter = view.begin();
        auto iter_end = view.end();
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xview, const_trivial_iterating)
    {
        using tensor_type = xtensor<double, 1>;
        using shape_type = tensor_type::shape_type;
        tensor_type arr1{shape_type{2}};
        std::fill(arr1.begin(), arr1.end(), 6);
        const tensor_type arr2 = arr1;
        auto view = xt::view(arr2, 0);
        auto iter = view.begin();
        auto iter_end = view.end();
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xview, const_view)
    {
        typename xtensor<double, 3>::shape_type shape3 = {1, 2, 3};
        typename xtensor<double, 2>::shape_type shape2 = {2, 3};
        const xtensor<double, 3> arr(shape3, 2.5);
        xtensor<double, 2> arr2(shape2, 0.0);
        xtensor<double, 2> ref(shape2, 2.5);
        arr2 = xt::view(arr, 0);
        EXPECT_EQ(ref, arr2);
    }

    TEST(xview, newaxis_count)
    {
        size_t count1 = newaxis_count<xnewaxis<size_t>, xnewaxis<size_t>, xnewaxis<size_t>, xrange<size_t>>();
        EXPECT_EQ(count1, size_t(3));
        size_t count2 = newaxis_count<xnewaxis<size_t>, xrange<size_t>, xnewaxis<size_t>>();
        EXPECT_EQ(count2, size_t(2));
        size_t count3 = newaxis_count_before<xnewaxis<size_t>, xnewaxis<size_t>, xnewaxis<size_t>, xrange<size_t>>(3);
        EXPECT_EQ(count3, size_t(3));
        size_t count4 = newaxis_count_before<xnewaxis<size_t>, xrange<size_t>, xnewaxis<size_t>>(2);
        EXPECT_EQ(count4, size_t(1));
    }

    TEST(xview, newaxis)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, all(), newaxis(), all());
        EXPECT_EQ(a(1, 1), view1(1, 0, 1));
        EXPECT_EQ(a(1, 2), view1(1, 0, 2));
        EXPECT_EQ(size_t(3), view1.dimension());
        EXPECT_EQ(size_t(3), view1.shape()[0]);
        EXPECT_EQ(size_t(1), view1.shape()[1]);
        EXPECT_EQ(size_t(4), view1.shape()[2]);

        auto view2 = view(a, all(), all(), newaxis());
        EXPECT_EQ(a(1, 1), view2(1, 1, 0));
        EXPECT_EQ(a(1, 2), view2(1, 2, 0));
        EXPECT_EQ(size_t(3), view2.dimension());
        EXPECT_EQ(size_t(3), view2.shape()[0]);
        EXPECT_EQ(size_t(4), view2.shape()[1]);
        EXPECT_EQ(size_t(1), view2.shape()[2]);

        auto view3 = view(a, 1, newaxis(), all());
        EXPECT_EQ(a(1, 1), view3(0, 1));
        EXPECT_EQ(a(1, 2), view3(0, 2));
        EXPECT_EQ(size_t(2), view3.dimension());

        auto view4 = view(a, 1, all(), newaxis());
        EXPECT_EQ(a(1, 1), view4(1, 0));
        EXPECT_EQ(a(1, 2), view4(2, 0));
        EXPECT_EQ(size_t(2), view4.dimension());

        auto view5 = view(view1, 1);
        EXPECT_EQ(a(1, 1), view5(0, 1));
        EXPECT_EQ(a(1, 2), view5(0, 2));
        EXPECT_EQ(size_t(2), view5.dimension());

        auto view6 = view(view2, 1);
        EXPECT_EQ(a(1, 1), view6(1, 0));
        EXPECT_EQ(a(1, 2), view6(2, 0));
        EXPECT_EQ(size_t(2), view6.dimension());

        std::array<std::size_t, 3> idx1 = {1, 0, 2};
        EXPECT_EQ(a(1, 2), view1.element(idx1.begin(), idx1.end()));

        std::array<std::size_t, 3> idx2 = {1, 2, 0};
        EXPECT_EQ(a(1, 2), view2.element(idx2.begin(), idx2.end()));

        std::array<std::size_t, 2> idx3 = {1, 2};
        EXPECT_EQ(a(1, 2), view3.element(idx3.begin(), idx3.end()));

        xt::xarray<float> x5 = xt::ones<float>({1,4,16,16});
        auto view7 = xt::view(x5, xt::all(), xt::newaxis(), xt::all(), xt::all(), xt::all());
        std::array<std::size_t, 5> idx4 = {0, 0, 2, 14, 12};
        EXPECT_EQ(view7.element(idx4.begin(), idx4.end()), 1.f);
    }

    TEST(xview, newaxis_iterating)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, all(), all(), newaxis());
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

        auto view2 = view(a, all(), newaxis(), all());
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

    TEST(xview, newaxis_function)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        xarray<double> b(view_shape_type(1, 4));
        std::copy(data.cbegin(), data.cbegin() + 4, b.template begin<layout_type::row_major>());

        auto v = view(b, newaxis(), all());
        xarray<double> res = a + v;

        std::vector<double> data2{2, 4, 6, 8, 6, 8, 10, 12, 10, 12, 14, 16};
        xarray<double> expected(shape);
        std::copy(data2.cbegin(), data2.cend(), expected.template begin<layout_type::row_major>());

        EXPECT_EQ(expected, res);
    }

    TEST(xview, range_adaptor)
    {
        using namespace xt::placeholders;
        using t = xarray<int>;
        t a = {1, 2, 3, 4, 5};

        auto n = xnone();

        auto v1 = view(a, range(3, _));
        t v1e = {4, 5};
        EXPECT_TRUE(v1e == v1);

        auto v2 = view(a, range(_, 2));
        t v2e = {1, 2};
        EXPECT_TRUE(v2e == v2);

        auto v3 = view(a, range(n, n));
        t v3e = {1, 2, 3, 4, 5};
        EXPECT_TRUE(v3e == v3);

        auto v4 = view(a, range(n, 2, -1));
        t v4e = {5, 4};
        EXPECT_TRUE(v4e == v4);

        auto v5 = view(a, range(2, n, -1));
        t v5e = {3, 2, 1};
        EXPECT_TRUE(v5e == v5);

        auto v6 = view(a, range(n, n, n));
        t v6e = {1, 2, 3, 4, 5};
        EXPECT_TRUE(v6e == v6);

        auto v7 = view(a, range(1, n, 2));
        t v7e = {2, 4};
        EXPECT_TRUE(v7e == v7);

        auto v8 = view(a, range(2, n, 2));
        t v8e = {3, 5};
        EXPECT_TRUE(v8e == v8);
    }

    TEST(xview, data_interface)
    {
        using namespace xt::placeholders;
        using T = xarray<int>;
        xarray<int> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        using shape_type = typename T::shape_type;
        using index_type = typename T::shape_type;
        using size_type = typename T::size_type;

        auto next_idx = [](index_type& idx, const shape_type& shape)
        {
            for (size_type j = shape.size(); j != 0; --j)
            {
                size_type i = j - 1;
                if (idx[i] >= shape[i] - 1)
                {
                    idx[i] = 0;
                }
                else
                {
                    idx[i]++;
                    return idx;
                }
            }
            // return empty index, happens at last iteration step, but remains unused
            return index_type();
        };

        auto v1 = view(a, xt::all(), 1);
        auto shape1 = v1.shape();
        auto idx1 = index_type(shape1.size(), 0);
        auto strides1 = v1.strides();
        for (std::size_t i = 0; i < v1.size(); ++i)
        {
            auto linear_idx = std::inner_product(idx1.begin(), idx1.end(), strides1.begin(), std::size_t(0));
            EXPECT_EQ(v1[idx1], v1.data()[v1.data_offset() + linear_idx]);
            next_idx(idx1, shape1);
        }

        auto v2 = view(a, 1, range(_, _, 2));
        auto shape2 = v2.shape();
        auto idx2 = index_type(shape2.size(), 0);
        auto strides2 = v2.strides();
        for (std::size_t i = 0; i < v2.size(); ++i)
        {
            auto linear_idx = std::inner_product(idx2.begin(), idx2.end(), strides2.begin(), std::size_t(0));
            EXPECT_EQ(v2[idx2], v2.data()[v2.data_offset() + linear_idx]);
            next_idx(idx2, shape2);
        }
    }

    TEST(xview, strides_type)
    {
        xt::xtensor<float, 2> a{
            { 1, 2 },
            { 3, 4 },
            { 5, 6 }
        };
        auto row = xt::view(a, 1, xt::all());
        if (a.layout() == layout_type::row_major)
        {
            bool cond1 = std::is_same<decltype(row)::strides_type, std::array<std::ptrdiff_t, 1>>::value;
            bool cond2 = std::is_same<decltype(row.strides()), const xt::sequence_view<std::array<std::ptrdiff_t, 2>, 1, 2>&>::value;
            EXPECT_TRUE(cond1);
            EXPECT_TRUE(cond2);
        }
        else
        {
            bool cond1 = std::is_same<decltype(row)::strides_type, std::array<std::ptrdiff_t, 1>>::value;
            bool cond2 = std::is_same<decltype(row.strides()), const std::array<std::ptrdiff_t, 1>&>::value;
            EXPECT_TRUE(cond1);
            EXPECT_TRUE(cond2);
        }
    }

    TEST(xview, transpose)
    {
        xt::xarray<int> vector = xt::linspace(1, 10, 10);
        auto matrix = xt::view(vector, xt::all(), xt::newaxis());
        auto mt = xt::transpose(matrix);
        EXPECT_EQ(mt.shape(), std::vector<std::size_t>({1, 10}));
        EXPECT_EQ(mt.strides(), std::vector<std::ptrdiff_t>({0, 1}));
        int sum = 0;
        for (std::size_t i = 0; i < vector.size(); ++i)
        {
            sum += mt(0, i);
        }
        EXPECT_EQ(55, sum);
    }

    TEST(xview, incompatible_shape)
    {
        xarray<int> a = xarray<int>::from_shape({4, 3, 2});
        xarray<int> b = xarray<int>::from_shape({2, 3, 4});
        auto v = view(a, all());

        EXPECT_FALSE(broadcastable(v.shape(), b.shape()));
        EXPECT_FALSE(broadcastable(b.shape(), v.shape()));
        XT_EXPECT_THROW(assert_compatible_shape(b, v), broadcast_error);
        XT_EXPECT_THROW(assert_compatible_shape(v, b), broadcast_error);
        XT_EXPECT_THROW(v = b, broadcast_error);
        XT_EXPECT_THROW(noalias(v) = b, broadcast_error);
    }

    TEST(xview, strides)
    {
        // Strides: 72/24/6/1
        xarray<int, layout_type::row_major> a = xarray<int, layout_type::row_major>::from_shape({5, 3, 4, 6});

        using strides_type = std::vector<std::ptrdiff_t>;
        auto s1 = view(a, 1, 1, xt::all(), xt::all()).strides();
        strides_type s1e = {6, 1};
        EXPECT_EQ(s1, s1e);

        auto s2 = view(a, 1, xt::all(), xt::all(), 1).strides();
        strides_type s2e = {24, 6};
        EXPECT_EQ(s2, s2e);

        auto s3 = view(a, 1, xt::all(), 1, xt::newaxis(), xt::newaxis(), xt::all()).strides();
        strides_type s3e = {24, 0, 0, 1};
        EXPECT_EQ(s3, s3e);

        auto s4 = view(a, xt::range(0, 1, 2), 1, 0, xt::all(), xt::newaxis()).strides();
        strides_type s4e = {0, 1, 0};
        EXPECT_EQ(s4, s4e);

        auto s4x = view(a, xt::range(0, 5, 2), 1, 0, xt::all(), xt::newaxis()).strides();
        strides_type s4xe = {72 * 2, 1, 0};
        EXPECT_EQ(s4x, s4xe);

        auto s5 = view(a, xt::all(), 1).strides();
        strides_type s5e = {72, 6, 1};
        EXPECT_EQ(s5, s5e);

        auto s6 = view(a, xt::all(), 1, 1, xt::newaxis(), xt::all()).strides();
        strides_type s6e = {72, 0, 1};
        EXPECT_EQ(s6, s6e);

        auto s7 = view(a, xt::all(), 1, xt::newaxis(), xt::all()).strides();
        strides_type s7e = {72, 0, 6, 1};
        EXPECT_EQ(s7, s7e);
    }

    TEST(xview, strides_issue_2228)
    {
        {
            xt::xtensor<uint8_t, 2> aTensor = xt::zeros<uint8_t>({4, 3});
            auto aView = xt::view(aTensor, xt::newaxis(), xt::newaxis(), xt::newaxis(), xt::all(), xt::all());
            uint8_t vIndex = 0;
            for (size_t vRow = 0; vRow < 3; ++vRow) {
                for (size_t vCol = 0; vCol < 3; ++vCol) {
                    aView(size_t{0}, size_t{0}, size_t{0}, vRow, vCol) = vIndex;
                    EXPECT_EQ(aTensor(vRow, vCol), vIndex);
                    ++vIndex;
                }
            }
        }

        {
            xt::xtensor<bool, 2> aTensor = xt::zeros<bool>({4, 3});
            auto aView = xt::view(aTensor, xt::newaxis(), xt::newaxis(), xt::newaxis(), xt::all(), xt::all());
            for (size_t vRow = 0; vRow < 4; ++vRow) {
                for (size_t vCol = 0; vCol < 3; ++vCol) {
                    aView(size_t{0}, size_t{0}, size_t{0}, vRow, vCol) = true;
                }
            }
            EXPECT_EQ(aTensor, aView);
        }
    }

    TEST(xview, to_scalar)
    {
        std::array<std::size_t, 3> sh{2,2,2};
        xtensor<double, 3> a(sh, 123);
        xtensor_fixed<double, xshape<2, 2, 2>> af = a;
        xarray<double> b = a;

        auto av = view(a, 1, 1);
        const auto av1 = view(a, 1, 1, 0);
        const double& ad1 = av1;
        EXPECT_EQ(ad1, av1());

        bool ax = is_xscalar<std::decay_t<decltype(av)>>::value;
        EXPECT_FALSE(ax);
        ax = is_xscalar<std::decay_t<decltype(av1)>>::value;
        EXPECT_TRUE(ax);
        auto bv = view(b, 1, 1, 1);
        ax = is_xscalar<decltype(bv)>::value;
        EXPECT_FALSE(ax);

        auto afv = view(af, 1, 1);
        auto afv1 = view(af, 1, 1, 0);

        double& afd1 = view(af, 1, 1, 0);
        EXPECT_EQ(afd1, af(1, 1, 0));
        ax = is_xscalar<decltype(afv)>::value;
        EXPECT_FALSE(ax);
        ax = is_xscalar<decltype(afv1)>::value;
        EXPECT_TRUE(ax);

        const xtensor<double, 2> ac = {{1,2}, {3,4}};
        double a1 = view(ac, 0, 0);
        const double& a2 = view(ac, 0, 0);

        EXPECT_EQ(a1, a2);

        double conv = av1;
        double conv1 = afv1;
        EXPECT_EQ(conv, conv1);
    }

    template <class V, class A>
    inline void test_view_iter(V& v, A& exp)
    {
        auto iter_expv1 = exp.begin();
        for (auto iter = v.begin(); iter != v.end(); ++iter)
        {
            EXPECT_EQ(*iter, *iter_expv1);
            ++iter_expv1;
        }

        auto citer_expv1 = exp.template begin<layout_type::column_major>();
        for (auto iter = v.template begin<layout_type::column_major>();
            iter != v.template end<layout_type::column_major>(); ++iter)
        {
            EXPECT_EQ(*iter, *citer_expv1);
            ++citer_expv1;
        }

        auto riter_expv1 = exp.rbegin();
        for (auto iter = v.rbegin(); iter != v.rend(); ++iter)
        {
            EXPECT_EQ(*iter, *riter_expv1);
            ++riter_expv1;
        }

        auto rciter_expv1 = exp.template rbegin<layout_type::column_major>();
        for (auto iter = v.template rbegin<layout_type::column_major>();
             iter != v.template rend<layout_type::column_major>(); ++iter)
        {
            EXPECT_EQ(*iter, *rciter_expv1);
            ++rciter_expv1;
        }
    }

    TEST(xview, random_stepper)
    {
        xt::xarray<double, layout_type::row_major> data = xt::arange(0, 100);
        data.reshape({5, 5, 4});
        xt::xarray<double> x = data;

        xt::xarray<double> expected;
        if (XTENSOR_DEFAULT_TRAVERSAL == layout_type::row_major)
        {
           expected = {
               0, 1, 2, 3,
               20, 21, 22, 23,
               40, 41, 42, 43,
               60, 61, 62, 63,
               80, 81, 82, 83
           };
        }
        else
        {
           expected = {
               0, 1, 2, 3, 4,
               25, 26, 27, 28, 29,
               50, 51, 52, 53, 54,
               75, 76, 77, 78, 79
           };
        }
        auto v = xt::view(x, all(), 0);

        auto it1 = v.begin();
        auto it3 = v.rbegin();

        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            std::ptrdiff_t ix = static_cast<std::ptrdiff_t>(i);
            EXPECT_EQ(*(it1 + ix), expected[i]);
            EXPECT_EQ(*(it3 + ix), expected[expected.size() - 1 - i]);
        }
    }

    TEST(xview, keep_slice)
    {
        xtensor<double, 3, layout_type::row_major> a = {{{ 1, 2, 3, 4},
                                                         { 5, 6, 7, 8}},
                                                        {{ 9,10,11,12},
                                                         {13,14,15,16}},
                                                        {{17,18,19,20},
                                                         {21,22,23,24}}};

        auto v1 = xt::view(a, keep(1), keep(0, 1), keep(0, 3));
        xtensor<double, 3> exp_v1 = {{{9, 12}, {13, 16}}};

        EXPECT_EQ(v1, exp_v1);

        test_view_iter(v1, exp_v1);

        auto v2 = xt::view(a, keep(1), xt::all(), xt::range(0, xt::xnone(), 3));
        EXPECT_EQ(v2, v1);
        EXPECT_EQ(v2, exp_v1);

        auto v3 = xt::view(a, keep(1), keep(1, 1, 1, 1), keep(0, 3));
        xtensor<double, 3> exp_v3 = {{{13, 16}, {13, 16}, {13, 16}, {13, 16}}};
        EXPECT_EQ(v3, exp_v3);

        test_view_iter(v3, exp_v3);

        auto v4 = xt::view(a, keep(0, 2), keep(0));
        xtensor<double, 3> exp_v4 = {{{  1.,   2.,   3.,   4.}},
                                     {{ 17.,  18.,  19.,  20.}}};
        EXPECT_EQ(v4, exp_v4);

        v4(0, 0) = 123;
        v4(1, 0) = 123;
        EXPECT_EQ(a(0, 0, 0), 123);
        EXPECT_EQ(a(1, 0, 0), 123);

        v3(0, 2, 1) = 1000;
        EXPECT_EQ(a(1, 1, 3), 1000);

        bool b = detail::is_strided_view<decltype(a), xkeep_slice<int>, int>::value;
        EXPECT_FALSE(b);
        b = detail::is_strided_view<decltype(a), xrange<int>, xrange<int>, int>::value;
        EXPECT_TRUE(b);
    }

    TEST(xview, keep_negative)
    {
        xtensor<double, 3, layout_type::row_major> a = {{{ 1, 2, 3, 4},
                                                         { 5, 6, 7, 8}},
                                                        {{ 9,10,11,12},
                                                         {13,14,15,16}},
                                                        {{17,18,19,20},
                                                         {21,22,23,24}}};

        auto v1 = xt::view(a, keep(-2), keep(-0, -1), keep(0, -1));
        xtensor<double, 3> exp_v1 = {{{9, 12}, {13, 16}}};
        EXPECT_EQ(v1, exp_v1);
    }

    TEST(xview, drop_slice)
    {
        xtensor<double, 3, layout_type::row_major> a = {{{ 1, 2, 3, 4},
                                                         { 5, 6, 7, 8}},
                                                        {{ 9,10,11,12},
                                                         {13,14,15,16}},
                                                        {{17,18,19,20},
                                                         {21,22,23,24}}};

        auto v1 = xt::view(a, drop(0, 2), keep(0, 1), drop(1, 2));
        xtensor<double, 3> exp_v1 = { { { 9, 12 },{ 13, 16 } } };
        EXPECT_EQ(v1, exp_v1);
        test_view_iter(v1, exp_v1);

        auto v2 = xt::view(a, drop(0, 2), xt::all(), xt::range(0, xt::xnone(), 3));
        EXPECT_EQ(v2, v1);
        EXPECT_EQ(v2, exp_v1);

        auto v4 = xt::view(a, drop(1), drop(1));
        xtensor<double, 3> exp_v4 = {{{ 1.,   2.,   3.,   4.}},
                                     {{17.,  18.,  19.,  20.}}};
        EXPECT_EQ(v4, exp_v4);

        v4(0, 0) = 123;
        v4(1, 0) = 123;
        EXPECT_EQ(a(0, 0, 0), 123);
        EXPECT_EQ(a(1, 0, 0), 123);

        bool b = detail::is_strided_view<decltype(a), xkeep_slice<int>, int>::value;
        EXPECT_FALSE(b);
        b = detail::is_strided_view<decltype(a), xrange<int>, xrange<int>, int>::value;
        EXPECT_TRUE(b);

        std::vector<size_t> empty;
        auto v5 = xt::view(a, drop(empty), drop(empty), drop(empty));
        v5(1, 1, 1) = 456;
        EXPECT_EQ(v5, a);
        EXPECT_EQ(a(1, 1, 1), 456);
    }

    TEST(xview, drop_negative)
    {
        xtensor<double, 3, layout_type::row_major> a = {{{ 1, 2, 3, 4},
                                                         { 5, 6, 7, 8}},
                                                        {{ 9,10,11,12},
                                                         {13,14,15,16}},
                                                        {{17,18,19,20},
                                                         {21,22,23,24}}};

        //auto v1 = xt::view(a, keep(-2), keep(-0, -1), keep(0, -1));
        auto v1 = xt::view(a, drop(-3, -1), keep(0, 1), drop(-3, -2));
        xtensor<double, 3> exp_v1 = { { { 9, 12 },{ 13, 16 } } };
        EXPECT_EQ(v1, exp_v1);
    }

    TEST(xview, const_keep_drop_slice)
    {
        xt::xtensor<double, 1> xs = xt::arange<double>(10);
        const auto kidx = xt::keep(0, 3, 5);
        const auto didx = xt::drop(1, 2, 4, 6, 7, 8, 9);
        auto kv = xt::view(xs, kidx);
        auto dv = xt::view(xs, didx);
        xt::xtensor<double, 1> kres = kv;
        xt::xtensor<double, 1> dres = dv;
        xt::xtensor<double, 1> expected = { 0., 3., 5. };
        EXPECT_EQ(kres, expected);
        EXPECT_EQ(dres, expected);
    }

    TEST(xview, mixed_types)
    {
        xt::xarray<std::uint8_t> input;
        xt::xarray<float> output;
        input.resize({ { 50,16,16,3 } });
        output.resize({ { 50,16,16,3 } });

        input.fill(std::uint8_t(1));
        output.fill(float(2.));
        for (int i = 0; i<50; ++i)
        {
            auto in_view = xt::view(input, i);
            auto out_view = xt::view(output, i);
            out_view = in_view;
        }

        EXPECT_EQ(output(0, 5, 5, 2), 1.f);
    }

    TEST(xview, where_operation)
    {
        xt::xtensor<size_t, 2> I = {{0, 0}, {1, 1}, {2, 2}};
        auto col = xt::view(I, xt::all(), 0);
        auto idx = xt::where(xt::equal(col, size_t(0)));

        std::vector<std::size_t> exp_idx = {0};
        EXPECT_EQ(idx[0], exp_idx);

        auto idx2 = xt::where(col > size_t(0));
        std::vector<std::size_t> exp_idx2 = {1, 2};
        EXPECT_EQ(idx2.size(), 1u);
        EXPECT_EQ(idx2[0], exp_idx2);
    }

    TEST(xview, contiguous)
    {
        using xtes = xt::xtensor<double, 4, layout_type::row_major>;
        using xarr = xt::xarray<double, layout_type::row_major>;
        using xfix = xt::xtensor_fixed<double, xshape<3, 4, 2, 5>, layout_type::row_major>;

        using ctes = xt::xtensor<double, 4, layout_type::column_major>;
        using carr = xt::xarray<double, layout_type::column_major>;
        using cfix = xt::xtensor_fixed<double, xshape<3, 4, 2, 5>, layout_type::column_major>;

        EXPECT_TRUE((detail::is_contiguous_view<xtes, xall<int>, xall<int>, xall<int>>()));
        EXPECT_TRUE((detail::is_contiguous_view<xarr, xall<int>, xall<int>, xall<int>>()));
        EXPECT_TRUE((detail::is_contiguous_view<xfix, xall<int>, xall<int>, xall<int>>()));
        EXPECT_TRUE((detail::is_contiguous_view<xtes, int, int, xall<int>>()));
        EXPECT_TRUE((detail::is_contiguous_view<xtes, int, xall<int>, xall<int>>()));
        EXPECT_TRUE((detail::is_contiguous_view<xtes, int, xall<int>, xall<int>, xall<int>>()));
        EXPECT_TRUE((detail::is_contiguous_view<xtes, int, int, xrange<int>>()));
        EXPECT_TRUE((detail::is_contiguous_view<xtes, int, xrange<int>>()));
        EXPECT_FALSE((detail::is_contiguous_view<xtes, int, xrange<int>, int>()));

        EXPECT_TRUE((detail::is_contiguous_view<ctes, xall<int>, xall<int>, xall<int>, xall<int>>()));
        EXPECT_FALSE((detail::is_contiguous_view<ctes, int, int, xall<int>>()));
        EXPECT_FALSE((detail::is_contiguous_view<ctes, int, xall<int>, xall<int>>()));
        EXPECT_FALSE((detail::is_contiguous_view<ctes, int, xall<int>, xall<int>, xall<int>>()));
        EXPECT_FALSE((detail::is_contiguous_view<ctes, int, int, xrange<int>>()));
        EXPECT_FALSE((detail::is_contiguous_view<ctes, int, xrange<int>>()));
        EXPECT_FALSE((detail::is_contiguous_view<ctes, int, xrange<int>, int>()));

        EXPECT_TRUE((detail::is_contiguous_view<ctes, xall<int>, xall<int>, int, int>()));
        EXPECT_TRUE((detail::is_contiguous_view<cfix, xall<int>, xall<int>, int, int>()));
        EXPECT_FALSE((detail::is_contiguous_view<xarr, xall<int>, xall<int>, int, int>()));
        EXPECT_TRUE((detail::is_contiguous_view<ctes, xall<int>, xall<int>, xrange<int>, int>()));
        EXPECT_FALSE((detail::is_contiguous_view<ctes, xall<int>, xrange<int>, xrange<int>, int>()));
        EXPECT_FALSE((detail::is_contiguous_view<ctes, xall<int>, xrange<int>, xall<int>, int>()));
        EXPECT_FALSE((detail::is_contiguous_view<ctes, xall<int>, xrange<int>, xrange<int>, int>()));
        EXPECT_FALSE((detail::is_contiguous_view<ctes, xall<int>, xstepped_range<int>, int, int>()));
    }

    TEST(xview, sequence_view)
    {
        using vector_type = std::vector<int>;
        using array_type = std::array<int, 7>;
        auto a = vector_type({0,1,2,3,4,5,6});
        auto b = array_type({0,1,2,3,4,5,6});

        auto va = sequence_view<vector_type, 3>(a);
        auto vb = sequence_view<array_type, 3>(b);

        EXPECT_EQ(va[0], a[3]);
        EXPECT_EQ(va[1], a[4]);
        EXPECT_EQ(va.end(), a.end());
        EXPECT_TRUE(std::equal(a.begin() + 3, a.end(), va.begin()));
        EXPECT_EQ(a.size() - 3, va.size());

        EXPECT_EQ(vb[0], b[3]);
        EXPECT_EQ(vb[1], b[4]);
        EXPECT_EQ(*(vb.end() - 1), *(b.end() - 1));
        EXPECT_TRUE(std::equal(b.begin() + 3, b.end(), vb.begin()));
        EXPECT_EQ(b.size() - 3, vb.size());

        vector_type cvta = va;
        std::array<int, 4> cvtb = vb;
        vector_type cvta_expected = { 3, 4, 5, 6 };
        std::array<int, 4> cvtb_expected = { 3, 4, 5, 6};

        EXPECT_EQ(cvta, cvta_expected);
        EXPECT_EQ(cvtb, cvtb_expected);

        auto vae = sequence_view<vector_type, 3, 5>(a);
        auto vbe = sequence_view<array_type, 3, 5>(b);

        EXPECT_EQ(vae[0], b[3]);
        EXPECT_EQ(vae[1], b[4]);
        EXPECT_EQ(vae.back(), b[4]);
        EXPECT_EQ(*vae.end(), *(a.end() - 2));
        EXPECT_TRUE(std::equal(a.begin() + 3, a.end() - 1, vae.begin()));
        EXPECT_EQ(std::size_t(2), vae.size());

        auto r_iter = vae.rbegin();
        EXPECT_EQ(static_cast<std::size_t>(std::distance(a.rbegin(), a.rend())), a.size());
        EXPECT_EQ(static_cast<std::size_t>(std::distance(r_iter, vae.rend())), vae.size());

        for (std::size_t i = 0; i < vae.size(); ++i)
        {
            EXPECT_EQ(*r_iter, b[4 - i]);
            ++r_iter;
        }
        EXPECT_EQ(r_iter, vae.rend());

        EXPECT_EQ(vbe[0], b[3]);
        EXPECT_EQ(vbe[1], b[4]);
        EXPECT_EQ(vbe.back(), b[4]);
        EXPECT_EQ(*vbe.end(), *(a.end() - 2));
        EXPECT_TRUE(std::equal(a.begin() + 3, a.end() - 1, vbe.begin()));
        EXPECT_EQ(std::size_t(2), vbe.size());

        auto rb_iter = vbe.rbegin();
        EXPECT_EQ(static_cast<std::size_t>(std::distance(a.rbegin(), a.rend())), a.size());
        EXPECT_EQ(static_cast<std::size_t>(std::distance(rb_iter, vbe.rend())), vbe.size());

        for (std::size_t i = 0; i < vbe.size(); ++i)
        {
            EXPECT_EQ(*rb_iter, b[4 - i]);
            ++rb_iter;
        }
        EXPECT_EQ(rb_iter, vbe.rend());
    }

    TEST(xview, data_offset)
    {
        xt::xtensor<double, 6> ax = xt::random::rand<double>({3, 3, 3, 3, 3, 3});

        auto do1 = xt::view(ax, 1, 1, newaxis(), 1).data_offset();
        auto dos = xt::strided_view(ax, {1, 1, newaxis(), 1}).data_offset();

        EXPECT_EQ(do1, dos);
        EXPECT_EQ(ax.storage()[do1], ax(1, 1, 1, 0, 0, 0));
        auto doe = ax.strides()[0] * 1 + ax.strides()[1] * 1 + ax.strides()[2] * 1;
        EXPECT_EQ(static_cast<std::size_t>(doe), do1);

        auto do2 = xt::view(ax, 1, 2, newaxis(), range(1, 2), range(2, 2, 4), all()).data_offset();
        auto dos2 = xt::strided_view(ax, {1, 2, newaxis(), range(1, 2), range(2, 2, 4), all()}).data_offset();
        EXPECT_EQ(do2, dos2);
        auto doe2 = ax.strides()[0] * 1 + ax.strides()[1] * 2 + ax.strides()[2] * 1 + ax.strides()[3] * 2;
        EXPECT_EQ(static_cast<std::size_t>(doe2), do2);
    }

    TEST(xview, view_simd_test)
    {
        if (XTENSOR_DEFAULT_LAYOUT == layout_type::row_major)
        {
            xt::xarray<double> a = xt::arange<double>(3 * 4 * 5);
            a.reshape({3, 4, 5});
            xt::xarray<double> b = xt::arange<double>(4 * 5);
            b.reshape({4, 5});
            xt::xarray<double> c = xt::broadcast(b, {3, 4, 5});
            noalias(view(a, 1, all(), all())) = b;
            noalias(view(a, 2, all(), all())) = view(a, 0, all(), all());
            EXPECT_EQ(a, c);

            auto vxt = view(a, 1, all(), all());
            auto vxa = view(xt::arange<double>(100), range(0, 10));

            using assign_traits = xassign_traits<decltype(vxt), decltype(b)>;

    #if XTENSOR_USE_XSIMD
            EXPECT_TRUE(assign_traits::simd_linear_assign());
    #endif

            using assign_traits2 = xassign_traits<decltype(b), decltype(vxa)>;

    #if XTENSOR_USE_XSIMD
            EXPECT_FALSE(assign_traits2::simd_linear_assign());
    #endif
        }
    }

    xt::xtensor<double,2> view_assign_func(const xt::xtensor<double, 2>& a, int idx)
    {
        xt::xtensor<double, 2> b;
        switch(idx)
        {
            case 1: b = xt::view(a,      xt::all(), xt::range(0, 1)); break;
            case 2: b.assign(xt::view(a, xt::all(), xt::range(0, 1))); break;
            case 3: b = xt::view(a,      xt::all(), xt::range(0, 2)); break;
            case 4: b = 2.*xt::view(a,   xt::all(), xt::range(0, 1)); break;
            case 5: b = xt::view(2.*a,   xt::all(), xt::range(0, 1)); break;
            default: b = a; break;
        }
        return b;
    }

    TEST(xview, assign)
    {
        xt::xtensor<double, 2> input = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xt::xtensor<double, 2> exp1 = {{1, 4, 7}};
        exp1.reshape({3, 1});

        xt::xtensor<double, 2> exp2 = {{1, 2}, {4, 5}, {7, 8}};

        EXPECT_EQ(view_assign_func(input, 1), exp1);
        EXPECT_EQ(view_assign_func(input, 2), exp1);
        EXPECT_EQ(view_assign_func(input, 3), exp2);
        EXPECT_EQ(view_assign_func(input, 4), 2 * exp1);
        EXPECT_EQ(view_assign_func(input, 5), 2 * exp1);
    }

    TEST(xview, view_on_strided_view)
    {
        // Compilation test only
        xt::xarray<float> original = xt::xarray<float>::from_shape({ 3, 2, 5 });
        original.fill(float(0.));
        auto str_view = xt::strided_view(original, { 1, xt::ellipsis() }); //i is an int
        auto result = xt::view(str_view, xt::all(), xt::all());
        EXPECT_EQ(result(0), 0.f);
    }

    TEST(xview, assign_scalar_to_non_contiguous_view)
    {
        // Compilation test only
        xt::xtensor<int, 2> arr = xt::ones<int>({10, 10});
        auto v = xt::view(arr, xt::keep(0, -1), xt::all());
        v = 0;
    }

    TEST(xview, assign_scalar_to_contiguous_view_of_view)
    {
        xt::xarray<double> arr
          {{0., 1., 2.},
           {3., 4., 5.},
           {6., 7., 8.}};
        auto vv = xt::view(xt::view(arr, 1), 0);
        vv = 100.0;
        EXPECT_EQ(arr(1, 0), 100.0);
    }

    TEST(xview, keep_assign)
    {
        xt::xtensor<int, 2> a = { {1, 2, 3, 4},
                                  {5, 6, 7, 8},
                                  {9, 10, 11, 12},
                                  {13, 14, 15, 16} };

        auto v = xt::view(xt::view(a, xt::all(), xt::keep(0, 1)), xt::all(), 0);
        xt::xtensor<int, 1> res = v;

        xt::xtensor<int, 1> exp = { 1, 5, 9, 13 };
        EXPECT_EQ(res, exp);
    }

    TEST(xview, view_view_assignment)
    {
        xt::xtensor<double, 4> a = xt::random::rand<double>({5, 5, 5, 5});

        std::size_t sa = 0, sb = 2;
        auto start = xt::view(a, 1); //3D
        auto res = xt::view(start, xt::all(), xt::all(), xt::keep(sa, sb));

        auto expres = xt::exp(res);

        xt::xarray<double> assgment = expres;
        auto expv = xt::exp(xt::view(a, 1, xt::all(), xt::all(), xt::range(0, 3, 2)));
        EXPECT_EQ(assgment, expv);
    }

    TEST(xview, view_on_fixed)
    {
        xt::xtensor_fixed<double, xt::xshape<3>> a{1./8, 1, -1./8};
        auto v = xt::view(a, xt::all(), xt::newaxis());
        EXPECT_EQ(v.dimension(), 2u);
        EXPECT_EQ(v.shape(), (std::array<std::size_t, 2>{3, 1}));

        auto b = a * xt::view(a, xt::all(), xt::newaxis());

        xt::xarray<double> exp = {{ 0.015625,  0.125   , -0.015625},
                                  { 0.125   ,  1.      , -0.125   },
                                  {-0.015625, -0.125   ,  0.015625}};

        EXPECT_EQ(b, exp);
    }

    TEST(xview, periodic)
    {
        xt::xtensor<size_t,2> a = {{0,1,2}, {3,4,5}};
        xt::xtensor<size_t,2> b = {{0,1,2}, {30,40,50}};
        auto view = xt::view(a, xt::keep(1), xt::all());
        view.periodic(-1,3) = 30;
        view.periodic(-1,4) = 40;
        view.periodic(-1,5) = 50;
        EXPECT_EQ(a, b);
    }

    TEST(xview, in_bounds)
    {
        xt::xtensor<size_t,2> a = {{0,1,2}, {3,4,5}};
        auto view = xt::view(a, xt::keep(1), xt::all());
        EXPECT_TRUE(view.in_bounds(0,0) == true);
        EXPECT_TRUE(view.in_bounds(2,0) == false);
    }

    TEST(xview, strides_compute_out_of_bounds)
    {
        // check that the compute_strides_impl does not access `a` strides out
        // of bound! Can be observed with Valgrind or MSVC debug
        xt::xtensor<double, 1> a = {1};
        auto v1 = xt::view(a, xt::all(), xt::newaxis());
        EXPECT_EQ(v1.dimension(), 2ul);
        EXPECT_EQ(v1.strides().size(), 2ul);
        EXPECT_EQ(v1.strides()[0], 0);
        EXPECT_EQ(v1.strides()[1], 0);
    }

    template <class E>
    auto transform(E& x) {
      x += 2;
    }

    TEST(xview, nontrivial_strides)
    {
        using farray = xt::xtensor<float, 2, xt::layout_type::column_major>;
        const farray x_orig = xt::random::randn<float>({2, 2});

        using namespace xt::placeholders;

        farray x_view = x_orig;

        auto x1 = xt::view(x_view, xt::all(), xt::range(_, 1));
        auto x2 = xt::view(x_view, xt::all(), xt::range(1, _));

        transform(x1);
        transform(x2);
        EXPECT_TRUE(xt::allclose(x_view, float(2) + x_orig));

        for (auto it = x1.begin(); it != x1.end(); ++it)
        {
            *it += 5;
        }
        for (auto it = x2.begin(); it != x2.end(); ++it)
        {
            *it += 5;
        }
        EXPECT_TRUE(xt::allclose(x_view, float(7) + x_orig));
    }

    TEST(xview, element)
    {
        xarray<int> a = { {1, 2, 3}, {4, 5, 6} };
        auto v = view(a, 0);
        std::array<std::size_t, 2> idx = { 0, 1 };
        int res = v.element(idx.cbegin(), idx.cend());
        EXPECT_EQ(res, 2);
    }

    TEST(xview, view_reshape_view)
    {
        xtensor<int, 1> a = { 0, 1, 2 };
        xtensor<int, 1> b = { 2, 3, 4 };
        auto tmp = reshape_view(a, {3});
        auto res = view(std::move(tmp), xt::all());
        noalias(view(reshape_view(a, {3}), xt::all())) = b;
        EXPECT_EQ(tmp, res);
    }

    TEST(xview, view_on_bool)
    {
        xt::xarray<bool> a { { false, false }, { false, false } };
        xt::xarray<bool> b { {  true,  true }, {  true,  true } };
        xt::view( a, 0 ) = xt::view( b, 0 );
        EXPECT_TRUE(a(0, 0));
        EXPECT_TRUE(a(0, 1));
        EXPECT_FALSE(a(1, 0));
        EXPECT_FALSE(a(1, 1));
    }

    TEST(xview, first_rows_on_2dim_xarray)
    {
        xt::xarray<int> array{
            { 1, 2 },
            { 3, 4 },
        };

        const auto first_row = xt::row(array, 0);
        const auto second_row = xt::row(array, 1);

        EXPECT_EQ(first_row(0), 1);
        EXPECT_EQ(first_row(1), 2);
        EXPECT_EQ(second_row(0), 3);
        EXPECT_EQ(second_row(1), 4);
    }

    TEST(xview, last_rows_on_2dim_xarray)
    {
        xt::xarray<int> array{
            { 1, 2 },
            { 3, 4 },
            { 5, 6 },
        };

        const auto last_row = xt::row(array, -1);
        const auto second_last_row = xt::row(array, -2);

        EXPECT_EQ(last_row(0), 5);
        EXPECT_EQ(last_row(1), 6);
        EXPECT_EQ(second_last_row(0), 3);
        EXPECT_EQ(second_last_row(1), 4);
    }

    TEST(xiew, row_on_2dim_xtensor)
    {
        xt::xtensor<int, 2> tensor{
            { 1, 2 },
            { 3, 4 },
        };

        std::cout << tensor.shape().size() << std::endl;

        const auto row0 = xt::row(tensor, 0);
        const auto row1 = xt::row(tensor, 1);

        EXPECT_EQ(row0(0), 1);
        EXPECT_EQ(row0(1), 2);
        EXPECT_EQ(row1(0), 3);
        EXPECT_EQ(row1(1), 4);
    }

    TEST(xiew, row_on_2dim_xtensor_fixed)
    {
        xt::xtensor_fixed<int, xshape<2, 2>> tensor_fixed{
            { 1, 2 },
            { 3, 4 },
        };

        const auto row0 = xt::row(tensor_fixed, 0);
        const auto row1 = xt::row(tensor_fixed, 1);

        EXPECT_EQ(row0(0), 1);
        EXPECT_EQ(row0(1), 2);
        EXPECT_EQ(row1(0), 3);
        EXPECT_EQ(row1(1), 4);
    }

    TEST(xview, row_on_3dim_array)
    {
        xt::xarray<int> arr{
            { { 1, 2 }, { 3, 4 } },
            { { 5, 6 }, { 7, 8 } },
        };

        XT_ASSERT_THROW(
            const auto row = xt::row(arr, 0),
            std::invalid_argument
        );
    }

    TEST(xview, first_cols_on_2dim_xarray)
    {
        xt::xarray<int> array{
            { 1, 2 },
            { 3, 4 },
        };

        const auto first_col = xt::col(array, 0);
        const auto second_col = xt::col(array, 1);

        EXPECT_EQ(first_col(0), 1);
        EXPECT_EQ(first_col(1), 3);
        EXPECT_EQ(second_col(0), 2);
        EXPECT_EQ(second_col(1), 4);
    }

    TEST(xview, last_cols_on_2dim_xarray)
    {
        xt::xarray<int> array{
            { 1, 2, 3 },
            { 4, 5, 6 },
        };

        const auto last_col = xt::col(array, -1);
        const auto second_last_col = xt::col(array, -2);

        EXPECT_EQ(last_col(0), 3);
        EXPECT_EQ(last_col(1), 6);
        EXPECT_EQ(second_last_col(0), 2);
        EXPECT_EQ(second_last_col(1), 5);
    }

    TEST(xview, col_on_2dim_xtensor)
    {
        xt::xtensor<int, 2> tensor{
            { 1, 2 },
            { 3, 4 },
        };

        const auto col0 = xt::col(tensor, 0);
        const auto col1 = xt::col(tensor, 1);

        EXPECT_EQ(col0(0), 1);
        EXPECT_EQ(col0(1), 3);
        EXPECT_EQ(col1(0), 2);
        EXPECT_EQ(col1(1), 4);
    }

    TEST(xview, col_on_2dim_xtensor_fixed)
    {
        xt::xtensor_fixed<int, xshape<2, 2>> tensor_fixed{
            { 1, 2 },
            { 3, 4 },
        };

        const auto col0 = xt::col(tensor_fixed, 0);
        const auto col1 = xt::col(tensor_fixed, 1);

        EXPECT_EQ(col0(0), 1);
        EXPECT_EQ(col0(1), 3);
        EXPECT_EQ(col1(0), 2);
        EXPECT_EQ(col1(1), 4);
    }

    TEST(xview, col_on_3dim_array)
    {
        xt::xarray<int> arr{
            { { 1, 2 }, { 3, 4 } },
            { { 5, 6 }, { 7, 8 } },
        };

        XT_ASSERT_THROW(
            const auto col = xt::col(arr, 0),
            std::invalid_argument
        );
    }

    // This code should not compile!
    //TEST(xview, col_on_3dim_xtensor)
    //{
    //    xt::xtensor<int, 3> tensor;
    //    xt::row(tensor, 0);
    //    xt::col(tensor, 0);
    //}
}
