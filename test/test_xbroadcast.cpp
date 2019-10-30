/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xbroadcast.hpp"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "test_common_macros.hpp"

namespace xt
{
    TEST(xbroadcast, broadcast)
    {
        xarray<double> m1
          {{1, 2, 3},
           {4, 5, 6}};

        auto m1_broadcast = broadcast(m1, {1, 2, 3});
        ASSERT_EQ(1.0, m1_broadcast(0, 0, 0));
        ASSERT_EQ(4.0, m1_broadcast(0, 1, 0));
        ASSERT_EQ(5.0, m1_broadcast(0, 1, 1));
        ASSERT_EQ(m1_broadcast.layout(), m1.layout());
        EXPECT_ANY_THROW(m1_broadcast.at(0, 0, 0, 0));
        EXPECT_ANY_THROW(m1_broadcast.at(10, 10, 10));

        auto shape = std::vector<std::size_t>{1, 2, 3};
        auto m1_broadcast2 = broadcast(m1, shape);
        ASSERT_EQ(1.0, m1_broadcast2(0, 0, 0));
        ASSERT_EQ(4.0, m1_broadcast2(0, 1, 0));
        ASSERT_EQ(5.0, m1_broadcast2(0, 1, 1));

        xarray<double> m1_assigned = m1_broadcast;
        ASSERT_EQ(5.0, m1_assigned(0, 1, 1));
    }

    TEST(xbroadcast, element)
    {
        xarray<double> m1
          {{1, 2, 3},
           {4, 5, 6}};

        auto m1_broadcast = broadcast(m1, {4, 2, 3});

        // access with the right number of arguments
        std::array<std::size_t, 3> index1 = {0, 1, 1};
        ASSERT_EQ(5.0, m1_broadcast.element(index1.begin(), index1.end()));
        // too many arguments = using the last ones only
        std::array<std::size_t, 4> index3 = {4, 0, 1, 1};
        ASSERT_EQ(5.0, m1_broadcast.element(index3.begin(), index3.end()));
    }

    TEST(xbroadcast, indexed_access)
    {
        xarray<double> m1
          {{ 1, 2, 3 },
           { 4, 5, 6 }};

        auto m1_broadcast = broadcast(m1, { 4, 2, 3 });
        std::array<std::size_t, 3> index1 = { 0, 1, 1 };
        ASSERT_EQ(5.0, m1_broadcast[index1]);
        ASSERT_EQ(5.0, (m1_broadcast[{0, 1, 1}]));
        std::array<std::size_t, 4> index3 = { 4, 0, 1, 1 };
        ASSERT_EQ(5.0, m1_broadcast[index3]);
        ASSERT_EQ(5.0, (m1_broadcast[{4, 0, 1, 1}]));
    }

    TEST(xbroadcast, shape_forwarding)
    {
        std::array<std::size_t, 2> bc_shape{3, 3};
        auto m1_broadcast = broadcast(123, bc_shape);
        ASSERT_EQ(m1_broadcast(0, 0), 123);
    }

    TEST(xbroadcast, xiterator)
    {
        xarray<int> m1 = {1, 2, 3};
        auto m1_broadcast = broadcast(m1, {2, 3});
        size_t nb_iter = 3;
        using shape_type = std::vector<size_t>;

        // broadcast_iterator
        {
            auto iter = m1_broadcast.template begin<layout_type::row_major>();
            auto iter_end = m1_broadcast.template end<layout_type::row_major>();
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(1, *iter);
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }

        // shaped_xiterator
        {
            shape_type shape = {2, 2, 3};
            auto iter = m1_broadcast.template begin<layout_type::row_major>(shape);
            auto iter_end = m1_broadcast.template end<layout_type::row_major>(shape);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(1, *iter);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }
    }

    TEST(xbroadcast, reverse_xiterator)
    {
        xarray<int> m1 = {1, 2, 3};
        auto m1_broadcast = broadcast(m1, {2, 3});
        size_t nb_iter = 3;
        using shape_type = std::vector<size_t>;

        // reverse_broadcast_iterator
        {
            auto iter = m1_broadcast.template rbegin<layout_type::row_major>();
            auto iter_end = m1_broadcast.template rend<layout_type::row_major>();
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(3, *iter);
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }

        // reverse_shaped_xiterator
        {
            shape_type shape = {2, 2, 3};
            auto iter = m1_broadcast.template rbegin<layout_type::row_major>(shape);
            auto iter_end = m1_broadcast.template rend<layout_type::row_major>(shape);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(3, *iter);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }
    }

    TEST(xbroadcast, same_shape)
    {
        xt::xarray<double, xt::layout_type::column_major> arr
            {{ 0.0,  1.0,  2.0},
             {10.0, 11.0, 12.0}};

        auto t = xt::broadcast(arr, {2, 3});

        xt::xarray<double, xt::layout_type::column_major> cm_arr = t;
        xt::xarray<double, xt::layout_type::row_major> rm_arr = t;
        
        EXPECT_TRUE(xt::allclose(rm_arr, cm_arr));

        EXPECT_EQ(cm_arr(0, 0),  0.0);
        EXPECT_EQ(cm_arr(0, 1),  1.0);
        EXPECT_EQ(cm_arr(0, 2),  2.0);
        EXPECT_EQ(cm_arr(1, 0), 10.0);
        EXPECT_EQ(cm_arr(1, 1), 11.0);
        EXPECT_EQ(cm_arr(1, 2), 12.0);
    }
}
