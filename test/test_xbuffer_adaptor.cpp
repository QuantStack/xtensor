/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xbuffer_adaptor.hpp"

#include "gtest/gtest.h"
#include "test_common_macros.hpp"

namespace xt
{
    using buffer_adaptor = xbuffer_adaptor<double*>;
    using allocator = std::allocator<double>;
    using owner_adaptor = xbuffer_adaptor<double*&, acquire_ownership>;

    TEST(xbuffer_adaptor, owner_destructor)
    {
        size_t size = 100;
        double* data = allocator{}.allocate(size);
        owner_adaptor adapt(data, size);
        EXPECT_EQ(data, adapt.data());
    }

    TEST(xbuffer_adaptor, owner_move)
    {
        size_t size = 100;
        double* data = allocator{}.allocate(size);
        owner_adaptor adapt(data, size);

        owner_adaptor adapt2(std::move(adapt));
        EXPECT_EQ(data, adapt2.data());
        EXPECT_EQ(size, adapt2.size());
        EXPECT_EQ(size_t(0), adapt.size());
    }

    TEST(xbuffer_adaptor, owner_copy_assign)
    {
        size_t size1 = 100;
        double* data1 = allocator{}.allocate(size1);
        data1[0] = 2.5;
        owner_adaptor adapt1(data1, size1);

        size_t size2 = 200;
        double* data2 = allocator{}.allocate(size2);
        data2[0] = 1.2;
        owner_adaptor adapt2(data2, size2);

        adapt1 = adapt2;
        EXPECT_EQ(adapt1.size(), adapt2.size());
        EXPECT_EQ(adapt1[0], adapt2[0]);
    }

    TEST(xbuffer_adaptor, owner_move_assign)
    {
        size_t size1 = 100;
        double* data1 = allocator{}.allocate(size1);
        data1[0] = 2.5;
        owner_adaptor adapt1(data1, size1);

        size_t size2 = 200;
        double* data2 = allocator{}.allocate(size2);
        double data2_ref = 1.2;
        data2[0] = data2_ref;
        owner_adaptor adapt2(data2, size2);

        adapt1 = std::move(adapt2);
        EXPECT_EQ(adapt1.size(), size2);
        EXPECT_EQ(adapt1[0], data2_ref);
    }

    TEST(xbuffer_adaptor, owner_resize)
    {
        size_t size1 = 100;
        double* data1 = allocator{}.allocate(size1);
        owner_adaptor adapt(data1, size1);

        size_t size2 = 50;
        adapt.resize(size2);

        EXPECT_EQ(adapt.size(), size2);
    }

    TEST(xbuffer_adaptor, owner_iterating)
    {
        size_t size = 100;
        double* data = allocator{}.allocate(size);
        owner_adaptor adapt(data, size);

        std::fill(adapt.begin(), adapt.end(), 1.2);
        EXPECT_EQ(data[0], 1.2);
        EXPECT_EQ(data[size / 2], 1.2);
        EXPECT_EQ(data[size - 1], 1.2);
    }

    TEST(xbuffer_adaptor, no_owner_copy)
    {
        size_t size = 100;
        double* data = new double[size];
        buffer_adaptor adapt1(data, size);

        buffer_adaptor adapt2(adapt1);
        EXPECT_EQ(adapt1.size(), adapt2.size());
        EXPECT_EQ(adapt1.data(), adapt2.data());
        delete[] data;
    }

    TEST(xbuffer_adaptor, no_owner_move)
    {
        size_t size = 100;
        double* data = new double[size];
        buffer_adaptor adapt1(data, size);

        buffer_adaptor adapt2(std::move(adapt1));
        EXPECT_EQ(adapt1.size(), adapt2.size());
        EXPECT_EQ(adapt1.data(), adapt2.data());
        delete[] data;
    }

    TEST(xbuffer_adaptor, no_owner_copy_assign)
    {
        size_t size1 = 100;
        double* data1 = new double[size1];
        buffer_adaptor adapt1(data1, size1);

        size_t size2 = 200;
        double* data2 = new double[size2];
        buffer_adaptor adapt2(data2, size2);

        adapt1 = adapt2;
        EXPECT_EQ(adapt1.size(), adapt2.size());
        EXPECT_EQ(adapt1.data(), adapt2.data());

        delete[] data2;
        delete[] data1;
    }

    TEST(xbuffer_adaptor, no_owner_move_assign)
    {
        size_t size1 = 100;
        double* data1 = new double[size1];
        buffer_adaptor adapt1(data1, size1);

        size_t size2 = 200;
        double* data2 = new double[size2];
        buffer_adaptor adapt2(data2, size2);

        adapt1 = std::move(adapt2);
        EXPECT_EQ(adapt1.size(), adapt2.size());
        EXPECT_EQ(adapt1.data(), adapt2.data());

        delete[] data2;
        delete[] data1;
    }

    TEST(xbuffer_adaptor, no_owner_resize)
    {
        size_t size1 = 100;
        double* data1 = new double[size1];
        buffer_adaptor adapt(data1, size1);

        size_t size2 = 50;
        EXPECT_THROW(adapt.resize(size2), std::runtime_error);
        EXPECT_EQ(adapt.size(), size1);
    }

    TEST(xbuffer_adaptor, no_owner_iterating)
    {
        size_t size = 100;
        double* data = new double[size];
        buffer_adaptor adapt(data, size);

        std::fill(adapt.begin(), adapt.end(), 1.2);
        EXPECT_EQ(data[0], 1.2);
        EXPECT_EQ(data[size / 2], 1.2);
        EXPECT_EQ(data[size - 1], 1.2);

        delete[] data;
    }
}
