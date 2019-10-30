/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xoptional_assembly_storage.hpp"

#include "gtest/gtest.h"
#include "xtensor/xstorage.hpp"
#include "xtensor/xio.hpp"

namespace xt
{
    using storage_type = uvector<int>;
    using flag_storage_type = uvector<bool>;

    TEST(xoptional_assembly_storage, constructor)
    {
        storage_type v = {56, 2, 3, 5};
        flag_storage_type f = {false, true, true, true};
        auto stor = optional_assembly_storage(v, f);
    }

    TEST(xoptional_assembly_storage, empty)
    {
        storage_type v1 = {};
        flag_storage_type f1 = {};
        auto stor1 = optional_assembly_storage(v1, f1);
        ASSERT_EQ(v1.empty(), stor1.empty());
    }

    TEST(xoptional_assembly_storage, size)
    {
        storage_type v = {56, 2, 3, 5};
        flag_storage_type f = {false, true, true, true};
        auto stor = optional_assembly_storage(v, f);
        ASSERT_EQ(stor.size(), std::size_t(4));
    }

    TEST(xoptional_assembly_storage, resize)
    {
        storage_type v = {56, 2, 3, 5};
        flag_storage_type f = {false, true, true, true};
        auto stor = optional_assembly_storage(v, f);
        ASSERT_EQ(stor.size(), std::size_t(4));
        stor.resize(std::size_t(5));
        ASSERT_EQ(stor.size(), std::size_t(5));
        stor.resize(std::size_t(2));
        ASSERT_EQ(stor.size(), std::size_t(2));
    }

    TEST(xoptional_assembly_storage, access)
    {
        storage_type v = {56, 2, 3, 5};
        flag_storage_type f = {false, true, true, true};
        auto stor = optional_assembly_storage(v, f);

        ASSERT_EQ(stor[0], xtl::missing<int>());
        stor[0].has_value() = true;
        ASSERT_EQ(stor[0], 56);
        ASSERT_EQ(stor[1], 2);
        stor[1] = 123;
        ASSERT_EQ(stor[1], 123);
        ASSERT_EQ(stor[2], 3);
    }

    TEST(xoptional_assembly_storage, front)
    {
        storage_type v = {56, 2, 3, 5};
        flag_storage_type f = {false, true, true, true};
        auto stor = optional_assembly_storage(v, f);
        ASSERT_EQ(stor[0], stor.front());
    }

    TEST(xoptional_assembly_storage, back)
    {
        storage_type v = {56, 2, 3, 5};
        flag_storage_type f = {false, true, true, true};
        auto stor = optional_assembly_storage(v, f);
        ASSERT_EQ(stor[3], stor.back());
    }

    TEST(xoptional_assembly_storage, iterator)
    {
        storage_type v = {56, 2, 3, 5};
        flag_storage_type f = {false, true, true, true};
        auto stor = optional_assembly_storage(v, f);

        auto it = stor.begin();
        auto it2 = stor.begin();
        it++;
        ASSERT_EQ(*it, 2);
        it->has_value() = false;
        ASSERT_EQ(*it, xtl::missing<int>());
        it += 2;
        ASSERT_EQ(*it, 5);
        it -= 2;
        ASSERT_EQ(*it, xtl::missing<int>());
        ASSERT_FALSE(it == stor.end());
        it += 3;
        ASSERT_TRUE(it == stor.end());
        it--;
        ASSERT_EQ(*it, 5);
        ASSERT_TRUE(it2 < it);
    }

    TEST(xoptional_assembly_storage, const_iterator)
    {
        storage_type v = {56, 2, 3, 5};
        flag_storage_type f = {false, true, true, true};
        auto stor = optional_assembly_storage(v, f);

        auto it = stor.cbegin();
        auto it2 = stor.cbegin();
        it++;
        ASSERT_EQ(*it, 2);
        it += 2;
        ASSERT_EQ(*it, 5);
        it -= 2;
        ASSERT_EQ(*it, 2);
        ASSERT_FALSE(it == stor.cend());
        it += 3;
        ASSERT_TRUE(it == stor.cend());
        it--;
        ASSERT_EQ(*it, 5);
        ASSERT_TRUE(it2 < it);
    }

    TEST(xoptional_assembly_storage, reverse_iterator)
    {
        storage_type v = {56, 2, 3, 5};
        flag_storage_type f = {false, true, true, true};
        auto stor = optional_assembly_storage(v, f);

        auto it = stor.rbegin();
        auto it2 = stor.rbegin();
        it++;
        ASSERT_EQ(*it, 3);
        it += 2;
        ASSERT_EQ(*it, xtl::missing<int>());
        it -= 2;
        ASSERT_EQ(*it, 3);
        ASSERT_FALSE(it == stor.rend());
        it += 3;
        ASSERT_TRUE(it == stor.rend());
        it--;
        ASSERT_EQ(*it, xtl::missing<int>());
        ASSERT_TRUE(it2 < it);
    }

    TEST(xoptional_assembly_storage, swap)
    {
        storage_type v1 = {56, 2, 3, 5};
        flag_storage_type f1 = {false, true, true, true};
        auto stor1 = optional_assembly_storage(v1, f1);

        storage_type v2 = {};
        flag_storage_type f2 = {};
        auto stor2 = optional_assembly_storage(v2, f2);

        stor2.swap(stor1);
        ASSERT_EQ(stor1.size(), size_t(0));
        ASSERT_EQ(stor2.size(), size_t(4));
    }

    TEST(xoptional_assembly_storage, operators)
    {
        storage_type v1 = {56, 2, 3, 5};
        flag_storage_type f1 = {false, true, true, true};
        auto stor1 = optional_assembly_storage(v1, f1);

        storage_type v2 = {56, 2, 3, 6};
        flag_storage_type f2 = {false, true, true, true};
        auto stor2 = optional_assembly_storage(v2, f2);

        ASSERT_TRUE(v1 == v1);
        ASSERT_FALSE(v1 != v1);
        ASSERT_TRUE(v1 != v2);
        ASSERT_FALSE(v1 == v2);

        ASSERT_FALSE(v1 < v1);
        ASSERT_TRUE(v1 < v2);
        ASSERT_TRUE(v1 <= v1);
        ASSERT_TRUE(v1 <= v2);

        ASSERT_FALSE(v1 > v1);
        ASSERT_TRUE(v2 > v1);
        ASSERT_TRUE(v1 >= v1);
        ASSERT_TRUE(v2 >= v1);

        swap(v1, v2);
        ASSERT_EQ(v1[3], 6);
        ASSERT_EQ(v2[3], 5);
    }
}
