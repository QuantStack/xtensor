/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "test_common_macros.hpp"
#include "test_common_macros.hpp"
#include "xtensor/xtensor_config.hpp"
#include "xtensor/xstorage.hpp"
#include <numeric>

namespace xt
{
    using vector_type = uvector<double, XTENSOR_DEFAULT_ALLOCATOR(double)>;

    /***********
     * uvector *
     ***********/

    TEST(uvector, constructor)
    {
        vector_type a;
        EXPECT_EQ(size_t(0), a.size());

        vector_type b(10);
        EXPECT_EQ(size_t(10), b.size());

        vector_type c(10, 2.5);
        EXPECT_EQ(size_t(10), c.size());
        EXPECT_EQ(2.5, c[2]);

        std::vector<double> src(10, 1.5);
        vector_type d(src.cbegin(), src.cend());
        EXPECT_EQ(size_t(10), d.size());
        EXPECT_EQ(1.5, d[2]);
    }

    TEST(uvector, resize)
    {
        vector_type a;
        for (size_t i = 1; i < 11; ++i)
        {
            size_t size1 = i * 10;
            a.resize(size1);
            EXPECT_EQ(size1, a.size());
            size_t size2 = size1 - 5;
            a.resize(size2);
            EXPECT_EQ(size2, a.size());
        }
    }

    TEST(uvector, access)
    {
        vector_type a(10);
        a[0] = 1.0;
        EXPECT_EQ(1.0, a[0]);
        a[3] = 3.2;
        EXPECT_EQ(3.2, a[3]);
        a[5] = 2.7;
        EXPECT_EQ(2.7, a[5]);

        a.front() = 0.0;
        EXPECT_EQ(0.0, a[0]);

        a.back() = 1.0;
        EXPECT_EQ(1.0, a[9]);

        EXPECT_EQ(a.at(5), 2.7);
        XT_EXPECT_ANY_THROW(a.at(12));
    }

    TEST(uvector, iterator)
    {
        vector_type a(10);
        std::iota(a.begin(), a.end(), 0.);
        for (size_t i = 0; i < a.size(); ++i)
        {
            EXPECT_EQ(double(i), a[i]);
        }
    }

    /***********
     * svector *
     ***********/

    using svector_type = svector<std::size_t, 4>;

    TEST(svector, behavior)
    {
        svector_type s = {1,2,3,4};
        svector_type s2 = s;
        std::vector<std::size_t> v(s.begin(), s.end());
        std::vector<std::size_t> v2 = {1,2,3,4};

        EXPECT_TRUE(std::equal(s.begin(), s.end(), v.begin()));
        EXPECT_TRUE(std::equal(s2.begin(), s2.end(), v2.begin()));

        s.erase(s.begin(), s.begin() + 2);
        v.erase(v.begin(), v.begin() + 2);
        EXPECT_TRUE(std::equal(s.begin(), s.end(), v.begin()));

        s2.erase(s2.begin() + 1, s2.end());
        v2.erase(v2.begin() + 1, v2.end());
        EXPECT_TRUE(std::equal(s2.begin(), s2.end(), v2.begin()));

        std::size_t i1 = 50;
        std::size_t i2 = 50;
        EXPECT_TRUE(s2.on_stack());
        s2.push_back(10);
        s2.push_back(20);
        s2.push_back(30);
        s2.push_back(40);
        s2.push_back(i1);
        s2.push_back(std::move(i1));
        v2.push_back(10);
        v2.push_back(20);
        v2.push_back(30);
        v2.push_back(40);
        v2.push_back(i2);
        v2.push_back(std::move(i2));
        EXPECT_FALSE(s2.on_stack());
        EXPECT_TRUE(std::equal(s2.begin(), s2.end(), v2.begin()));
    }

    TEST(svector, insert)
    {
        svector_type s = {1,2,3,4};
        svector_type s2 = s;
        std::vector<std::size_t> v(s.begin(), s.end());
        std::vector<std::size_t> v2 = {1,2,3,4};

        s.insert(s.begin(), std::size_t(55));
        s.insert(s.begin() + 2, std::size_t(123));
        v.insert(v.begin(), std::size_t(55));
        v.insert(v.begin() + 2, std::size_t(123));
        std::size_t nr = 12321;
        s.insert(s.end(), nr);
        v.insert(v.end(), nr);

        EXPECT_TRUE(std::equal(s.begin(), s.end(), v.begin()));
    }

    TEST(svector, insert_range)
    {
        svector_type s0 = {1,2,3,4};
        svector_type s1 = s0;
        svector_type s2 = s0;
        std::vector<std::size_t> v0(s0.begin(), s0.end());
        std::vector<std::size_t> v1(v0);
        std::vector<std::size_t> v2(v0);
        std::vector<std::size_t> ins = {1, 4};

        s0.insert(s0.begin(), ins.cbegin(), ins.cend());
        v0.insert(v0.begin(), ins.cbegin(), ins.cend());
        s1.insert(s1.begin()+2, ins.cbegin(), ins.cend());
        v1.insert(v1.begin()+2, ins.cbegin(), ins.cend());
        s2.insert(s2.begin()+4, ins.cbegin(), ins.cend());
        v2.insert(v2.begin()+4, ins.cbegin(), ins.cend());

        EXPECT_TRUE(std::equal(s0.begin(), s0.end(), v0.begin()));
        EXPECT_TRUE(std::equal(s1.begin(), s1.end(), v1.begin()));
        EXPECT_TRUE(std::equal(s2.begin(), s2.end(), v2.begin()));
    }

    TEST(svector, insert_initializer_list)
    {
        svector_type s0 = {1,2,3,4};
        svector_type s1 = s0;
        svector_type s2 = s0;
        std::vector<std::size_t> v0(s0.begin(), s0.end());
        std::vector<std::size_t> v1(v0);
        std::vector<std::size_t> v2(v0);

        s0.insert(s0.begin(), {1u, 4u});
        v0.insert(v0.begin(), {1u, 4u});
        s1.insert(s1.begin()+2, {1u, 4u});
        v1.insert(v1.begin()+2, {1u, 4u});
        s2.insert(s2.begin()+4, {1u, 4u});
        v2.insert(v2.begin()+4, {1u, 4u});

        EXPECT_TRUE(std::equal(s0.begin(), s0.end(), v0.begin()));
        EXPECT_TRUE(std::equal(s1.begin(), s1.end(), v1.begin()));
        EXPECT_TRUE(std::equal(s2.begin(), s2.end(), v2.begin()));
    }

    TEST(svector, constructor)
    {
        svector_type a;
        EXPECT_EQ(size_t(0), a.size());

        svector_type b(10);
        EXPECT_EQ(size_t(10), b.size());

        svector_type c(10, 2);
        EXPECT_EQ(size_t(10), c.size());
        EXPECT_EQ(size_t(2), c[2]);

        std::vector<std::size_t> src(10, std::size_t(1));
        svector_type d(src.cbegin(), src.cend());
        EXPECT_EQ(size_t(10), d.size());
        EXPECT_EQ(size_t(1), d[2]);

        svector_type e(src);
        EXPECT_EQ(size_t(10), d.size());
        EXPECT_EQ(size_t(1), d[2]);

        svector_type f = { 1, 2, 3, 4 };
        EXPECT_EQ(size_t(4), f.size());
        EXPECT_EQ(size_t(3), f[2]);

        svector<std::size_t, 8> ov = { 1, 2, 3, 4, 5, 6, 7, 8 };
        svector_type g(ov);
        EXPECT_EQ(size_t(8), g.size());
        EXPECT_EQ(size_t(3), g[2]);
    }

    TEST(svector, assign)
    {
        svector_type a = { 1, 2, 3, 4 };

        svector_type src1(10, 2);
        a = src1;
        EXPECT_EQ(size_t(10), a.size());
        EXPECT_EQ(size_t(2), a[2]);

        std::vector<size_t> src2(5, 1);
        a = src2;
        EXPECT_EQ(size_t(5), a.size());
        EXPECT_EQ(size_t(1), a[2]);

        a = { 1, 2, 3, 4 };
        EXPECT_EQ(size_t(4), a.size());
        EXPECT_EQ(size_t(3), a[2]);

        svector<std::size_t, 4> src3(10, 1);
        a = src3;
        EXPECT_EQ(size_t(10), a.size());
        EXPECT_EQ(size_t(1), a[2]);
    }

    TEST(svector, resize)
    {
        svector_type a;
        for (size_t i = 1; i < 11; ++i)
        {
            size_t size1 = i * 10;
            a.resize(size1);
            EXPECT_EQ(size1, a.size());
            size_t size2 = size1 - 5;
            a.resize(size2);
            EXPECT_EQ(size2, a.size());
        }

        svector_type b = { 1, 3, 4 };
        b.resize(6);
        EXPECT_EQ(b[0], 1u);
        EXPECT_EQ(b[1], 3u);
        EXPECT_EQ(b[2], 4u);
    }

    TEST(svector, swap)
    {
        using std::swap;

        {
            svector_type a = { 1, 3, 4, 6, 7 };
            svector_type b = {};
            svector_type abu = a;
            svector_type bbu = b;

            swap(a, b);

            EXPECT_EQ(a, bbu);
            EXPECT_EQ(b, abu);
        }

        {
            svector_type a = { 1, 3 ,4 };
            svector_type b = { 2, 1, 5, 3, 9, 12 };
            svector_type abu = a;
            svector_type bbu = b;

            swap(a, b);

            EXPECT_EQ(a, bbu);
            EXPECT_EQ(b, abu);
        }

        {
            svector_type a = { 10, 13, 14 };
            svector_type b = { 12, 15, 17 };
            svector_type abu = a;
            svector_type bbu = b;

            swap(a, b);

            EXPECT_EQ(a, bbu);
            EXPECT_EQ(b, abu);
        }
    }

    TEST(svector, access)
    {
        svector_type a(10);
        a[0] = size_t(1);
        EXPECT_EQ(size_t(1), a[0]);
        a[3] = size_t(3);
        EXPECT_EQ(size_t(3), a[3]);
        a[5] = size_t(2);
        EXPECT_EQ(size_t(2), a[5]);

        a.front() = size_t(0);
        EXPECT_EQ(size_t(0), a[0]);

        a.back() = size_t(1);
        EXPECT_EQ(size_t(1), a[9]);

        EXPECT_EQ(a.at(5), size_t(2));
        XT_EXPECT_ANY_THROW(a.at(12));
    }

    TEST(svector, iterator)
    {
        svector_type a(10);
        std::iota(a.begin(), a.end(), std::size_t(0));
        for (size_t i = 0; i < a.size(); ++i)
        {
            EXPECT_EQ(i, a[i]);
        }
    }

    TEST(fixed_shape, fixed_shape)
    {
        fixed_shape<3, 4, 5> af;
        using cast_type = typename fixed_shape<3, 4, 5>::cast_type;
        cast_type a = af;
        EXPECT_EQ(a[0], size_t(3));
        EXPECT_EQ(a[2], size_t(5));
        EXPECT_EQ(a.back(), size_t(5));
        EXPECT_EQ(a.front(), size_t(3));
        EXPECT_EQ(a.size(), size_t(3));
    }

    TEST(fixed_shape, operator_eq)
    {
        fixed_shape<2, 3> a, b;
        fixed_shape<5> c;
        fixed_shape<2, 4> d;
        EXPECT_TRUE(a == b);
        EXPECT_FALSE(a == c);
        EXPECT_FALSE(a == d);
    }
}
