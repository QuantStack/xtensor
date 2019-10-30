/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xadapt.hpp"

#include "gtest/gtest.h"
#include "xtensor/xstrides.hpp"

namespace xt
{
    using vec_type = std::vector<int>;

    TEST(xarray_adaptor, adapt)
    {
        vec_type v(4, 0);
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({2, 2});

        auto a1 = adapt(v, s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, v[std::size_t(a1.strides()[1])]);

        shape_type str({2, 1});
        auto a2 = adapt(v, s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, v[2]);
    }

    TEST(xarray_adaptor, adapt_layout)
    {
        vec_type v(4, 0);
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({2, 2});

        auto a1 = adapt<layout_type::dynamic>(v, s, layout_type::row_major);
        a1(0, 1) = 1;
        EXPECT_EQ(1, v[std::size_t(a1.strides()[1])]);
    }

    TEST(xarray_adaptor, pointer_no_ownership)
    {
        size_t size = 4;
        std::unique_ptr<int[]> data(new int[size]);
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({2, 2});

        auto a1 = adapt(data.get(), size, no_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data[std::size_t(a1.strides()[1])]);

        shape_type str({2, 1});
        auto a2 = adapt(data.get(), size, no_ownership(), s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, data[2]);
    }

    TEST(xarray_adaptor, pointer_acquire_ownership)
    {
        size_t size = 4;
        int* data = std::allocator<int>{}.allocate(size);
        int* data2 =  std::allocator<int>{}.allocate(size);;
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({2, 2});

        auto a1 = adapt(data, size, acquire_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data[std::size_t(a1.strides()[1])]);

        shape_type str({2, 1});
        auto a2 = adapt(data2, size, acquire_ownership(), s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, data2[2]);
    }

    TEST(xarray_adaptor, c_stack_array)
    {
        double data[4] = { 1., 2., 3., 4. };
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({2, 2});
        auto a = adapt(data, s);

        EXPECT_EQ(a.size(), 4u);
        if(XTENSOR_DEFAULT_LAYOUT == xt::layout_type::row_major)
        {
            EXPECT_EQ(a(0, 1), 2.);
            EXPECT_EQ(a(1, 0), 3.);
        }
        else
        {
            EXPECT_EQ(a(0, 1), 3.);
            EXPECT_EQ(a(1, 0), 2.);
        }
        shape_type str({2, 1});
        auto b = adapt(data, s, str);
        EXPECT_EQ(b.size(), 4u);
        EXPECT_EQ(b(0, 1), 2.);
        EXPECT_EQ(b(1, 0), 3.);
    }

    TEST(xarray_adaptor, no_ownership_assign)
    {
        size_t size = 1;
        int data1 = 0;
        int data2 = 1;
        int data3;
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({ 1 });

        auto a1 = adapt(&data1, size, no_ownership(), s);
        auto a2 = adapt(&data2, size, no_ownership(), s);
        auto a3 = adapt(&data3, size, no_ownership(), s);
        a3 = a1 + a2;
        EXPECT_EQ(1, data3);
    }

    TEST(xarray_adaptor, acquire_ownership_assign)
    {
        size_t size = 1;
        int* data1 = std::allocator<int>{}.allocate(1);
        data1[0] = 0;
        int* data2 = std::allocator<int>{}.allocate(1);
        data2[0] = 1;
        int* data3 = nullptr;
        using shape_type = std::vector<vec_type::size_type>;
        shape_type s({ 1 });

        auto a1 = adapt(data1, size, acquire_ownership(), s);
        auto a2 = adapt(data2, size, acquire_ownership(), s);
        auto a3 = adapt(data3, size_t(0), acquire_ownership(), s);
        a3 = a1 + a2;
        EXPECT_EQ(1, *data3);
    }

    TEST(xarray_adaptor, ptr_adapt_layout)
    {
        size_t size = 4;
        std::unique_ptr<int[]> data(new int[size]);

        using shape_type = std::vector<vec_type::size_type>;
        shape_type s = { size };

        auto a0 = adapt<layout_type::dynamic>(data.get(), size, no_ownership(), s, layout_type::row_major);
        a0(3) = 3;
        EXPECT_EQ(3, data[3]);
    }

    TEST(xtensor_adaptor, adapt)
    {
        vec_type v0(4, 0);
        auto a0 = adapt(v0);
        a0(0) = 1;
        a0(3) = 3;
        EXPECT_EQ(1, v0[0]);
        EXPECT_EQ(3, v0[3]);

        vec_type v(4, 0);
        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};

        auto a1 = adapt(v, s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, v[std::size_t(a1.strides()[1])]);

        shape_type str = {2, 1};
        auto a2 = adapt(v, s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, v[2]);
    }

    TEST(xtensor_adaptor, adapt_layout)
    {
        vec_type v(4, 0);
        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};

        auto a1 = adapt<layout_type::dynamic>(v, s, layout_type::column_major);
        a1(0, 1) = 1;
        EXPECT_EQ(1, v[std::size_t(a1.strides()[1])]);
    }

    TEST(xtensor_adaptor, pointer_no_ownership)
    {
        size_t size = 4;
        std::unique_ptr<int[]> data(new int[size]);

        auto a0 = adapt(data.get(), size, no_ownership());
        a0(3) = 3;
        EXPECT_EQ(3, data[3]);

        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};

        auto a1 = adapt(data.get(), size, no_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data[a1.strides()[1]]);

        shape_type str = {2, 1};
        auto a2 = adapt(data.get(), size, no_ownership(), s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, data[2]);
    }

    TEST(xtensor_adaptor, pointer_const_no_ownership)
    {
        size_t size = 4;
        std::unique_ptr<int[]> data(new int[size]);
        const int* const_data = data.get();

        auto a0 = adapt(data.get(), size, no_ownership());
        auto a0_view = adapt(const_data, size, no_ownership());
        a0(3) = 3;
        EXPECT_EQ(3, a0_view[3]);

        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};

        auto a1 = adapt(data.get(), size, no_ownership(), s);
        auto a1_view = adapt(data.get(), size, no_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, a1_view(0, 1));
    }

    TEST(xtensor_adaptor, pointer_acquire_ownership)
    {
        size_t size = 4;
        int* data0 = std::allocator<int>{}.allocate(size);
        int* data1 = std::allocator<int>{}.allocate(size);
        int* data2 = std::allocator<int>{}.allocate(size);

        auto a0 = adapt(data0, size, acquire_ownership());
        a0(3) = 3;
        EXPECT_EQ(3, data0[3]);

        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};

        auto a1 = adapt(data1, size, acquire_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data1[a1.strides()[1]]);

        shape_type str = {2, 1};
        auto a2 = adapt(data2, size, acquire_ownership(), s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, data2[2]);
    }

    TEST(xtensor_adaptor, move_pointer_acquire_ownership)
    {
        size_t size = 4;
        int* data = std::allocator<int>{}.allocate(size);
        int* data2 = std::allocator<int>{}.allocate(size);
        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};

        auto a1 = adapt(std::move(data), size, acquire_ownership(), s);
        a1(0, 1) = 1;
        EXPECT_EQ(1, data[a1.strides()[1]]);

        shape_type str = {2, 1};
        auto a2 = adapt(std::move(data2), size, acquire_ownership(), s, str);
        a2(1, 0) = 1;
        EXPECT_EQ(1, data2[2]);
    }

    TEST(xtensor_adaptor, c_stack_array)
    {
        double data[4] = { 1., 2., 3., 4. };
        using shape_type = std::array<vec_type::size_type, 2>;
        shape_type s = {2, 2};
        auto a = adapt(data, s);

        EXPECT_EQ(a.size(), 4u);
        if(XTENSOR_DEFAULT_LAYOUT == xt::layout_type::row_major)
        {
            EXPECT_EQ(a(0, 1), 2.);
            EXPECT_EQ(a(1, 0), 3.);
        }
        else
        {
            EXPECT_EQ(a(0, 1), 3.);
            EXPECT_EQ(a(1, 0), 2.);
        }

        shape_type str = {2, 1};
        auto b = adapt(data, s, str);
        EXPECT_EQ(b.size(), 4u);
        EXPECT_EQ(b(0, 1), 2.);
        EXPECT_EQ(b(1, 0), 3.);
    }

    TEST(xtensor_adaptor, no_ownership_assign)
    {
        size_t size = 1;
        int data1 = 0;
        int data2 = 1;
        int data3;
        using shape_type = std::array<vec_type::size_type, 1>;
        shape_type s = { 1 };

        auto a1 = adapt(&data1, size, no_ownership(), s);
        auto a2 = adapt(&data2, size, no_ownership(), s);
        auto a3 = adapt(&data3, size, no_ownership(), s);
        a3 = a1 + a2;
        EXPECT_EQ(1, data3);
    }

    TEST(xtensor_adaptor, const_no_ownership_assign)
    {
        size_t size = 1;
        int data1 = 0;
        int data2 = 1;
        int data3;
        const int* const p_data1 = &data1;
        const int* const p_data2 = &data2;
        int* p_data3 = &data3;

        using shape_type = std::array<vec_type::size_type, 1>;
        shape_type s = { 1 };

        auto a1 = adapt(p_data1, size, no_ownership(), s);
        auto a2 = adapt(p_data2, size, no_ownership(), s);
        auto a3 = adapt(p_data3, size, no_ownership(), s);

        a3 = a1 + a2;

        EXPECT_EQ(1, data3);
    }

    TEST(xtensor_adaptor, acquire_ownership_assign)
    {
        size_t size = 1;
        int* data1 = std::allocator<int>{}.allocate(size);
        data1[0] = 0;
        int* data2 = std::allocator<int>{}.allocate(size);
        data2[0] = 1;
        int* data3 = nullptr;
        using shape_type = std::array<vec_type::size_type, 1>;
        shape_type s = { 1 };

        auto a1 = adapt(data1, size, acquire_ownership(), s);
        auto a2 = adapt(data2, size, acquire_ownership(), s);
        auto a3 = adapt(data3, size_t(0), acquire_ownership(), s);
        a3 = a1 + a2;
        EXPECT_EQ(1, *data3);
    }

    TEST(xtensor_adaptor, ptr_adapt_layout)
    {
        size_t size = 4;
        std::unique_ptr<int[]> data(new int[size]);

        using shape_type = std::array<vec_type::size_type, 1>;
        shape_type s = { size };

        auto a0 = adapt<layout_type::dynamic>(data.get(), size, no_ownership(), s, layout_type::column_major);
        a0(3) = 3;
        EXPECT_EQ(3, data[3]);
    }

    TEST(xarray_adaptor, short_syntax)
    {
        std::vector<int> a({1,2,3,4,5,6,7,8});
        auto xa = adapt(&a[1], std::vector<std::size_t>({3}));
        xa(1) = 123;

        EXPECT_EQ(a[2], 123);
        EXPECT_EQ(xa(2), 4);
    }

    TEST(xtensor_adaptor, nice_syntax)
    {
        std::vector<int> a({1,2,3,4,5,6,7,8});
#ifndef X_OLD_CLANG
        auto xa = adapt(&a[0], {2, 4});
        bool truthy = std::is_same<decltype(xa)::shape_type, std::array<std::size_t, 2>>::value;
        EXPECT_TRUE(truthy);
#else
        auto xa = adapt(&a[0], {2, 4});
        bool truthy = std::is_same<decltype(xa)::shape_type, xt::dynamic_shape<size_t>>::value;
        EXPECT_TRUE(truthy);
#endif
        xa(0, 0) = 100;
        xa(1, 3) = 1000;
        EXPECT_EQ(a[0], 100);
        EXPECT_EQ(a[7], 1000);
    }

    TEST(xtensor_fixed_adaptor, adapt)
    {
        std::vector<int> a({1,2,3,4,5,6,7,8});
        auto xa = adapt(&a[0], xshape<2, 4>());
        xa(0, 0) = 100;
        xa(1, 3) = 1000;
        EXPECT_EQ(a[0], 100);
        EXPECT_EQ(a[7], 1000);
        bool truthy = std::is_same<decltype(xa)::shape_type, xshape<2, 4>>::value;
        EXPECT_TRUE(truthy);
        const std::vector<int> b({5,5,19,5});
        auto xb = adapt(&b[0], xshape<2, 2>());
        if (XTENSOR_DEFAULT_LAYOUT == layout_type::row_major)
        {
            EXPECT_EQ(xb(1, 0), 19);
        }
        else
        {
            EXPECT_EQ(xb(1, 0), 5);
        }
    }

    namespace xadapt_test
    {
        struct Buffer {
            std::vector<double> buf;
            Buffer(std::vector<double>& ibuf) : buf(ibuf) {}
        };
    }

#ifndef X_OLD_CLANG
    TEST(xtensor_adaptor, smart_ptr)
    {
        auto data = std::vector<double>{1,2,3,4,5,6,7,8};
        auto shared_buf = std::make_shared<xadapt_test::Buffer>(data);
        auto unique_buf = std::make_unique<xadapt_test::Buffer>(data);

        std::shared_ptr<double> dptr(new double[8], std::default_delete<double[]>());
        dptr.get()[2] = 2.1;
        auto xdptr = adapt_smart_ptr(dptr, {4, 2});

        if (XTENSOR_DEFAULT_LAYOUT == layout_type::row_major)
        {
            EXPECT_EQ(xdptr(1, 0), 2.1);
            xdptr(3, 1) = 123.;
            EXPECT_EQ(dptr.get()[7], 123.);
        }
        else
        {
            EXPECT_EQ(xdptr(2, 0), 2.1);
            xdptr(3, 1) = 123.;
            EXPECT_EQ(dptr.get()[7], 123.);
        }

        EXPECT_EQ(shared_buf.use_count(), 1);
        {
            auto obj = adapt_smart_ptr(shared_buf.get()->buf.data(), {2, 4}, shared_buf);
            EXPECT_EQ(shared_buf.use_count(), 2);
        }
        EXPECT_EQ(shared_buf.use_count(), 1);

        {
            auto obj = adapt_smart_ptr(unique_buf.get()->buf.data(), {2, 4}, std::move(unique_buf));
        }
    }
#endif
}
