/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xoptional.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "xtensor/xio.hpp"

namespace xt
{
    TEST(xoptional, tensor)
    {
        xtensor_optional<double, 2> m
            {{ 1.0 ,       2.0         },
             { 3.0 , xtl::missing<double>() }};

        ASSERT_EQ(m(0, 0).value(), 1.0);
        ASSERT_EQ(m(1, 0).value(), 3.0);
        ASSERT_FALSE(m(1, 1).has_value());
        ASSERT_EQ((m[{0, 0}].value()), 1.0);
    }

    TEST(xoptional, adaptor)
    {
        using vtype = xtl::xoptional_vector<double>;
        vtype v(4, 0.);
        v[0] = 1.;
        v[1] = 2.;
        v[2] = 3.;
        v[3] = xtl::missing<double>();

        using tadaptor = xtensor_adaptor<vtype&, 2, layout_type::row_major, xoptional_expression_tag>;
        tadaptor ta(v, {2, 2});
        ASSERT_EQ(ta(0, 0).value(), 1.0);
        ASSERT_EQ(ta(1, 0).value(), 3.0);
        ASSERT_FALSE(ta(1, 1).has_value());
        ASSERT_EQ((ta[{0, 0}].value()), 1.0);

        using aadaptor = xarray_adaptor<vtype&, layout_type::row_major, dynamic_shape<std::size_t>, xoptional_expression_tag>;
        aadaptor aa(v, {2, 2});
        ASSERT_EQ(aa(0, 0).value(), 1.0);
        ASSERT_EQ(aa(1, 0).value(), 3.0);
        ASSERT_FALSE(aa(1, 1).has_value());
        ASSERT_EQ((aa[{0, 0}].value()), 1.0);
    }

    TEST(xoptional, operation)
    {
        xtensor_optional<double, 2> m1
            {{ 0.0 ,       2.0         },
             { 3.0 , xtl::missing<double>() }};

        xtensor<double, 2> m2
            {{ 1.0 , 2.0 },
             { 3.0 , 1.0 }};

        auto res_add = m1 + m2;
        ASSERT_EQ(res_add(0, 0).value(), 1.0);
        ASSERT_EQ(res_add(1, 0).value(), 6.0);
        ASSERT_FALSE(res_add(1, 1).has_value());
        ASSERT_EQ(res_add.value()(0, 0), 1.0);
        ASSERT_TRUE(res_add.has_value()(0, 0));
        ASSERT_FALSE(res_add.has_value()(1, 1));

        auto res_mul = m1 * m2;
        ASSERT_EQ(res_mul(0, 0).value(), 0.0);
        ASSERT_EQ(res_mul(1, 0).value(), 9.0);
        ASSERT_FALSE(res_mul(1, 1).has_value());

        auto res_div = m1 / m2;
        ASSERT_EQ(res_div(0, 0).value(), 0.0);
        ASSERT_EQ(res_div(1, 0).value(), 1.0);
        ASSERT_FALSE(res_div(1, 1).has_value());

        xtensor_optional<double, 2> res = m1 + m2;
        ASSERT_EQ(res(0, 0).value(), 1.0);
        ASSERT_EQ(res(1, 0).value(), 6.0);
        ASSERT_FALSE(res(1, 1).has_value());

        xtensor_optional<double, 2> res_neg = -m1;
        ASSERT_EQ(res_neg(0, 0).value(), 0.0);
        ASSERT_EQ(res_neg(0, 1).value(), -2.0);
        ASSERT_EQ(res_neg(1, 0).value(), -3.0);
        ASSERT_FALSE(res_neg(1, 1).has_value());
    }

    TEST(xoptional, bool_operation)
    {
        xtensor_optional<bool, 2> m1
            {{ false ,       true },
             { false , xtl::missing<bool>() }};

        xtensor_optional<bool, 2> res = m1 && m1;
        EXPECT_FALSE(res(0, 0).value());
        EXPECT_TRUE(res(0, 1).value());
        EXPECT_FALSE(res(1, 0).value());
        EXPECT_EQ(res(1, 1), xtl::missing<bool>());
    }

    TEST(xoptional, xio)
    {
        std::ostringstream oss;
        xtensor_optional<double, 2> m
            {{ 0.0 ,       2.0         },
             { 3.0 , xtl::missing<double>() }};

        oss << m;
        std::string expect = "{{  0,   2},\n {  3, N/A}}";
        ASSERT_EQ(oss.str(), expect);
    }

    TEST(xoptional, ufunc)
    {
        xtensor_optional<double, 2> m
            {{ 0.0 ,       2.0         },
             { 3.0 , xtl::missing<double>() }};

        auto flag_view = xt::has_value(m);

        xtensor<bool, 2> res = flag_view;

        ASSERT_TRUE(res(0, 0));
        ASSERT_TRUE(res(0, 1));
        ASSERT_TRUE(res(1, 0));
        ASSERT_FALSE(res(1, 1));

        auto value_view = xt::value(m);

        xtensor<double, 2> resv = value_view;
        flag_view(1, 1) = true;
        ASSERT_TRUE(m(1, 1).has_value());
        value_view(1, 1) = 4.0;
        ASSERT_EQ(m(1, 1).value(), 4.0);
    }

    TEST(xoptional, ufunc_nonoptional)
    {
        xtensor<double, 2> m
            {{ 0.0 , 2.0 },
             { 3.0 , 1.0 }};

        auto flag_view = has_value(m);

        xtensor<bool, 2> res = flag_view;
        ASSERT_TRUE(res(0, 0));
        ASSERT_TRUE(res(0, 1));
        ASSERT_TRUE(res(1, 0));
        ASSERT_TRUE(res(1, 1));
    }

    TEST(xoptional, dynamic_view)
    {
        xarray_optional<int> a = {{{0, 1, 2, 3},
                                   {4, 5, 6, 7},
                                   {8, 9, 10, 11}},
                                  {{12, 13, 14, 15},
                                   {16, 17, 18, 19},
                                   {20, 21, 22, 23}}};
        a(1, 0, 1).has_value() = false;
        a(1, 2, 3).has_value() = false;

        auto view0 = dynamic_view(a, xdynamic_slice_vector({ 1, keep(0, 2), range(1, 4) }));
        auto v_view = view0.value();
        auto hv_view = view0.has_value();

        EXPECT_EQ(v_view.shape()[0], std::size_t(2));
        EXPECT_EQ(v_view.shape()[1], std::size_t(3));
        EXPECT_EQ(hv_view.shape()[0], std::size_t(2));
        EXPECT_EQ(hv_view.shape()[1], std::size_t(3));

        EXPECT_EQ(v_view(0, 0), 13);
        EXPECT_EQ(v_view(0, 1), 14);
        EXPECT_EQ(v_view(0, 2), 15);
        EXPECT_EQ(v_view(1, 0), 21);
        EXPECT_EQ(v_view(1, 1), 22);
        EXPECT_EQ(v_view(1, 2), 23);

        EXPECT_FALSE(hv_view(0, 0));
        EXPECT_TRUE(hv_view(0, 1));
        EXPECT_TRUE(hv_view(0, 2));
        EXPECT_TRUE(hv_view(1, 0));
        EXPECT_TRUE(hv_view(1, 1));
        EXPECT_FALSE(hv_view(1, 2));
    }

    TEST(xoptional, function_on_view)
    {
        xarray_optional<int> a = {{{0, 1, 2, 3},
                                   {4, 5, 6, 7},
                                   {8, 9, 10, 11}},
                                  {{12, 13, 14, 15},
                                   {16, 17, 18, 19},
                                   {20, 21, 22, 23}}};

        a(1, 0, 1).has_value() = false;
        a(1, 2, 3).has_value() = false;

        auto va = dynamic_view(a, xdynamic_slice_vector({ 1, keep(0, 2), range(1, 4) }));

        xarray_optional<int> b = {{0, 1, 2},
                                  {3, 4, 5}};

        auto f = va + b;
        auto vf = f.value();
        auto hvf = f.has_value();

        EXPECT_EQ(vf(0,0), 13);
        EXPECT_EQ(vf(0,1), 15);
        EXPECT_EQ(vf(0,2), 17);
        EXPECT_EQ(vf(1,0), 24);
        EXPECT_EQ(vf(1,1), 26);
        EXPECT_EQ(vf(1,2), 28);

        EXPECT_FALSE(hvf(0,0));
        EXPECT_TRUE(hvf(0,1));
        EXPECT_TRUE(hvf(0,2));
        EXPECT_TRUE(hvf(1,0));
        EXPECT_TRUE(hvf(1,1));
        EXPECT_FALSE(hvf(1,2));

        xarray_optional<int> res = f;
        for(size_t i = 0; i < f.shape()[0]; ++i)
        {
            for(size_t j = 0; j < f.shape()[1]; ++j)
            {
                EXPECT_EQ(f(i, j), res(i, j));
            }
        }
    }

    TEST(xoptional, broadcast)
    {
        xarray_optional<int> a = {{1, 2, 3},
                                  {4, 5, 6}};
        a(0, 1).has_value() = false;
        a(1, 2).has_value() = false;

        auto br = xt::broadcast(a, {2, 2, 3});
        auto vbr = br.value();
        auto hvbr = br.has_value();

        for(size_t i = 0; i < br.shape()[0]; ++i)
        {
            for(size_t j = 0; j < br.shape()[1]; ++j)
            {
                for(size_t k = 0; k < br.shape()[2]; ++k)
                {
                    EXPECT_EQ(vbr(i, j, k), a(0, j, k).value());
                    EXPECT_EQ(hvbr(i, j, k), a(0, j, k).has_value());
                }
            }
        }
    }

    class point
    {
    public:

        point() = default;
        point(int x, int y)
            : m_x(x), m_y(y)
        {
        }

        int& x() noexcept { return m_x; }
        int& y() noexcept { return m_y; }

        const int& x() const noexcept { return m_x; }
        const int& y() const noexcept { return m_y; }

    private:

        int m_x;
        int m_y;
    };

    struct abs_func
    {
        using value_type = int;
        using reference = int&;
        using const_reference = const int&;
        using pointer = int*;
        using const_pointer = const int*;

        reference operator()(point& p) const noexcept { return p.x(); }
        const_reference operator()(const point& p) const noexcept { return p.x(); }

        // Compilation trick: a functor view on xarray_optional<point> expects the functor
        // to provide operator() accepting reference and const_reference from xarray_optional<point>
        // Since we never use these overloads, no need to declare the right one, nor to provide
        // any implementation.
        template <class T> T&& operator()(T&&) const noexcept;
    };

    TEST(xoptional, functor_view)
    {
        xarray_optional<point> a = {{point(0, 0), point(0, 1)},
                                    {point(1, 0), point(1, 1)}};
        a(0, 0).has_value() = false;
        a(1, 0).has_value() = false;

        xfunctor_view<abs_func, xarray_optional<point>&> fv(abs_func(), a);
        auto vfv = fv.value();
        auto hvfv = fv.has_value();

        EXPECT_EQ(vfv(0, 0), a(0, 0).value().x());
        EXPECT_EQ(vfv(0, 1), a(0, 1).value().x());
        EXPECT_EQ(vfv(1, 0), a(1, 0).value().x());
        EXPECT_EQ(vfv(1, 1), a(1, 1).value().x());

        EXPECT_FALSE(hvfv(0, 0));
        EXPECT_TRUE(hvfv(0, 1));
        EXPECT_FALSE(hvfv(1, 0));
        EXPECT_TRUE(hvfv(1, 1));

        vfv(0, 0) = 4;
        hvfv(0, 0) = true;

        EXPECT_EQ(a(0, 0).value().x(), 4);
        EXPECT_TRUE(a(0, 0).has_value());
    }

    TEST(xoptional, index_view)
    {
        xarray_optional<int> a = {{1, 2, 3},
                                  {4, 5, 6}};
        a(0, 0).has_value() = false;
        a(1, 2).has_value() = false;

        auto iv = index_view(a, {{0ul,0ul}, {0ul,2ul}, {1ul,1ul}, {1ul,2ul}});
        auto viv = iv.value();
        auto hviv = iv.has_value();

        EXPECT_EQ(viv(0), a(0, 0).value());
        EXPECT_EQ(viv(1), a(0, 2).value());
        EXPECT_EQ(viv(2), a(1, 1).value());
        EXPECT_EQ(viv(3), a(1, 2).value());

        EXPECT_FALSE(hviv(0));
        EXPECT_TRUE(hviv(1));
        EXPECT_TRUE(hviv(2));
        EXPECT_FALSE(hviv(3));
    }

    TEST(xoptional, reducer)
    {
        xarray_optional<int> a = {{1, 2, 3},
                                  {4, 5, 6}};
        a(1, 2).has_value() = false;

        auto red = sum(a, {1});
        auto vred = red.value();
        auto hvred = red.has_value();

        EXPECT_EQ(vred(0), 6);
        EXPECT_EQ(vred(1), 15);

        EXPECT_TRUE(hvred(0));
        EXPECT_FALSE(hvred(1));
    }

    TEST(xoptional, strided_view)
    {
        xarray_optional<int> a = {{{0, 1, 2, 3},
                                   {4, 5, 6, 7},
                                   {8, 9, 10, 11}},
                                  {{12, 13, 14, 15},
                                   {16, 17, 18, 19},
                                   {20, 21, 22, 23}}};
        a(1, 0, 1).has_value() = false;
        a(1, 2, 3).has_value() = false;

        auto view0 = strided_view(a, {1, range(0, 2), range(1, 4)});
        auto v_view = view0.value();
        auto hv_view = view0.has_value();

        EXPECT_EQ(v_view.shape()[0], std::size_t(2));
        EXPECT_EQ(v_view.shape()[1], std::size_t(3));
        EXPECT_EQ(hv_view.shape()[0], std::size_t(2));
        EXPECT_EQ(hv_view.shape()[1], std::size_t(3));

        for(size_t i = 0; i < v_view.shape()[0]; ++i)
        {
            for(size_t j = 0; j < v_view.shape()[1]; ++j)
            {
                EXPECT_EQ(v_view(i, j), a(1, i, j+1).value());
                EXPECT_EQ(hv_view(i, j), a(1, i, j+1).has_value());
            }
        }
    }

    TEST(xoptional, view)
    {
        xarray_optional<int> a = {{{0, 1, 2, 3},
                                   {4, 5, 6, 7},
                                   {8, 9, 10, 11}},
                                  {{12, 13, 14, 15},
                                   {16, 17, 18, 19},
                                   {20, 21, 22, 23}}};
        a(1, 0, 1).has_value() = false;
        a(1, 2, 3).has_value() = false;

        auto view0 = view(a, 1, range(0, 2), range(1, 4));
        auto v_view = view0.value();
        auto hv_view = view0.has_value();

        EXPECT_EQ(v_view.shape()[0], std::size_t(2));
        EXPECT_EQ(v_view.shape()[1], std::size_t(3));
        EXPECT_EQ(hv_view.shape()[0], std::size_t(2));
        EXPECT_EQ(hv_view.shape()[1], std::size_t(3));

        for(size_t i = 0; i < v_view.shape()[0]; ++i)
        {
            for(size_t j = 0; j < v_view.shape()[1]; ++j)
            {
                EXPECT_EQ(v_view(i, j), a(1, i, j+1).value());
                EXPECT_EQ(hv_view(i, j), a(1, i, j+1).has_value());
            }
        }
    }

    struct float_identity
    {
        template <class T>
        int operator()(T t) const
        {
            return static_cast<int>(t);
        }
    };
    
    struct bool_even
    {
        template <class T>
        bool operator()(T t) const
        {
            return t%2 == 0;
        }
    };

    struct opt_func_tester
    {
        using value_functor_type = float_identity;
        using flag_functor_type = bool_even;

        value_functor_type value_functor() const
        {
            return m_value_functor;
        }

        flag_functor_type flag_functor() const
        {
            return m_flag_functor;
        }

        template <class T>
        xtl::xoptional<int, bool> operator()(T t) const
        {
            return xtl::xoptional<int, bool>(m_value_functor(t),
                                             m_flag_functor(t));
        }

        value_functor_type m_value_functor;
        flag_functor_type m_flag_functor;
    };

    TEST(xoptional, generator)
    {
        using gen_type = xgenerator<opt_func_tester, xtl::xoptional<int, bool>, dynamic_shape<std::size_t>>;
        gen_type g(opt_func_tester(), {4});
        auto vg = g.value();
        auto hvg = g.has_value();

        EXPECT_EQ(vg(0), 0);
        EXPECT_EQ(vg(1), 1);
        EXPECT_EQ(vg(2), 2);
        EXPECT_EQ(vg(3), 3);

        EXPECT_TRUE(hvg(0));
        EXPECT_FALSE(hvg(1));
        EXPECT_TRUE(hvg(2));
        EXPECT_FALSE(hvg(3));
    }

#define UNARY_OPTIONAL_TEST_IMPL(FUNC)                                         \
    xtensor_optional<double, 2> m1{{0.25, 1}, {0.75, xtl::missing<double>()}}; \
    xtensor<double, 2> m2{{0.25, 1}, {0.75, 1}};                               \
    ASSERT_TRUE(FUNC(m1)(0, 1).has_value());                                   \
    ASSERT_EQ(FUNC(m2)(0, 1), FUNC(m1)(0, 1).value());                         \
    ASSERT_FALSE(FUNC(m1)(1, 1).has_value());

#define UNARY_OPTIONAL_TEST(FUNC)      \
    TEST(xoptional, FUNC)              \
    {                                  \
        UNARY_OPTIONAL_TEST_IMPL(FUNC) \
    }

#define UNARY_OPTIONAL_TEST_QUALIFIED(FUNC) \
    TEST(xoptional, FUNC)                   \
    {                                       \
        UNARY_OPTIONAL_TEST_IMPL(xt::FUNC)  \
    }

#define BINARY_OPTIONAL_TEST(FUNC)                                                   \
    TEST(xoptional, FUNC)                                                            \
    {                                                                                \
        xtensor_optional<double, 2> m1{{0.25, 0.5}, {0.75, xtl::missing<double>()}}; \
        xtensor_optional<double, 2> m2{{0.25, xtl::missing<double>()}, {0.75, 1.}};  \
        xtensor<double, 2> m3{{0.25, 0.5}, {0.75, 1.}};                              \
        ASSERT_TRUE(FUNC(m1, m3)(0, 1).has_value());                                 \
        ASSERT_EQ(FUNC(m3, m3)(0, 1), FUNC(m1, m3)(0, 1).value());                   \
        ASSERT_FALSE(FUNC(m1, m3)(1, 1).has_value());                                \
        ASSERT_TRUE(FUNC(m3, m1)(0, 1).has_value());                                 \
        ASSERT_EQ(FUNC(m3, m3)(0, 1), FUNC(m3, m1)(0, 1).value());                   \
        ASSERT_FALSE(FUNC(m3, m1)(1, 1).has_value());                                \
        ASSERT_TRUE(FUNC(m1, m2)(1, 0).has_value());                                 \
        ASSERT_EQ(FUNC(m3, m3)(1, 0), FUNC(m1, m2)(1, 0).value());                   \
        ASSERT_FALSE(FUNC(m1, m2)(0, 1).has_value());                                \
        ASSERT_FALSE(FUNC(m1, m2)(1, 1).has_value());                                \
    }

#define TERNARY_OPTIONAL_TEST_IMPL(FUNC)                                         \
    xtensor_optional<double, 2> m1{{0.25, 0.5}, {0.75, xtl::missing<double>()}}; \
    xtensor<double, 2> m4{{0.25, 0.5}, {0.75, 1.}};                              \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m1, m4, m4)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m1, m4, m4)(1, 1).has_value());                            \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m4, m1, m4)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m4, m1, m4)(1, 1).has_value());                            \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m4, m4, m1)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m4, m4, m1)(1, 1).has_value());                            \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m1, m1, m4)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m1, m1, m4)(1, 1).has_value());                            \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m1, m4, m1)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m1, m4, m1)(1, 1).has_value());                            \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m4, m1, m1)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m4, m1, m1)(1, 1).has_value());                            \
    ASSERT_EQ(FUNC(m4, m4, m4)(0, 0), FUNC(m1, m1, m1)(0, 0).value());           \
    ASSERT_FALSE(FUNC(m1, m1, m1)(1, 1).has_value());

#define TERNARY_OPTIONAL_TEST(FUNC)          \
    TEST(xoptional, FUNC)                    \
    {                                        \
        TERNARY_OPTIONAL_TEST_IMPL(xt::FUNC) \
    }

    UNARY_OPTIONAL_TEST(abs)
    UNARY_OPTIONAL_TEST(fabs)
    BINARY_OPTIONAL_TEST(fmod)
    BINARY_OPTIONAL_TEST(remainder)
    TERNARY_OPTIONAL_TEST(fma)
    BINARY_OPTIONAL_TEST(fmax)
    BINARY_OPTIONAL_TEST(fmin)
    BINARY_OPTIONAL_TEST(fdim)
    UNARY_OPTIONAL_TEST(sign)
    UNARY_OPTIONAL_TEST(exp)
    UNARY_OPTIONAL_TEST(exp2)
    UNARY_OPTIONAL_TEST(expm1)
    UNARY_OPTIONAL_TEST(log)
    UNARY_OPTIONAL_TEST(log10)
    UNARY_OPTIONAL_TEST(log2)
    UNARY_OPTIONAL_TEST(log1p)
    BINARY_OPTIONAL_TEST(pow)
    UNARY_OPTIONAL_TEST(sqrt)
    UNARY_OPTIONAL_TEST(cbrt)
    BINARY_OPTIONAL_TEST(hypot)
    UNARY_OPTIONAL_TEST(sin)
    UNARY_OPTIONAL_TEST(cos)
    UNARY_OPTIONAL_TEST(tan)
    UNARY_OPTIONAL_TEST(acos)
    UNARY_OPTIONAL_TEST(asin)
    UNARY_OPTIONAL_TEST(atan)
    BINARY_OPTIONAL_TEST(atan2)
    UNARY_OPTIONAL_TEST(sinh)
    UNARY_OPTIONAL_TEST(cosh)
    UNARY_OPTIONAL_TEST(tanh)
    UNARY_OPTIONAL_TEST(acosh)
    UNARY_OPTIONAL_TEST(asinh)
    UNARY_OPTIONAL_TEST(atanh)
    UNARY_OPTIONAL_TEST(erf)
    UNARY_OPTIONAL_TEST(erfc)
    UNARY_OPTIONAL_TEST(tgamma)
    UNARY_OPTIONAL_TEST(lgamma)
    UNARY_OPTIONAL_TEST_QUALIFIED(isfinite)
    UNARY_OPTIONAL_TEST_QUALIFIED(isinf)
    UNARY_OPTIONAL_TEST_QUALIFIED(isnan)

#undef TERNARY_OPTIONAL_TEST
#undef TERNARY_OPTIONAL_TEST_IMPL
#undef BINARY_OPTIONAL_TEST
#undef UNARY_OPTIONAL_TEST_QUALIFIED
#undef UNARY_OPTIONAL_TEST
#undef UNARY_OPTIONAL_TEST_IMPL
}
