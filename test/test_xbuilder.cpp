/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifdef _MSC_VER
#define VS_SKIP_CONCATENATE_FIXED 1
#endif

#include "test_common.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"

#include "xtensor/xio.hpp"
#include <sstream>

namespace xt
{
    using std::size_t;
    using shape_t = std::vector<std::size_t>;

    TEST(xbuilder, ones)
    {
        auto m = ones<double>({1, 2});
        ASSERT_EQ(size_t(2), m.dimension());
        ASSERT_EQ(1.0, m(0, 1));
        xarray<double> m_assigned = m;
        ASSERT_EQ(1.0, m_assigned(0, 1));

        // assignment with narrowing type cast
        // (check that the compiler doesn't issue a warning)
        xarray<uint8_t> c = cast<uint8_t>(m);
        ASSERT_EQ(1, c(0, 1));
    }

    TEST(xbuilder, like)
    {
        bool type_equal = false;
        auto arr = xarray<int>::from_shape({3,2,5});
        auto xfx = xtensor_fixed<int, xt::xshape<3, 3, 3>>();

        auto onas = ones_like(arr);
        EXPECT_EQ(onas.shape(), arr.shape());
        type_equal = std::is_same<typename decltype(onas)::value_type, int>::value;
        EXPECT_TRUE(type_equal);
        type_equal = std::is_same<typename decltype(onas)::shape_type, xt::dynamic_shape<std::size_t>>::value;
        EXPECT_TRUE(type_equal);
        EXPECT_EQ(onas(1, 1), 1);

        auto zeras = zeros_like(arr);
        EXPECT_EQ(zeras.shape(), arr.shape());
        type_equal = std::is_same<typename decltype(zeras)::value_type, int>::value;
        EXPECT_TRUE(type_equal);
        EXPECT_EQ(zeras(1, 1), 0);

        auto empty = empty_like(arr);
        EXPECT_EQ(empty.shape(), arr.shape());
        type_equal = std::is_same<typename decltype(empty)::value_type, int>::value;
        EXPECT_TRUE(type_equal);

        auto full = full_like(arr, 123);
        EXPECT_EQ(full.shape(), arr.shape());
        type_equal = std::is_same<typename decltype(full)::value_type, int>::value;
        EXPECT_TRUE(type_equal);
        EXPECT_EQ(full(1, 1), 123);

        auto f_xfx = full_like(xfx, 2332);
        EXPECT_EQ(f_xfx(0, 2), 2332);
    }

    TEST(xbuilder, arange_simple)
    {
        auto ls = arange<double>(50);
        ASSERT_EQ(ls.dimension(), size_t(1));
        decltype(ls)::shape_type expected_shape = {50};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 0);
        auto ls_49 = ls(49);
        ASSERT_EQ(49, ls_49);
        ASSERT_EQ(ls(29), 29);
        xarray<double> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), size_t(1));
        ASSERT_EQ(m_assigned.shape()[0], size_t(50));
        ASSERT_EQ(m_assigned[{0}], 0);
        ASSERT_EQ(m_assigned[{49}], 49);
        ASSERT_EQ(m_assigned[{29}], 29);

        xarray<double> b({2, 50}, 1.);
        xarray<double> res = b + ls;
        ASSERT_EQ(50, res(1, 49));
    }

    TEST(xbuilder, arange_reshape)
    {
        auto rs0 = arange<double>(50).reshape({std::size_t(5), std::size_t(10)});
        auto rs1 = arange<double>(50).reshape({-1, 10});

        auto gen0 = arange<double>(50);
        auto ls0 = gen0.reshape({std::size_t(5), std::size_t(10)});
        auto gen1 = arange<double>(50);
        auto ls1 = gen1.reshape({-1, 10});

        decltype(ls0)::shape_type expected_shape = {5, 10};
        EXPECT_EQ(rs0.shape(), expected_shape);
        EXPECT_EQ(rs1.shape(), expected_shape);
        EXPECT_EQ(ls0.shape(), expected_shape);
        EXPECT_EQ(ls1.shape(), expected_shape);

        EXPECT_EQ(rs0(4, 9), 49);
        EXPECT_EQ(rs1(4, 9), 49);
        EXPECT_EQ(ls0(4, 9), 49);
        EXPECT_EQ(ls1(4, 9), 49);
    }

    TEST(xbuilder, arange_min_max)
    {
        auto ls = arange<unsigned int>(10u, 20u);
        ASSERT_EQ(ls.dimension(), size_t(1));
        decltype(ls)::shape_type expected_shape = {10};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 10u);
        ASSERT_EQ(ls(9), 19u);
        ASSERT_EQ(ls(2), 12u);
        xarray<unsigned int> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), size_t(1));
        ASSERT_EQ(m_assigned.shape()[0], size_t(10));
        ASSERT_EQ(m_assigned[{0}], 10u);
        ASSERT_EQ(m_assigned[{9}], 19u);
        ASSERT_EQ(m_assigned[{2}], 12u);

        auto lc = arange<char>('a', 'd');
        decltype(lc)::shape_type expected_shape_2 = {3};
        ASSERT_EQ(lc.shape(), expected_shape_2);
        ASSERT_EQ(lc[{0}], 'a');
        ASSERT_EQ(lc[{1}], 'b');
        ASSERT_EQ(lc[{2}], 'c');
    }

    TEST(xbuilder, arange_min_max_step)
    {
        auto ls = arange<float>(10, 20, 0.5f);
        ASSERT_EQ(ls.dimension(), size_t(1));
        decltype(ls)::shape_type expected_shape = {20};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 10.f);
        ASSERT_EQ(ls(10), 15.f);
        ASSERT_EQ(ls(3), 11.5f);
        xarray<float> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), size_t(1));
        ASSERT_EQ(m_assigned.shape()[0], size_t(20));
        ASSERT_EQ(m_assigned[{0}], 10.f);
        ASSERT_EQ(m_assigned(10), 15.f);
        ASSERT_EQ(m_assigned(3), 11.5f);

        auto l3 = arange<float>(0, 1, 0.3f);
        decltype(l3)::shape_type expected_shape_2 = {4};
        ASSERT_EQ(l3.shape(), expected_shape_2);
        ASSERT_EQ(l3[{0}], 0.f);
        ASSERT_EQ(3.f * 0.3f, l3[{3}]);

        auto l4 = arange<int>(0, 10, 3);
        ASSERT_EQ(l4.shape(), expected_shape_2);
        ASSERT_EQ(l4[{0}], 0);
        ASSERT_EQ(l4[{1}], 3);
        ASSERT_EQ(l4[{2}], 6);
        ASSERT_EQ(l4[{3}], 9);
    }

    TEST(xbuilder, arange_reverse)
    {
        auto a0 = arange(6, 5, -1);
        EXPECT_EQ(a0.dimension(), size_t(1));
        decltype(a0)::shape_type expected_shape0 = {1};
        EXPECT_EQ(a0.shape(), expected_shape0);
        EXPECT_EQ(a0(0), 6);

        auto a1 = arange(8, 5, -1);
        EXPECT_EQ(a1.dimension(), size_t(1));
        decltype(a1)::shape_type expected_shape1 = {3};
        EXPECT_EQ(a1.shape(), expected_shape1);
        EXPECT_EQ(a1(0), 8);
        EXPECT_EQ(a1(1), 7);
        EXPECT_EQ(a1(2), 6);

        auto a2 = arange(5, 6, -1);
        EXPECT_EQ(a2.dimension(), size_t(1));
        decltype(a2)::shape_type expected_shape2 = {0};
        EXPECT_EQ(a2.shape(), expected_shape2);

        auto a3 = arange(8, 5, 1);
        EXPECT_EQ(a3.dimension(), size_t(1));
        decltype(a3)::shape_type expected_shape3 = {0};
        EXPECT_EQ(a3.shape(), expected_shape3);
    }

    TEST(xbuilder, linspace)
    {
        auto ls = linspace<float>(20.f, 50.f);
        ASSERT_EQ(ls.dimension(), size_t(1));
        decltype(ls)::shape_type expected_shape = {50};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 20.f);
        ASSERT_EQ(ls(49), 50.f);

        float at_3 = 20 + 3 * (50.f - 20.f) / (50.f - 1.f);
        ASSERT_EQ(ls(3), at_3);

        xarray<float> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), size_t(1));
        ASSERT_EQ(m_assigned.shape()[0], size_t(50));
        ASSERT_EQ(m_assigned[{0}], 20.f);
        ASSERT_EQ(m_assigned(49), 50.f);
        ASSERT_EQ(m_assigned(3), at_3);
    }

    TEST(xbuilder, linspace_reshape)
    {
        xarray<double> a = linspace<double>(20., 50.).reshape({5, 10});
        EXPECT_EQ(a.dimension(), size_t(2));
        decltype(a)::shape_type expected_shape = {5, 10};
        EXPECT_EQ(a.shape(), expected_shape);
        EXPECT_EQ(a(0, 0), 20.);
        EXPECT_EQ(a(4, 9), 50.);
    }

    TEST(xbuilder, linspace_1_point)
    {
        xt::xarray<double> a = linspace<double>(0., 0., 1, false);
        decltype(a)::shape_type expected_shape = {1};
        EXPECT_EQ(a.dimension(), size_t(1));
        EXPECT_EQ(a.shape(), expected_shape);
        EXPECT_EQ(0., a(0));

        xt::xarray<double> b = linspace<double>(0., 0., 1, true);
        EXPECT_EQ(b.dimension(), size_t(1));
        EXPECT_EQ(b.shape(), expected_shape);
        EXPECT_EQ(0., b(0));

        xt::xarray<double> c = linspace<double>(0., 2., 1, true);
        EXPECT_EQ(c.dimension(), size_t(1));
        EXPECT_EQ(c.shape(), expected_shape);
        EXPECT_EQ(0., c(0));

        auto d = linspace<double>(0., 2., 1, true);
        EXPECT_EQ(c, d);
    }

    TEST(xbuilder, linspace_n_samples_endpoint)
    {
        auto ls = linspace<float>(20.f, 50.f, 100, false);
        ASSERT_EQ(ls.dimension(), size_t(1));
        decltype(ls)::shape_type expected_shape = {100};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 20.f);

        float at_end = 49.7f;
        ASSERT_EQ(ls(99), at_end);

        float at_3 = 20.9f;
        ASSERT_EQ(ls(3), at_3);

        xarray<float> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), size_t(1));
        ASSERT_EQ(m_assigned.shape()[0], size_t(100));
        ASSERT_EQ(m_assigned[{0}], 20.f);
        ASSERT_EQ(m_assigned(99), at_end);
        ASSERT_EQ(m_assigned(3), at_3);
    }

    TEST(xbuilder, linspace_endpoint)
    {
        double at_end = 99.78730976641236;
        xt::xtensor<double, 1> ls = linspace<double>(0., at_end, 100);
        ASSERT_EQ(ls(99), at_end);
    }

    TEST(xbuilder, linspace_integer)
    {
        xarray<int> ls = linspace<int>(0, 10, 13);
        xarray<int> expected = {0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10};

        ASSERT_TRUE(all(equal(ls, expected)));
    }

    TEST(xbuilder, linspace_double)
    {
        // access_imp will also be tested here
        double at_end = 99.78730976641236;
        xt::xarray<double> a = xt::linspace(0., at_end, 100);
        auto b = xt::linspace(0., at_end, 100);
        EXPECT_EQ(a, b);
    }

    TEST(xbuilder, logspace)
    {
        auto ls = logspace<double>(2., 3., 4);
        ASSERT_EQ(ls.dimension(), size_t(1));
        decltype(ls)::shape_type expected_shape = {4};
        ASSERT_EQ(ls.shape(), expected_shape);
        ASSERT_EQ(ls[{0}], 100);

        double at_1 = std::pow(10.0, (2.0 + 1.0 / 3.0));
        ASSERT_EQ(ls(1), at_1);

        ASSERT_EQ(ls(3), 1000);
        xarray<double> m_assigned = ls;
        ASSERT_EQ(m_assigned.dimension(), size_t(1));
        ASSERT_EQ(m_assigned.shape()[0], size_t(4));
        ASSERT_EQ(m_assigned[{0}], 100);
        ASSERT_EQ(m_assigned(1), at_1);
        ASSERT_EQ(m_assigned(3), 1000);
    }

    TEST(xbuilder, eye)
    {
        auto e = eye(5);
        ASSERT_EQ(size_t(2), e.dimension());
        decltype(e)::shape_type expected_shape = {5, 5};
        ASSERT_EQ(expected_shape, e.shape());

        ASSERT_TRUE(e(1, 1));
        xindex idx({1, 0});
        ASSERT_FALSE(e[idx]);

        xarray<bool> m_assigned = e;
        ASSERT_TRUE(m_assigned(2, 2));
        ASSERT_FALSE(m_assigned(4, 2));

        xindex idx2({2, 2});
        ASSERT_TRUE(e.element(idx2.begin(), idx2.end()));

        ASSERT_TRUE(e[idx2]);
        ASSERT_TRUE((e[{2, 2}]));

        auto e2 = eye(5, -1);
        EXPECT_TRUE(e2(1, 0));
        EXPECT_FALSE(e2(0, 0));
    }

    TEST(xbuilder, concatenate)
    {
        xarray<double> a = {{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}};

        auto c = concatenate(xtuple(a, a, a), 2);

        shape_t expected_shape = {2, 2, 9};
        ASSERT_EQ(expected_shape, c.shape());
        ASSERT_EQ(c(1, 1, 2), c(1, 1, 5));
        ASSERT_EQ(11, c(1, 1, 2));
        ASSERT_EQ(11, c(1, 1, 5));

        xarray<double> e = {{1, 2, 3}};
        xarray<double> f = {{2, 3, 4}};
        xarray<double> k = concatenate(xtuple(e, f));
        xarray<double> l = concatenate(xtuple(e, f), 1);

        shape_t ex_k = {2, 3};
        shape_t ex_l = {1, 6};
        ASSERT_EQ(ex_k, k.shape());
        ASSERT_EQ(ex_l, l.shape());
        ASSERT_EQ(4, k(1, 2));
        ASSERT_EQ(3, l(0, 2));
        ASSERT_EQ(3, l(0, 4));

        auto t = concatenate(xtuple(arange(2), arange(2, 5), arange(5, 8)));
        ASSERT_TRUE(arange(8) == t);
        
        xt::xarray<double> fa = xt::ones<double>({ 3, 4, 5, 0 });
        xt::xarray<double> sa = xt::ones<double>({ 3, 4, 5 });
        xt::xarray<double> ta = xt::ones<double>({ 3, 4, 5, 3 });

        XT_EXPECT_ANY_THROW(xt::concatenate(xt::xtuple(fa, sa)));
        XT_EXPECT_ANY_THROW(xt::concatenate(xt::xtuple(fa, ta)));
    }

#ifndef VS_SKIP_CONCATENATE_FIXED
    // This test mimics the relevant parts of `TEST(xbuilder, concatenate)`
    TEST(xbuilder, concatenate_fixed)
    {
        xtensor_fixed<double, fixed_shape<2, 2, 3>> a = {{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}};

        auto c = concatenate<2>(xtuple(a, a, a));

        using expected_shape_c_t = fixed_shape<2, 2, 9>;
        ASSERT_EQ(expected_shape_c_t{}, c.shape());
        ASSERT_EQ(c(1, 1, 2), c(1, 1, 5));
        ASSERT_EQ(11, c(1, 1, 2));
        ASSERT_EQ(11, c(1, 1, 5));

        xtensor_fixed<double, fixed_shape<1, 3>> e = {{1, 2, 3}}, f = {{2, 3, 4}};
        auto k = concatenate<0>(xtuple(e, f));
        auto l = concatenate<1>(xtuple(e, f));

        using expected_shape_k_t = fixed_shape<2, 3>;
        using expected_shape_l_t = fixed_shape<1, 6>;
        ASSERT_EQ(expected_shape_k_t{}, k.shape());
        ASSERT_EQ(expected_shape_l_t{}, l.shape());
        ASSERT_EQ(4, k(1, 2));
        ASSERT_EQ(3, l(0, 2));
        ASSERT_EQ(3, l(0, 4));

        xtensor_fixed<double, fixed_shape<2>> x = arange(2);
        xtensor_fixed<double, fixed_shape<3>> y = arange(2, 5);
        xtensor_fixed<double, fixed_shape<3>> z = arange(5, 8);

        auto w1 = concatenate<0>(xtuple(x, y, z));
        auto w2 = concatenate(xtuple(x, y, z), 0);

        ASSERT_TRUE(arange(8) == w1);
        ASSERT_TRUE(w1 == w2);
    }
#endif

    TEST(xbuilder, access)
    {
        xarray<double> a = { { { 0, 1, 2 },{ 3, 4, 5 } },{ { 6, 7, 8 },{ 9, 10, 11 } } };
        auto c = concatenate(xtuple(a, a, a), 2);
        EXPECT_EQ(c(2, 3, 1, 1, 2), c(1, 1, 2));
    }

    TEST(xbuilder, unchecked)
    {
        auto ls = linspace<float>(20.f, 50.f, 100, false);
        EXPECT_EQ(ls.unchecked(10), ls(10));
    }

    TEST(xbuilder, stack)
    {
        xarray<double> a = {{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}};

        auto c = stack(xtuple(a, a, a), 2);

        shape_t expected_shape = {2, 2, 3, 3};

        ASSERT_EQ(expected_shape, c.shape());
        ASSERT_EQ(c(1, 1, 0, 2), c(1, 1, 1, 2));
        ASSERT_EQ(c(1, 1, 0, 2), c(1, 1, 2, 2));
        ASSERT_EQ(11, c(1, 1, 1, 2));
        ASSERT_EQ(11, c(1, 1, 2, 2));

        auto e = arange(1, 4);
        xarray<double> f = {2, 3, 4};
        xarray<double> k = stack(xtuple(e, f));
        xarray<double> l = stack(xtuple(e, f), 1);

        shape_t ex_k = {2, 3};
        shape_t ex_l = {3, 2};
        ASSERT_EQ(ex_k, k.shape());
        ASSERT_EQ(ex_l, l.shape());
        ASSERT_EQ(4, k(1, 2));
        ASSERT_EQ(3, l(1, 1));
        ASSERT_EQ(3, l(2, 0));

        auto t = stack(xtuple(arange(3), arange(3, 6), arange(6, 9)));
        xarray<double> ar = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
        ASSERT_TRUE(t == ar);
    }

    TEST(xbuilder, hstack)
    {
        xarray<int> a0 = {1, 2, 3};
        xarray<int> b0 = {2, 3, 4};
        xarray<int> e0 = {1, 2, 3, 2, 3, 4};
        auto c0 = hstack(xtuple(a0, b0));
        EXPECT_EQ(c0, e0);

        xarray<int> a1 = a0;
        a1.reshape({3, 1});
        xarray<int> b1 = b0;
        b1.reshape({3, 1});
        xarray<int> e1 = {{1, 2}, {2, 3}, {3, 4}};
        auto c1 = hstack(xtuple(a1, b1));
        EXPECT_EQ(c1, e1);

        xarray<int> a2 = {{1, 2, 3}, {4, 5 ,6}};
        xarray<int> b2 = {{7, 8}, {9, 10}};
        xarray<int> e2 = {{1, 2, 3, 7, 8}, {4, 5, 6, 9, 10}};
        auto c2 = hstack(xtuple(a2, b2));
    }

    TEST(xbuilder, vstack)
    {
        xarray<int> a0 = {1, 2, 3};
        xarray<int> b0 = {2, 3, 4};
        xarray<int> e0 = {{1, 2, 3}, {2, 3, 4}};
        auto c0 = vstack(xtuple(a0, b0));
        EXPECT_EQ(c0, e0);

        xarray<int> a1 = a0;
        a1.reshape({3, 1});
        xarray<int> b1 = b0;
        b1.reshape({3, 1});
        xarray<int> e1 = { 1, 2, 3, 2, 3, 4 };
        e1.reshape({6, 1});
        auto c1 = vstack(xtuple(a1, b1));
        EXPECT_EQ(c1, e1);

        xarray<int> a2 = {{1, 2, 3}, {4, 5 ,6}, {7, 8, 9}};
        xarray<int> b2 = {{10, 11, 12}};
        xarray<int> e2 = {{1, 2, 3}, {4, 5 ,6}, {7, 8, 9}, {10, 11, 12}};
        auto c2 = vstack(xtuple(a2, b2));
        EXPECT_EQ(c2, e2);
    }

    TEST(xbuilder, meshgrid)
    {
        auto mesh = meshgrid(linspace<double>(0.0, 1.0, 3), linspace<double>(0.0, 1.0, 2));
        xarray<double> expect0 = {{0, 0}, {0.5, 0.5}, {1, 1}};
        xarray<double> expect1 = {{0, 1}, {0, 1}, {0, 1}};
        ASSERT_TRUE(all(equal(std::get<0>(mesh), expect0)));
        ASSERT_TRUE(all(equal(std::get<1>(mesh), expect1)));
    }

    TEST(xbuilder, meshgrid_arange)
    {
        auto xrange = xt::arange(0, 2);
        auto yrange = xt::arange(0, 2);
        auto grid = xt::meshgrid(xrange, yrange);
        std::ostringstream stream;
        stream << std::get<0>(grid) << std::endl;
    }

    TEST(xbuilder, triu)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> t = xt::triu(e);

        xarray<double> expected = {{1, 2, 3},
                                   {0, 5, 6},
                                   {0, 0, 9}};

        xarray<double> expected_2 = {{1, 2, 3},
                                     {4, 5, 6},
                                     {0, 8, 9}};

        xarray<double> expected_3 = {{0, 2, 3},
                                     {0, 0, 6},
                                     {0, 0, 0}};

        ASSERT_EQ(size_t(2), t.dimension());
        shape_t expected_shape = {3, 3};
        ASSERT_EQ(expected_shape, t.shape());

        ASSERT_EQ(expected, t);

        xarray<double> t3 = xt::triu(e, 1);
        ASSERT_EQ(expected_3, t3);

        xarray<double> t2 = xt::triu(e, -1);
        ASSERT_EQ(expected_2, t2);
    }

    TEST(xbuilder, tril)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> t = xt::tril(e);

        xarray<double> expected = {{1, 0, 0},
                                   {4, 5, 0},
                                   {7, 8, 9}};

        xarray<double> expected_2 = {{1, 2, 0},
                                     {4, 5, 6},
                                     {7, 8, 9}};

        xarray<double> expected_3 = {{0, 0, 0},
                                     {4, 0, 0},
                                     {7, 8, 0}};

        ASSERT_EQ(size_t(2), t.dimension());
        shape_t expected_shape = {3, 3};
        ASSERT_EQ(expected_shape, t.shape());

        ASSERT_EQ(expected, t);

        xarray<double> t2 = xt::tril(e, 1);
        ASSERT_EQ(expected_2, t2);

        xarray<double> t3 = xt::tril(e, -1);
        ASSERT_EQ(expected_3, t3);
    }

    TEST(xbuilder, diagonal)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

        xarray<double> t = xt::diagonal(e);

        xarray<double> expected = {1, 5, 9};
        ASSERT_EQ(expected, t);

        xt::xarray<double> f = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}};

        xarray<double> exp_1 = {1, 5};
        ASSERT_TRUE(all(equal(exp_1, xt::diagonal(f, 1))));
        xarray<double> exp_2 = {0, 4, 8};
        EXPECT_EQ(exp_2, xt::diagonal(f));
        xarray<double> exp_3 = {3, 7, 11};
        EXPECT_EQ(exp_3, xt::diagonal(f, -1));
        xarray<double> exp_4 = {6, 10};
        EXPECT_EQ(exp_4, xt::diagonal(f, -2));
    }

    TEST(xbuilder, diagonal_advanced)
    {
        xarray<double> e = {{{{0, 1, 2}, {3, 4, 5}},
                             {{6, 7, 8}, {9, 10, 11}}},
                            {{{12, 13, 14}, {15, 16, 17}},
                             {{18, 19, 20}, {21, 22, 23}}}};

        xarray<double> d1 = xt::diagonal(e);

        xarray<double> expected = {{{0, 18},
                                    {1, 19},
                                    {2, 20}},
                                   {{3, 21},
                                    {4, 22},
                                    {5, 23}}};
        ASSERT_EQ(expected, d1);

        std::vector<double> d2 = {6, 7, 8, 9, 10, 11};
        xarray<double> expected_2;
        expected_2.resize({2, 3, 1});
        std::copy(d2.begin(), d2.end(), expected_2.template begin<layout_type::row_major>());

        xarray<double> t2 = xt::diagonal(e, 1);
        ASSERT_EQ(expected_2, t2);

        std::vector<double> d3 = {3, 9, 15, 21};
        xarray<double> expected_3;
        expected_3.resize({2, 2, 1});
        std::copy(d3.begin(), d3.end(), expected_3.template begin<layout_type::row_major>());
        xarray<double> t3 = xt::diagonal(e, -1, 2, 3);
        ASSERT_EQ(expected_3, t3);
    }

    TEST(xbuilder, diag)
    {
        xarray<double> e = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> t = xt::diag(xt::diagonal(e));
        xarray<double> expected = xt::eye(3) * e;

        ASSERT_EQ(expected, t);
    }

    TEST(xbuilder, arange_broadcast)
    {
        auto a = arange<int>(1);
        xarray<int> b = { 1, 2, 3 };
        xarray<int> res = a + b;
        EXPECT_EQ(res, b);
    }

    TEST(xbuilder, empty)
    {

        bool b = false;

        auto e1 = empty<double>({3, 4, 1});
        b = std::is_same<decltype(e1), xtensor<double, 3>>::value;
        EXPECT_TRUE(b);
        b = std::is_same<decltype(empty<int, layout_type::column_major>({3,3,3})),
                         xtensor<int, 3, layout_type::column_major>>::value;
        EXPECT_TRUE(b);

        auto es = empty<double>(std::array<std::size_t, 3>{3, 4, 1});
        b = std::is_same<decltype(es), xtensor<double, 3>>::value;
        EXPECT_TRUE(b);

        auto e2 = empty<double>(xshape<3, 3, 3>());
        b = std::is_same<decltype(e2), xtensor_fixed<double, xshape<3, 3, 3>>>::value;
        EXPECT_TRUE(b);

        auto shapef = xshape<3, 2>();
        auto e22 = empty<double, layout_type::column_major>(shapef);
        b = std::is_same<decltype(e22), xtensor_fixed<double, xshape<3, 2>, layout_type::column_major>>::value;
        EXPECT_TRUE(b);

        xt::dynamic_shape<std::size_t> sd = {3, 2, 1};
        auto ed1 = empty<double>(sd);
        auto ed2 = empty<double, layout_type::column_major>(dynamic_shape<std::size_t>({3, 3, 3}));
        auto ed3 = empty<double>(std::vector<std::size_t>({3, 3, 3}));
        b = std::is_same<decltype(ed1), xarray<double>>::value;
        EXPECT_TRUE(b);
        b = std::is_same<decltype(ed2), xarray<double, layout_type::column_major>>::value;
        EXPECT_TRUE(b);
        b = std::is_same<decltype(ed3), xarray<double>>::value;
        EXPECT_TRUE(b);
    }
}
