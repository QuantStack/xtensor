/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xdynamic_view.hpp"

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xindex_view.hpp"

namespace xt
{
    using shape_t = std::vector<std::size_t>;
    using view_shape_type = dynamic_shape<std::size_t>;

    TEST(xdynamic_view, keep)
    {
        xarray<int> a = { {{0, 1, 2, 3},
                           {4, 5, 6, 7},
                           {8, 9, 10, 11}},
                          {{12, 13, 14, 15},
                           {16, 17, 18, 19},
                           {20, 21, 22, 23}} };

        auto view0 = dynamic_view(a, xdynamic_slice_vector({ 1, keep(0, 2), range(1, 4) }));
        EXPECT_EQ(view0.dimension(), size_t(2));
        EXPECT_EQ(view0.shape()[0], size_t(2));
        EXPECT_EQ(view0.shape()[1], size_t(3));
        EXPECT_EQ(view0(0, 0), a(1, 0, 1));
        EXPECT_EQ(view0(0, 1), a(1, 0, 2));
        EXPECT_EQ(view0(0, 2), a(1, 0, 3));
        EXPECT_EQ(view0(1, 0), a(1, 2, 1));
        EXPECT_EQ(view0(1, 1), a(1, 2, 2));
        EXPECT_EQ(view0(1, 2), a(1, 2, 3));

        auto view1 = dynamic_view(a, xdynamic_slice_vector({ all(), 1, keep(0, 2, 3) }));
        EXPECT_EQ(view1.dimension(), size_t(2));
        EXPECT_EQ(view1.shape()[0], size_t(2));
        EXPECT_EQ(view1.shape()[1], size_t(3));
        EXPECT_EQ(view1(0, 0), a(0, 1, 0));
        EXPECT_EQ(view1(0, 1), a(0, 1, 2));
        EXPECT_EQ(view1(0, 2), a(0, 1, 3));
        EXPECT_EQ(view1(1, 0), a(1, 1, 0));
        EXPECT_EQ(view1(1, 1), a(1, 1, 2));
        EXPECT_EQ(view1(1, 2), a(1, 1, 3));

        auto view2 = dynamic_view(a, { all(), 1, keep(0, 2, 3) });
        EXPECT_EQ(view1, view2);

        auto view3 = dynamic_view(a, { all(), 1, newaxis(), keep(0, 2, 3) });
        EXPECT_EQ(view3.dimension(), size_t(3));
        EXPECT_EQ(view3.shape()[0], size_t(2));
        EXPECT_EQ(view3.shape()[1], size_t(1));
        EXPECT_EQ(view3.shape()[2], size_t(3));
        EXPECT_EQ(view3(0, 0, 0), a(0, 1, 0));
        EXPECT_EQ(view3(0, 0, 1), a(0, 1, 2));
        EXPECT_EQ(view3(0, 0, 2), a(0, 1, 3));
        EXPECT_EQ(view3(1, 0, 0), a(1, 1, 0));
        EXPECT_EQ(view3(1, 0, 1), a(1, 1, 2));
        EXPECT_EQ(view3(1, 0, 2), a(1, 1, 3));

        auto view4 = dynamic_view(a, { 1, keep(0, 2), xrange<std::ptrdiff_t>(1, 4) });
        EXPECT_EQ(view0, view4);

        auto view5 = dynamic_view(a, { 1, keep(0, 2), xstepped_range<std::ptrdiff_t>(1, 4, 1) });
        EXPECT_EQ(view0, view5);
    }

    TEST(xdynamic_view, keep_iterator)
    {
        xarray<int> a = { {{0, 1, 2, 3},
                           {4, 5, 6, 7},
                           {8, 9, 10, 11}},
                          {{12, 13, 14, 15},
                           {16, 17, 18, 19},
                           {20, 21, 22, 23}} };

        auto view0 = dynamic_view(a, xdynamic_slice_vector({ 1, keep(0, 2), range(1, 4) }));
        auto iter = view0.template begin<layout_type::row_major>();
        auto iter_end = view0.template end<layout_type::row_major>();

        EXPECT_EQ(*iter, a(1, 0, 1));
        ++iter;
        EXPECT_EQ(*iter, a(1, 0, 2));
        ++iter;
        EXPECT_EQ(*iter, a(1, 0, 3));
        ++iter;
        EXPECT_EQ(*iter, a(1, 2, 1));
        ++iter;
        EXPECT_EQ(*iter, a(1, 2, 2));
        ++iter;
        EXPECT_EQ(*iter, a(1, 2, 3));
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xdynamic_view, drop)
    {
        xarray<int> a = { {{0, 1, 2, 3},
                           {4, 5, 6, 7},
                           {8, 9, 10, 11}},
                          {{12, 13, 14, 15},
                           {16, 17, 18, 19},
                           {20, 21, 22, 23}} };

        auto view0 = dynamic_view(a, xdynamic_slice_vector({ 1, drop(1), range(1, 4) }));
        EXPECT_EQ(view0.dimension(), size_t(2));
        EXPECT_EQ(view0.shape()[0], size_t(2));
        EXPECT_EQ(view0.shape()[1], size_t(3));
        EXPECT_EQ(view0(0, 0), a(1, 0, 1));
        EXPECT_EQ(view0(0, 1), a(1, 0, 2));
        EXPECT_EQ(view0(0, 2), a(1, 0, 3));
        EXPECT_EQ(view0(1, 0), a(1, 2, 1));
        EXPECT_EQ(view0(1, 1), a(1, 2, 2));
        EXPECT_EQ(view0(1, 2), a(1, 2, 3));

        auto view1 = dynamic_view(a, xdynamic_slice_vector({ all(), 1, drop(1, 2) }));
        EXPECT_EQ(view1.dimension(), size_t(2));
        EXPECT_EQ(view1.shape()[0], size_t(2));
        EXPECT_EQ(view1.shape()[1], size_t(2));
        EXPECT_EQ(view1(0, 0), a(0, 1, 0));
        EXPECT_EQ(view1(0, 1), a(0, 1, 3));
        EXPECT_EQ(view1(1, 0), a(1, 1, 0));
        EXPECT_EQ(view1(1, 1), a(1, 1, 3));

        auto view3 = dynamic_view(a, { all(), 1, newaxis(), drop(1, 2) });
        EXPECT_EQ(view3.dimension(), size_t(3));
        EXPECT_EQ(view3.shape()[0], size_t(2));
        EXPECT_EQ(view3.shape()[1], size_t(1));
        EXPECT_EQ(view3.shape()[2], size_t(2));
        EXPECT_EQ(view3(0, 0, 0), a(0, 1, 0));
        EXPECT_EQ(view3(0, 0, 1), a(0, 1, 3));
        EXPECT_EQ(view3(1, 0, 0), a(1, 1, 0));
        EXPECT_EQ(view3(1, 0, 1), a(1, 1, 3));
    }

    TEST(xdynamic_view, drop_iterator)
    {
        xarray<int> a = { {{0, 1, 2, 3},
                           {4, 5, 6, 7},
                           {8, 9, 10, 11}},
                          {{12, 13, 14, 15},
                           {16, 17, 18, 19},
                           {20, 21, 22, 23}} };

        auto view0 = dynamic_view(a, xdynamic_slice_vector({ 1, drop(1), range(1, 4) }));
        auto iter = view0.template begin<layout_type::row_major>();
        auto iter_end = view0.template end<layout_type::row_major>();

        EXPECT_EQ(*iter, a(1, 0, 1));
        ++iter;
        EXPECT_EQ(*iter, a(1, 0, 2));
        ++iter;
        EXPECT_EQ(*iter, a(1, 0, 3));
        ++iter;
        EXPECT_EQ(*iter, a(1, 2, 1));
        ++iter;
        EXPECT_EQ(*iter, a(1, 2, 2));
        ++iter;
        EXPECT_EQ(*iter, a(1, 2, 3));
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xdynamic_view, semantic)
    {
        xarray<int> a = { {{0, 1, 2, 3},
                           {4, 5, 6, 7},
                           {8, 9, 10, 11}},
                          {{12, 13, 14, 15},
                           {16, 17, 18, 19},
                           {20, 21, 22, 23}} };

        xarray<int> b1 = { {13, 14, 15}, {21, 22, 23} };
        xarray<int> b2 = { {0, 1, 2}, {3, 4, 5} };
        xarray<int> exp = b1 + b2;
        auto view0 = dynamic_view(a, { 1, keep(0, 2), range(1, 4) });
        view0 += b2;
        EXPECT_EQ(view0, exp);
    }

    TEST(xdynamic_view, slice_conversion)
    {
        xrange<std::size_t> ru(std::size_t(1), std::size_t(12));
        xrange<std::ptrdiff_t> rs = ru;
        EXPECT_EQ(rs.size(), 11);
        EXPECT_EQ(rs(0), 1);

        auto rs2 = ru.convert<std::ptrdiff_t>();
        EXPECT_EQ(rs2, rs);

        xstepped_range<std::size_t> sru(std::size_t(1), std::size_t(5), std::size_t(2));
        xstepped_range<std::ptrdiff_t> srs = sru;
        EXPECT_EQ(srs.size(), 2);
        EXPECT_EQ(srs(0), 1);
        EXPECT_EQ(srs.step_size(), 2);

        auto srs2 = sru.convert<std::ptrdiff_t>();
        EXPECT_EQ(srs2, srs);

        xall<std::size_t> au(std::size_t(11));
        xall<std::ptrdiff_t> as(au);
        EXPECT_EQ(as.size(), 11);
        EXPECT_EQ(as(0), 0);

        auto as2  = au.convert<std::ptrdiff_t>();
        EXPECT_EQ(as2, as);

        xnewaxis<std::size_t> nau;
        xnewaxis<std::ptrdiff_t> nas(nau);
        EXPECT_EQ(nas.size(), 1);
        EXPECT_EQ(nas(0), 0);

        auto nas2 = nau.convert<std::ptrdiff_t>();
        EXPECT_EQ(nas2, nas);

        xkeep_slice<std::ptrdiff_t> ks({ 2, 3, -1 });
        ks.normalize(6);
        xkeep_slice<std::size_t> ku(ks);
        EXPECT_EQ(ku.size(), std::size_t(3));
        EXPECT_FALSE(ku.contains(0));
        EXPECT_FALSE(ku.contains(1));
        EXPECT_TRUE(ku.contains(2));
        EXPECT_TRUE(ku.contains(3));
        EXPECT_FALSE(ku.contains(4));
        EXPECT_TRUE(ku.contains(5));

        auto ks2 = ku.convert<std::size_t>();
        EXPECT_EQ(ks2, ks);

        xdrop_slice<std::ptrdiff_t> ds({ 2, 3, -1 });
        ds.normalize(6);
        xdrop_slice<std::size_t> du(ds);
        EXPECT_EQ(du.size(), std::size_t(3));
        EXPECT_TRUE(du.contains(0));
        EXPECT_TRUE(du.contains(1));
        EXPECT_FALSE(du.contains(2));
        EXPECT_FALSE(du.contains(3));
        EXPECT_TRUE(du.contains(4));
        EXPECT_FALSE(du.contains(5));

        auto ds2 = du.convert<std::size_t>();
        EXPECT_EQ(ds2, ds);
    }

    TEST(xdynamic_view, compilation_linux_issue_1349)
    {
        xt::xarray<float> array(xt::random::rand<float>({5, 4}));
        auto v = xt::view(array, xt::keep(2), xt::all());
        auto res  = xt::xarray<float>(xt::dynamic_view(v, {xt::keep(0)}));
        EXPECT_EQ(res(1), array(2, 1));
    }

    TEST(xdynamic_view, assignment)
    {
        xt::xarray<double, xt::layout_type::column_major> x = {{1, 4}, {2, 5}, {3, 6}};

        xt::xdynamic_slice_vector sv({});
        sv.push_back(xt::all());
        sv.push_back(xt::all());

        auto res = xt::dynamic_view(x, sv);
        xt::xarray<double, xt::layout_type::column_major> xrs = res;
        EXPECT_EQ(x, res);
    }

}
