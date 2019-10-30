/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xview.hpp"
#include "xtensor/xnoalias.hpp"

#include "test_xsemantic.hpp"
#include "test_common_macros.hpp"

namespace xt
{

    template <class F, class C>
    struct view_op_tester : operation_tester<F, C>
    {
        using storage_type = C;
        storage_type vres_rr;
        storage_type vres_rc;
        storage_type vres_rct;
        storage_type vres_ru;

        size_t x_slice;
        xrange<size_t> y_slice;
        xrange<size_t> z_slice;

        view_op_tester();
    };

    template <class F, class C>
    view_op_tester<F, C>::view_op_tester()
        : operation_tester<F, C>(), x_slice(0), y_slice(0, 2), z_slice(1, 4)
    {
        vres_rr = this->a;
        vres_rc = this->a;
        vres_rct = this->a;
        vres_ru = this->a;

        size_t imax = y_slice.size();
        size_t jmax = z_slice.size();
        for (size_t i = 0; i < imax; ++i)
        {
            for (size_t j = 0; j < jmax; ++j)
            {
                size_t si = y_slice(i);
                size_t sj = z_slice(j);
                vres_rr(x_slice, si, sj) = this->res_rr(x_slice, si, sj);
                vres_rc(x_slice, si, sj) = this->res_rc(x_slice, si, sj);
                vres_rct(x_slice, si, sj) = this->res_rct(x_slice, si, sj);
                vres_ru(x_slice, si, sj) = this->res_ru(x_slice, si, sj);
            }
        }
    }

    template <class C, std::size_t>
    struct redim_container
    {
        using type = C;
    };

    template <class T, std::size_t N, layout_type L, std::size_t NN>
    struct redim_container<xtensor<T, N, L>, NN>
    {
        using type = xtensor<T, NN, L>;
    };

    template <class C, std::size_t N>
    using redim_container_t = typename redim_container<C, N>::type;

    template <class C>
    class view_semantic : public ::testing::Test
    {
    public:

        using storage_type = C;
    };

    using testing_types = ::testing::Types<xarray_dynamic, xtensor_dynamic>;
    TYPED_TEST_CASE(view_semantic, testing_types);

    TYPED_TEST(view_semantic, a_plus_b)
    {
        view_op_tester<std::plus<>, TypeParam> t;
        auto viewa = view(t.a, t.x_slice, t.y_slice, t.z_slice);

        {
            SCOPED_TRACE("row_major + row_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa + viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major + column_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa + viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major + central_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa + viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major + unit_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa + viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TYPED_TEST(view_semantic, a_minus_b)
    {
        view_op_tester<std::minus<>, TypeParam> t;
        auto viewa = view(t.a, t.x_slice, t.y_slice, t.z_slice);

        {
            SCOPED_TRACE("row_major - row_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa - viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major - column_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa - viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major - central_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa - viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major - unit_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa - viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TYPED_TEST(view_semantic, a_times_b)
    {
        view_op_tester<std::multiplies<>, TypeParam> t;
        auto viewa = view(t.a, t.x_slice, t.y_slice, t.z_slice);

        {
            SCOPED_TRACE("row_major * row_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa * viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major * column_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa * viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major * central_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa * viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major * unit_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa * viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TYPED_TEST(view_semantic, a_divdide_by_b)
    {
        view_op_tester<std::divides<>, TypeParam> t;
        auto viewa = view(t.a, t.x_slice, t.y_slice, t.z_slice);

        {
            SCOPED_TRACE("row_major / row_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa / viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major / column_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa / viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major / central_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa / viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major / unit_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa / viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TYPED_TEST(view_semantic, a_plus_equal_b)
    {
        view_op_tester<std::plus<>, TypeParam> t;

        {
            SCOPED_TRACE("row_major += row_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb += viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major += column_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb += viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major += central_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb += viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major += unit_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb += viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TYPED_TEST(view_semantic, a_minus_equal_b)
    {
        view_op_tester<std::minus<>, TypeParam> t;

        {
            SCOPED_TRACE("row_major -= row_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb -= viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major -= column_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb -= viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major -= central_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb -= viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major -= unit_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb -= viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TYPED_TEST(view_semantic, a_times_equal_b)
    {
        view_op_tester<std::multiplies<>, TypeParam> t;

        {
            SCOPED_TRACE("row_major *= row_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb *= viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major *= column_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb *= viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major *= central_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb *= viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major *= unit_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb *= viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TYPED_TEST(view_semantic, a_divide_by_equal_b)
    {
        view_op_tester<std::divides<>, TypeParam> t;

        {
            SCOPED_TRACE("row_major /= row_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb /= viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major /= column_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb /= viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major /= central_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb /= viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major /= unit_major");
            TypeParam b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb /= viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TYPED_TEST(view_semantic, broadcast_equal)
    {
        using container_1d = redim_container_t<TypeParam, 1>;
        using container_2d = redim_container_t<TypeParam, 2>;
        container_2d a = {{1,  2,  3,  4},
                          {5,  6,  7,  8},
                          {9, 10, 11, 12}};
        container_2d b = a;
        auto viewa = view(a, all(), range(1, 4));
        auto viewb = view(b, all(), range(1, 4));
        container_1d c = {1, 2, 3};
        viewa = c;
        noalias(viewb) = c;
        container_2d res = {{1, 1, 2, 3},
                            {5, 1, 2, 3},
                            {9, 1, 2, 3}};

        EXPECT_EQ(res, a);
        EXPECT_EQ(res, b);
    }

    TYPED_TEST(view_semantic, scalar_equal)
    {
        using container_2d = redim_container_t<TypeParam, 2>;
        container_2d a = {{1, 2, 3, 4},
                          {5, 6, 7, 8},
                          {9, 10, 11, 12}};
        auto viewa = view(a, all(), range(1, 4));
        int b = 1;
        viewa = b;
        container_2d res = {{1, 1, 1, 1},
                            {5, 1, 1, 1},
                            {9, 1, 1, 1}};

        EXPECT_EQ(res, a);
    }

    TYPED_TEST(view_semantic, higher_dimension_broadcast)
    {
        using container_2d = redim_container_t<TypeParam, 2>;

        container_2d a = { {1, 2, 3}, {4, 5, 6} };
        container_2d b = { {11, 12, 13} };
        container_2d res = { { 11, 12, 13 }, { 4, 5, 6 } };

        auto viewa = view(a, 0, all());
        EXPECT_ANY_THROW(viewa = b);
    }
}
