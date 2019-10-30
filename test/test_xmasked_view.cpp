/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xmasked_view.hpp"

#include "gtest/gtest.h"
#include "xtensor/xoptional_assembly.hpp"
#include "xtensor/xio.hpp"
#include "test_common_macros.hpp"

namespace xt
{
    using data_type = xoptional_assembly<xarray<double>, xarray<bool>>;

    // data = {{ 1. ,  2., N/A },
    //         { N/A,  5.,  6. },
    //         { 7. ,  8.,  9. }}
    inline data_type make_test_data()
    {
        data_type d = {{ 1., 2., 3.},
                       { 4., 5., 6.},
                       { 7., 8., 9.}};
        d(0, 2).has_value() = false;
        d(1, 0).has_value() = false;
        return d;
    }

    // masked_data = {{ 1. ,  2., N/A },
    //                { N/A, N/A, N/A },
    //                { 7. ,  8.,  9. }}
    inline auto make_masked_data(data_type& data)
    {
        xarray<bool> mask = {{ true,  true,  true},
                             {false, false, false},
                             { true,  true,  true}};

        return masked_view(data, std::move(mask));
    }

    TEST(xmasked_view, dimension)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(data.dimension(), masked_data.dimension());
    }

    TEST(xmasked_view, size)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(data.size(), masked_data.size());
    }

    TEST(xmasked_view, shape)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(data.shape(), masked_data.shape());
    }

    // masked_data = {{ 1. ,  2., N/A },
    //                { N/A, N/A, N/A },
    //                { 7. ,  8.,  9. }}
    TEST(xmasked_view, access)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        auto masked_value = xtl::masked<xtl::xoptional<double, bool>>();
        EXPECT_EQ(masked_data(0, 0), 1.);
        EXPECT_EQ(masked_data(0, 1), 2.);
        EXPECT_EQ(masked_data(0, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data(1, 0), masked_value);
        EXPECT_EQ(masked_data(1, 1), masked_value);
        EXPECT_EQ(masked_data(1, 2), masked_value);
        EXPECT_EQ(masked_data(2, 0), 7.);
        EXPECT_EQ(masked_data(2, 1), 8.);
        EXPECT_EQ(masked_data(2, 2), 9.);

#ifdef XTENSOR_ENABLE_ASSERT
        EXPECT_ANY_THROW(masked_data(3, 3));
#endif
    }

    TEST(xmasked_view, at)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        auto masked_value = xtl::masked<xtl::xoptional<double, bool>>();
        EXPECT_EQ(masked_data.at(0, 0), 1.);
        EXPECT_EQ(masked_data.at(0, 1), 2.);
        EXPECT_EQ(masked_data.at(0, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(1, 0), masked_value);
        EXPECT_EQ(masked_data.at(1, 1), masked_value);
        EXPECT_EQ(masked_data.at(1, 2), masked_value);
        EXPECT_EQ(masked_data.at(2, 0), 7.);
        EXPECT_EQ(masked_data.at(2, 1), 8.);
        EXPECT_EQ(masked_data.at(2, 2), 9.);

        EXPECT_ANY_THROW(masked_data.at(3, 3));
    }

    TEST(xmasked_view, unchecked)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        auto masked_value = xtl::masked<xtl::xoptional<double, bool>>();
        EXPECT_EQ(masked_data.unchecked(0, 0), 1.);
        EXPECT_EQ(masked_data.unchecked(0, 1), 2.);
        EXPECT_EQ(masked_data.unchecked(0, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data.unchecked(1, 0), masked_value);
        EXPECT_EQ(masked_data.unchecked(1, 1), masked_value);
        EXPECT_EQ(masked_data.unchecked(1, 2), masked_value);
        EXPECT_EQ(masked_data.unchecked(2, 0), 7.);
        EXPECT_EQ(masked_data.unchecked(2, 1), 8.);
        EXPECT_EQ(masked_data.unchecked(2, 2), 9.);
    }

    TEST(xmasked_view, access2)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        auto masked_value = xtl::masked<xtl::xoptional<double, bool>>();

        auto val = masked_data[{0, 0}];
        EXPECT_EQ(val, 1.);
        auto val2 = masked_data[{1, 0}];
        EXPECT_EQ(val2, masked_value);

        auto index1 = std::array<int, 2>({0, 0});
        auto index2 = std::array<int, 2>({1, 0});
        EXPECT_EQ(masked_data[index1], 1.);
        EXPECT_EQ(masked_data[index2], masked_value);
    }

    TEST(xmasked_view, element)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        auto masked_value = xtl::masked<xtl::xoptional<double, bool>>();

        auto index1 = std::array<int, 2>({0, 0});
        auto index2 = std::array<int, 2>({1, 0});
        EXPECT_EQ(masked_data.element(index1.begin(), index1.end()), 1.);
        EXPECT_EQ(masked_data.element(index2.begin(), index2.end()), masked_value);
    }

    TEST(xmasked_view, fill)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        masked_data.fill(2.);

        auto masked_value = xtl::masked<xtl::xoptional<double, bool>>();
        EXPECT_EQ(masked_data.at(0, 0), 2.);
        EXPECT_EQ(masked_data.at(0, 1), 2.);
        EXPECT_EQ(masked_data.at(0, 2), 2.);
        EXPECT_EQ(masked_data.at(1, 0), masked_value);
        EXPECT_EQ(masked_data.at(1, 1), masked_value);
        EXPECT_EQ(masked_data.at(1, 2), masked_value);
        EXPECT_EQ(masked_data.at(2, 0), 2.);
        EXPECT_EQ(masked_data.at(2, 1), 2.);
        EXPECT_EQ(masked_data.at(2, 2), 2.);

        EXPECT_EQ(data.at(0, 0), 2.);
        EXPECT_EQ(data.at(0, 1), 2.);
        EXPECT_EQ(data.at(0, 2), 2.);
        EXPECT_EQ(data.at(1, 0), xtl::missing<double>());
        EXPECT_EQ(data.at(1, 1), 5.);
        EXPECT_EQ(data.at(1, 2), 6.);
        EXPECT_EQ(data.at(2, 0), 2.);
        EXPECT_EQ(data.at(2, 1), 2.);
        EXPECT_EQ(data.at(2, 2), 2.);
    }

    TEST(xmasked_view, non_optional_data)
    {
        xarray<double> data = {{ 1.,-2., 3.},
                               { 4., 5.,-6.},
                               { 7., 8.,-9.}};
        xarray<bool> mask = {{ true,  true,  true},
                             { true, false, false},
                             { true, false,  true}};

        auto masked_data = masked_view(data, mask);

        auto masked_value = xtl::masked<double>();

        EXPECT_EQ(masked_data(0, 0), 1.);
        EXPECT_EQ(masked_data.at(0, 1),-2.);
        EXPECT_EQ(masked_data.at(0, 2), 3.);
        EXPECT_EQ(masked_data.unchecked(1, 0), 4.);
        EXPECT_EQ(masked_data.unchecked(1, 1), masked_value);
        auto index1 = std::array<int, 2>({1, 2});
        EXPECT_EQ(masked_data[index1], masked_value);

        masked_data = 3.65;
        xarray<double> expected1 = {{3.65, 3.65, 3.65},
                                   {3.65, 5.  ,-6.  },
                                   {3.65, 8.  , 3.65}};
        EXPECT_EQ(data, expected1);

        masked_data += 3.;
        xarray<double> expected2 = {{6.65, 6.65, 6.65},
                                    {6.65, 5.  ,-6.  },
                                    {6.65, 8.  , 6.65}};
        EXPECT_EQ(data, expected2);
    }

    TEST(xmasked_view, assign)
    {
        xarray<double> data = {{ 1.,-2., 3.},
                               { 4., 5.,-6.},
                               { 7., 8.,-9.}};
        xarray<double> data2 = {{ 0.1, 0.2, 0.3},
                                { 0.4, 0.5, 0.6},
                                { 0.7, 0.8, 0.9}};
        xarray<bool> mask = {{ true,  true,  true},
                             { true, false, false},
                             { true, false,  true}};

        auto masked_data = masked_view(data, mask);

        masked_data = data2;
        xarray<double> expected1 = {{ 0.1, 0.2, 0.3},
                                    { 0.4, 5. ,-6. },
                                    { 0.7, 8. , 0.9}};
        EXPECT_EQ(data, expected1);
    }
}
