/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xcsv.hpp"

#include <sstream>
#include <iostream>

#include "gtest/gtest.h"
#include "xtensor/xmath.hpp" 
#include "xtensor/xio.hpp" 

namespace xt
{
    TEST(xcsv, load_double)
    {
        std::string source =
            "1.0, 2.0, 3.0, 4.0\n"
            "10.0, 12.0, 15.0, 18.0";

        std::stringstream source_stream(source);

        xtensor<double, 2> res = load_csv<double>(source_stream);

        xtensor<double, 2> exp
            {{ 1.0,  2.0,  3.0,  4.0},
             {10.0, 12.0, 15.0, 18.0}};

        ASSERT_TRUE(all(equal(res, exp)));
    }

    TEST(xcsv, load_double_with_options)
    {
        std::string source =
            "A B C D\n"
            "#0.0 1.0 1.1 1.2\n"
            "1.0 2.0 3.0 4.0\n"
            "10.0 12.0 15.0 18.0\n"
            "9.0, 8.0, 7.0, 6.";

        std::stringstream source_stream(source);

        auto res = load_csv<double>(source_stream, ' ', 1, 2, "#");

        xtensor<double, 2> exp
            {{ 1.0,  2.0,  3.0,  4.0},
             {10.0, 12.0, 15.0, 18.0}};

        ASSERT_TRUE(all(equal(res, exp)));
    }

    TEST(xcsv, dump_double)
    {
        xtensor<double, 2> data
            {{ 1.0,  2.0,  3.0,  4.0},
             {10.0, 12.0, 15.0, 18.0}};

        std::stringstream res;

        dump_csv(res, data);
        ASSERT_EQ("1,2,3,4\n10,12,15,18\n", res.str());
    }
}
