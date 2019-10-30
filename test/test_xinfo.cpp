/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xinfo.hpp"

#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

namespace xt
{
    TEST(xinfo, compiles)
    {
    	xarray<double> test = {{1,2,3}, {4,5,6}};
    	std::stringstream ss;

    	ss << info(test) << std::endl;
    }

    TEST(xinfo, typename)
    {
    	xarray<double> test = {{1,2,3}, {4,5,6}};
    	auto t_s = type_to_string<typename decltype(test)::value_type>();
    	std::string expected = "double";
    	EXPECT_EQ(expected, t_s);
    }
}
