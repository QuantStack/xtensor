/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xmime.hpp"


#include <iostream>

#include <nlohmann/json.hpp>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

namespace xt
{
    TEST(xmime, xarray_two_d)
    {
        xarray<double> e{{1, 2, 3, 4},
                         {5, 6, 7, 8},
                         {9, 10, 11, 12}};

        nlohmann::json ser = mime_bundle_repr_impl(e);

        nlohmann::json ref = {{"text/html", "<table style='border-style:solid;border-width:1px;'><tbody><tr><td style='font-family:monospace;' title='(0, 0)'><pre>  1.</pre></td><td style='font-family:monospace;' title='(0, 1)'><pre>  2.</pre></td><td style='font-family:monospace;' title='(0, 2)'><pre>  3.</pre></td><td style='font-family:monospace;' title='(0, 3)'><pre>  4.</pre></td></tr><tr><td style='font-family:monospace;' title='(1, 0)'><pre>  5.</pre></td><td style='font-family:monospace;' title='(1, 1)'><pre>  6.</pre></td><td style='font-family:monospace;' title='(1, 2)'><pre>  7.</pre></td><td style='font-family:monospace;' title='(1, 3)'><pre>  8.</pre></td></tr><tr><td style='font-family:monospace;' title='(2, 0)'><pre>  9.</pre></td><td style='font-family:monospace;' title='(2, 1)'><pre> 10.</pre></td><td style='font-family:monospace;' title='(2, 2)'><pre> 11.</pre></td><td style='font-family:monospace;' title='(2, 3)'><pre> 12.</pre></td></tr></tbody></table>"}};

        EXPECT_EQ(ser, ref);
    }
}
