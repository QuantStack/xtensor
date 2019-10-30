/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xpad.hpp"

#include <complex>
#include <limits>

#include "gtest/gtest.h"

namespace xt
{
    TEST(xpad, constant_a)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 1, 2, 0, 0, 0},
                                    {0, 0, 0, 3, 4, 5, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0, 0}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{2,2}, {3,3}}, xt::pad_mode::constant);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, constant_b)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, 0, xt::pad_mode::constant);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, constant_c)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2, 0, 0},
                                    {3, 4, 5, 0, 0},
                                    {0, 0, 0, 0, 0}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{0,1}, {0,2}}, xt::pad_mode::constant);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, constant_d)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 0, 0, 0, 0},
                                    {0, 0, 0, 1, 2},
                                    {0, 0, 3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{1,0}, {2,0}}, xt::pad_mode::constant);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, wrap_a)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2, 0, 1, 2, 0, 1, 2},
                                    {3, 4, 5, 3, 4, 5, 3, 4, 5},
                                    {0, 1, 2, 0, 1, 2, 0, 1, 2},
                                    {3, 4, 5, 3, 4, 5, 3, 4, 5},
                                    {0, 1, 2, 0, 1, 2, 0, 1, 2},
                                    {3, 4, 5, 3, 4, 5, 3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{2,2}, {3,3}}, xt::pad_mode::wrap);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, wrap_b)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, 0, xt::pad_mode::wrap);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, wrap_c)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2, 0, 1},
                                    {3, 4, 5, 3, 4},
                                    {0, 1, 2, 0, 1}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{0,1}, {0,2}}, xt::pad_mode::wrap);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, wrap_d)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{4, 5, 3, 4, 5},
                                    {1, 2, 0, 1, 2},
                                    {4, 5, 3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{1,0}, {2,0}}, xt::pad_mode::wrap);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, symmetric_a)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{5, 4, 3, 3, 4, 5, 5, 4, 3},
                                    {2, 1, 0, 0, 1, 2, 2, 1, 0},
                                    {2, 1, 0, 0, 1, 2, 2, 1, 0},
                                    {5, 4, 3, 3, 4, 5, 5, 4, 3},
                                    {5, 4, 3, 3, 4, 5, 5, 4, 3},
                                    {2, 1, 0, 0, 1, 2, 2, 1, 0}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{2,2}, {3,3}}, xt::pad_mode::symmetric);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, symmetric_b)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, 0, xt::pad_mode::symmetric);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, symmetric_c)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2, 2, 1},
                                    {3, 4, 5, 5, 4},
                                    {3, 4, 5, 5, 4}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{0,1}, {0,2}}, xt::pad_mode::symmetric);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, symmetric_d)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{1, 0, 0, 1, 2},
                                    {1, 0, 0, 1, 2},
                                    {4, 3, 3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{1,0}, {2,0}}, xt::pad_mode::symmetric);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, reflect_a)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{5, 4, 3, 4, 5, 4, 3},
                                    {2, 1, 0, 1, 2, 1, 0},
                                    {5, 4, 3, 4, 5, 4, 3},
                                    {2, 1, 0, 1, 2, 1, 0}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{1,1}, {2,2}}, xt::pad_mode::reflect);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, reflect_b)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, 0, xt::pad_mode::reflect);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, reflect_c)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{0, 1, 2, 1, 0},
                                    {3, 4, 5, 4, 3},
                                    {0, 1, 2, 1, 0}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{0,1}, {0,2}}, xt::pad_mode::reflect);

        EXPECT_EQ(b, c);
    }

    TEST(xpad, reflect_d)
    {
        xt::xtensor<size_t, 2> a = {{0, 1, 2},
                                    {3, 4, 5}};

        xt::xtensor<size_t, 2> b = {{5, 4, 3, 4, 5},
                                    {2, 1, 0, 1, 2},
                                    {5, 4, 3, 4, 5}};

        xt::xtensor<size_t, 2> c = xt::pad(a, {{1,0}, {2,0}}, xt::pad_mode::reflect);

        EXPECT_EQ(b, c);
    }
}
