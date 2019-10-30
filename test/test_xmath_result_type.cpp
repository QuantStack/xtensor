/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <complex>

#include "gtest/gtest.h"

// The following disables the conversion warnings. These warnings
// are legit and we don't want to avoid them with specific cast
// in xtensor implementation. However, we still want to check the
// results are correct and we don't want the warnings to pollute
// the output when building the tests suite.
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#pragma GCC diagnostic pop
#elif defined(_WIN32)
#pragma warning(push)
#pragma warning(disable: 4244)
#pragma warning(disable: 4267)
#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#pragma warning(pop)
#else
#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#endif

namespace xt
{
    using std::size_t;
    using shape_type = dynamic_shape<size_t>;

    /*******************
     * type conversion *
     *******************/

#define CHECK_RESULT_TYPE(EXPRESSION, EXPECTED_TYPE)                                 \
    {                                                                                \
        using result_type = typename std::decay_t<decltype(EXPRESSION)>::value_type; \
        EXPECT_TRUE((std::is_same<result_type, EXPECTED_TYPE>::value));              \
    }
#define ARRAY_TYPE(VALUE_TYPE)  \
    std::array<VALUE_TYPE, 2>

#define CHECK_TEMPLATED_RESULT_TYPE(FUNC, INPUT)                                     \
        CHECK_RESULT_TYPE(FUNC<unsigned char>(INPUT), int);                          \
        CHECK_RESULT_TYPE(FUNC<signed char>(INPUT), int);                            \
        CHECK_RESULT_TYPE(FUNC<char>(INPUT), int);                                   \
        CHECK_RESULT_TYPE(FUNC<unsigned short>(INPUT), int);                         \
        CHECK_RESULT_TYPE(FUNC<signed short>(INPUT), int);                           \
        CHECK_RESULT_TYPE(FUNC<short>(INPUT), int);                                  \
        CHECK_RESULT_TYPE(FUNC<unsigned int>(INPUT), unsigned int);                  \
        CHECK_RESULT_TYPE(FUNC<signed int>(INPUT), signed int);                      \
        CHECK_RESULT_TYPE(FUNC<int>(INPUT), int);                                    \
        CHECK_RESULT_TYPE(FUNC<unsigned long long>(INPUT), unsigned long long);      \
        CHECK_RESULT_TYPE(FUNC<signed long long>(INPUT), signed long long);

    TEST(xmath, result_type)
    {
        shape_type shape = {3, 2};
        xarray<unsigned char> auchar(shape);
        xarray<short> ashort(shape);
        xarray<int> aint(shape);
        xarray<unsigned int> auint(shape);
        xarray<unsigned long long> aulong(shape);
        xarray<float> afloat(shape);
        xarray<double> adouble(shape);
        xarray<std::complex<float>> afcomplex(shape);
        xarray<std::complex<double>> adcomplex(shape);

        /*****************
         * unsigned char *
         *****************/
        CHECK_RESULT_TYPE(auchar + auchar, int);
        CHECK_RESULT_TYPE(2 * auchar, int);
        CHECK_RESULT_TYPE(2.0 * auchar, double);
        CHECK_RESULT_TYPE(sqrt(auchar), double);
        CHECK_RESULT_TYPE(abs(auchar), unsigned char);
        CHECK_RESULT_TYPE(sum(auchar), unsigned long long);
        CHECK_RESULT_TYPE(mean(auchar), double);
        CHECK_RESULT_TYPE(minmax(auchar), ARRAY_TYPE(unsigned char));
        CHECK_TEMPLATED_RESULT_TYPE(mean, auchar);

        /*********
         * short *
         *********/
        CHECK_RESULT_TYPE(ashort + ashort, int);
        CHECK_RESULT_TYPE(2 * ashort, int);
        CHECK_RESULT_TYPE(2.0 * ashort, double);
        CHECK_RESULT_TYPE(sqrt(ashort), double);
        CHECK_RESULT_TYPE(abs(ashort), decltype(std::abs(short{})));
        CHECK_RESULT_TYPE(sum(ashort), long long);
        CHECK_RESULT_TYPE(mean(ashort), double);
        CHECK_RESULT_TYPE(minmax(ashort), ARRAY_TYPE(short));
        CHECK_TEMPLATED_RESULT_TYPE(mean, ashort);

        /*******
         * int *
         *******/
        CHECK_RESULT_TYPE(aint + aint, int);
        CHECK_RESULT_TYPE(2 * aint, int);
        CHECK_RESULT_TYPE(2.0 * aint, double);
        CHECK_RESULT_TYPE(sqrt(aint), double);
        CHECK_RESULT_TYPE(abs(aint), int);
        CHECK_RESULT_TYPE(sum(aint), long long);
        CHECK_RESULT_TYPE(mean(aint), double);
        CHECK_RESULT_TYPE(minmax(aint), ARRAY_TYPE(int));
        CHECK_TEMPLATED_RESULT_TYPE(mean, aint);

        /****************
         * unsigned int *
         ****************/
        CHECK_RESULT_TYPE(auint + auint, unsigned int);
        CHECK_RESULT_TYPE(2u * auint, unsigned int);
        CHECK_RESULT_TYPE(2.0 * auint, double);
        CHECK_RESULT_TYPE(sqrt(auint), double);
        CHECK_RESULT_TYPE(abs(auint), unsigned int);
        CHECK_RESULT_TYPE(sum(auint), unsigned long long);
        CHECK_RESULT_TYPE(mean(auint), double);
        CHECK_RESULT_TYPE(minmax(auint), ARRAY_TYPE(unsigned int));
        CHECK_TEMPLATED_RESULT_TYPE(mean, auint);

        /**********************
         * unsigned long long *
         **********************/
        CHECK_RESULT_TYPE(aulong + aulong, unsigned long long);
        CHECK_RESULT_TYPE(2ul * aulong, unsigned long long);
        CHECK_RESULT_TYPE(2.0 * aulong, double);
        CHECK_RESULT_TYPE(sqrt(aulong), double);
        CHECK_RESULT_TYPE(abs(aulong), unsigned long long);
        CHECK_RESULT_TYPE(sum(aulong), unsigned long long);
        CHECK_RESULT_TYPE(mean(aulong), double);
        CHECK_RESULT_TYPE(minmax(aulong), ARRAY_TYPE(unsigned long long));
        CHECK_TEMPLATED_RESULT_TYPE(mean, aulong);

        /*********
         * float *
         *********/
        CHECK_RESULT_TYPE(afloat + afloat, float);
        CHECK_RESULT_TYPE(2.0f * afloat, float);
        CHECK_RESULT_TYPE(2.0 * afloat, double);
        CHECK_RESULT_TYPE(sqrt(afloat), float);
        CHECK_RESULT_TYPE(abs(afloat), float);
        CHECK_RESULT_TYPE(sum(afloat), double);
        CHECK_RESULT_TYPE(mean(afloat), double);
        CHECK_RESULT_TYPE(minmax(afloat), ARRAY_TYPE(float));
        CHECK_TEMPLATED_RESULT_TYPE(mean, afloat);

        /**********
         * double *
         **********/
        CHECK_RESULT_TYPE(adouble + adouble, double);
        CHECK_RESULT_TYPE(2.0 * adouble, double);
        CHECK_RESULT_TYPE(sqrt(adouble), double);
        CHECK_RESULT_TYPE(abs(adouble), double);
        CHECK_RESULT_TYPE(sum(adouble), double);
        CHECK_RESULT_TYPE(mean(adouble), double);
        CHECK_RESULT_TYPE(minmax(adouble), ARRAY_TYPE(double));
        CHECK_TEMPLATED_RESULT_TYPE(mean, adouble);

        /***********************
         * std::complex<float> *
         ***********************/
        CHECK_RESULT_TYPE(afcomplex + afcomplex, std::complex<float>);
        CHECK_RESULT_TYPE(std::complex<float>(2.0) * afcomplex, std::complex<float>);
        CHECK_RESULT_TYPE(2.0f * afcomplex, std::complex<float>);
        CHECK_RESULT_TYPE(sqrt(afcomplex), std::complex<float>);
        CHECK_RESULT_TYPE(abs(afcomplex), float);
        CHECK_RESULT_TYPE(sum(afcomplex), std::complex<double>);
        CHECK_RESULT_TYPE(mean(afcomplex), std::complex<double>);

        /************************
         * std::complex<double> *
         ************************/
        CHECK_RESULT_TYPE(adcomplex + adcomplex, std::complex<double>);
        CHECK_RESULT_TYPE(std::complex<double>(2.0) * adcomplex, std::complex<double>);
        CHECK_RESULT_TYPE(2.0 * adcomplex, std::complex<double>);
        CHECK_RESULT_TYPE(sqrt(adcomplex), std::complex<double>);
        CHECK_RESULT_TYPE(abs(adcomplex), double);
        CHECK_RESULT_TYPE(sum(adcomplex), std::complex<double>);
        CHECK_RESULT_TYPE(mean(adcomplex), std::complex<double>);

        /***************
         * mixed types *
         ***************/
        CHECK_RESULT_TYPE(auchar + aint, int);
        CHECK_RESULT_TYPE(ashort + aint, int);
        CHECK_RESULT_TYPE(aulong + aint, unsigned long long);
        CHECK_RESULT_TYPE(afloat + aint, float);
        CHECK_RESULT_TYPE(adouble + aint, double);
        CHECK_RESULT_TYPE(adouble + adcomplex, std::complex<double>);
        CHECK_RESULT_TYPE(aulong + adouble, double);
    }


}

