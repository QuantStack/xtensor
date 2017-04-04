/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_FORWARD_HPP
#define XTENSOR_FORWARD_HPP

#include "xtensor_config.hpp"
#include "xstorage.hpp"
#include <memory>
#include <vector>

namespace xt
{
    template <class C>
    struct xcontainer_inner_types;

    template <class EC, class L, class SC = DEFAULT_SHAPE_CONTAINER(typename EC::value_type,
                                                                    typename EC::allocator_type,
                                                                    std::allocator<typename EC::size_type>)>
    class xarray_container;

    namespace layout_type
    {
        struct dynamic;
        struct row_major;
        struct column_major;
    }
    /**
     * @typedef xarray
     * Alias template on xarray_container with default parameters for data container
     * type and shape / strides container type. This allows to write
     *
     * \code{.cpp}
     * xt::xarray<double> a = {{1., 2.}, {3., 4.}};
     * \endcode
     *
     * instead of the heavier syntax
     *
     * \code{.cpp}
     * xt::xarray_container<std::vector<double>, std::vector<std::size_t>> a = ...
     * \endcode
     * 
     * @tparam T The value type of the elements.
     * @tparam A The allocator of the container holding the elements.
     * @tparam SA The allocator of the containers holding the shape and the strides.
     */
    template <class T, class L = layout_type::row_major, class A = std::allocator<T>, class SA = std::allocator<typename std::vector<T, A>::size_type>>
    using xarray = xarray_container<DEFAULT_DATA_CONTAINER(T, A), L, DEFAULT_SHAPE_CONTAINER(T, A, SA)>;

    template <class EC, std::size_t N>
    class xtensor_container;

    /**
     * @typedef xtensor
     * Alias template on xtensor_container with default parameters for data container
     * type. This allows to write
     *
     * \code{.cpp}
     * xt::xtensor<double, 2> a = {{1., 2.}, {3., 4.}};
     * \endcode
     *
     * instead of the heavier syntax
     *
     * \code{.cpp}
     * xt::xtensor_container<std::vector<double>, 2> a = ...
     * \endcode
     *
     * @tparam T The value type of the elements.
     * @tparam N The dimension of the tensor.
     * @tparam A The allocator of the containers holding the elements.
     */
    template <class T, std::size_t N, class A = std::allocator<T>>
    using xtensor = xtensor_container<DEFAULT_DATA_CONTAINER(T, A), N>;

    template <class CT, class... S>
    class xview;

    template <class T, class A, class BA>
    class xoptional_vector;

    /**
     * @typedef xarray_optional
     * Alias template on xarray_container for handling missing values
     *
     * @tparam T The value type of the elements.
     * @tparam A The allocator of the container holding the elements.
     * @tparam BA The allocator of the container holding the missing flags.
     * @tparam SA The allocator of the containers holding the shape and the strides.
     */
    template <class T, class A = std::allocator<T>, class BA = std::allocator<bool>, class SA = std::allocator<typename std::vector<T, A>::size_type>>
    using xarray_optional = xarray_container<xoptional_vector<T, A, BA>, DEFAULT_SHAPE_CONTAINER(T, A, SA)>;

    /**
     * @typedef xtensor_optional
     * Alias template on xtensor_container for handling missing values
     *
     * @tparam T The value type of the elements.
     * @tparam N The dimension of the tensor.
     * @tparam A The allocator of the containers holding the elements.
     * @tparam BA The allocator of the container holding the missing flags.
     */
    template <class T, std::size_t N, class A = std::allocator<T>, class BA = std::allocator<bool>>
    using xtensor_optional = xtensor_container<xoptional_vector<T, A, BA>, N>;
}

#endif
