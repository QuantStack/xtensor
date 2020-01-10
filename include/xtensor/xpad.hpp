/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_PAD_HPP
#define XTENSOR_PAD_HPP

#include "xarray.hpp"
#include "xtensor.hpp"
#include "xview.hpp"
#include "xstrided_view.hpp"

using namespace xt::placeholders;  // to enable _ syntax

namespace xt
{
    /**
     * @brief Defines different algorithms to be used in ``xt::pad``:
     * - ``constant``: Pads with a constant value.
     * - ``symmetric``: Pads with the reflection of the vector mirrored along the edge of the array.
     * - ``reflect``: Pads with the reflection of the vector mirrored on the first and last values
     *   of the vector along each axis.
     * - ``wrap``: Pads with the wrap of the vector along the axis. The first values are used to pad
     *   the end and the end values are used to pad the beginning.
     * - ``periodic`` : ``== wrap`` (pads with periodic repetitions of the vector).
     *
     * OpenCV to xtensor:
     * - ``BORDER_CONSTANT == constant``
     * - ``BORDER_REFLECT == symmetric``
     * - ``BORDER_REFLECT_101 == reflect``
     * - ``BORDER_WRAP == wrap``
     */
    enum class pad_mode
    {
        constant,
        symmetric,
        reflect,
        wrap,
        periodic
    };

    namespace detail
    {
        template <class S, class T>
        inline bool check_pad_width(const std::vector<std::vector<S>>& pad_width, const T& shape)
        {
            if (pad_width.size() != shape.size())
            {
                return false;
            }

            return true;
        }
    }

    /**
     * @brief Pad an array.
     *
     * @param e The array.
     * @param pad_width Number of values padded to the edges of each axis:
     * `{{before_1, after_1}, ..., {before_N, after_N}}`.
     * @param mode The type of algorithm to use. [default: `xt::pad_mode::constant`].
     * @param constant_value The value to set the padded values for each axis
     * (used in `xt::pad_mode::constant`).
     * @return The padded array.
     */
    template <class E,
              class S = typename std::decay_t<E>::size_type,
              class V = typename std::decay_t<E>::value_type>
    inline auto pad(E&& e,
                    const std::vector<std::vector<S>>& pad_width,
                    pad_mode mode = pad_mode::constant,
                    V constant_value = 0)
    {
        XTENSOR_ASSERT(detail::check_pad_width(pad_width, e.shape()));

        using size_type = typename std::decay_t<E>::size_type;
        using value_type = typename std::decay_t<E>::value_type;
        using return_type = temporary_type_t<value_type,
                                             typename std::decay_t<E>::shape_type,
                                             std::decay_t<E>::static_layout>;

        // place the original array in the center

        auto new_shape = e.shape();
        xt::xstrided_slice_vector sv;
        for (size_type axis = 0; axis < e.shape().size(); ++axis)
        {
            size_type nb = static_cast<size_type>(pad_width[axis][0]);
            size_type ne = static_cast<size_type>(pad_width[axis][1]);
            size_type ns = nb + e.shape(axis) + ne;
            new_shape[axis] = ns;
            sv.push_back(xt::range(nb, nb + e.shape(axis)));
        }

        if (mode == pad_mode::constant)
        {
            return_type out(new_shape, constant_value);
            xt::strided_view(out, sv) = e;
            return out;
        }

        return_type out(new_shape);
        xt::strided_view(out, sv) = e;

        // construct padded regions based on original image

        for (size_type axis = 0; axis < e.shape().size(); ++axis)
        {
            size_type nb = static_cast<size_type>(pad_width[axis][0]);
            size_type ne = static_cast<size_type>(pad_width[axis][1]);

            if (nb > static_cast<size_type>(0))
            {
                xt::xstrided_slice_vector svs(e.shape().size(), xt::all());
                xt::xstrided_slice_vector svt(e.shape().size(), xt::all());

                svt[axis] = xt::range(0, nb);

                if (mode == pad_mode::wrap || mode == pad_mode::periodic)
                {
                    XTENSOR_ASSERT(nb <= e.shape(axis));
                    svs[axis] = xt::range(e.shape(axis), nb+e.shape(axis));
                }
                else if (mode == pad_mode::symmetric)
                {
                    XTENSOR_ASSERT(nb <= e.shape(axis));
                    svs[axis] = xt::range(2*nb-1, nb-1, -1);
                }
                else if (mode == pad_mode::reflect)
                {
                    XTENSOR_ASSERT(nb <= e.shape(axis) - 1);
                    svs[axis] = xt::range(2*nb, nb, -1);
                }

                xt::strided_view(out, svt) = xt::strided_view(out, svs);
            }

            if (ne > static_cast<size_type>(0))
            {
                xt::xstrided_slice_vector svs(e.shape().size(), xt::all());
                xt::xstrided_slice_vector svt(e.shape().size(), xt::all());

                svt[axis] = xt::range(out.shape(axis)-ne, out.shape(axis));

                if (mode == pad_mode::wrap || mode == pad_mode::periodic)
                {
                    XTENSOR_ASSERT(ne <= e.shape(axis));
                    svs[axis] = xt::range(nb, nb+ne);
                }
                else if (mode == pad_mode::symmetric)
                {
                    XTENSOR_ASSERT(ne <= e.shape(axis));
                    if (ne == nb + e.shape(axis))
                    {
                        svs[axis] = xt::range(nb+e.shape(axis)-1, _, -1);
                    }
                    else
                    {
                        svs[axis] = xt::range(nb+e.shape(axis)-1, nb+e.shape(axis)-ne-1, -1);
                    }
                }
                else if (mode == pad_mode::reflect)
                {
                    XTENSOR_ASSERT(ne <= e.shape(axis) - 1);
                    if (ne == nb + e.shape(axis) - 1)
                    {
                        svs[axis] = xt::range(nb+e.shape(axis)-2, _, -1);
                    }
                    else
                    {
                        svs[axis] = xt::range(nb+e.shape(axis)-2, nb+e.shape(axis)-ne-2, -1);
                    }
                }

                xt::strided_view(out, svt) = xt::strided_view(out, svs);
            }
        }

        return out;
    }

    /**
     * @brief Pad an array.
     *
     * @param e The array.
     * @param pad_width Number of values padded to the edges of each axis:
     * `{before, after}`.
     * @param mode The type of algorithm to use. [default: `xt::pad_mode::constant`].
     * @param constant_value The value to set the padded values for each axis
     * (used in `xt::pad_mode::constant`).
     * @return The padded array.
     */
    template <class E,
              class S = typename std::decay_t<E>::size_type,
              class V = typename std::decay_t<E>::value_type>
    inline auto pad(E&& e,
                    const std::vector<S>& pad_width,
                    pad_mode mode = pad_mode::constant,
                    V constant_value = 0)
    {
        std::vector<std::vector<S>> pw(e.shape().size(), pad_width);

        return pad(e, pw, mode, constant_value);
    }

    /**
     * @brief Pad an array.
     *
     * @param e The array.
     * @param pad_width Number of values padded to the edges of each axis.
     * @param mode The type of algorithm to use. [default: `xt::pad_mode::constant`].
     * @param constant_value The value to set the padded values for each axis
     * (used in `xt::pad_mode::constant`).
     * @return The padded array.
     */
    template <class E,
              class S = typename std::decay_t<E>::size_type,
              class V = typename std::decay_t<E>::value_type>
    inline auto pad(E&& e,
                    S pad_width,
                    pad_mode mode = pad_mode::constant,
                    V constant_value = 0)
    {
        std::vector<std::vector<S>> pw(e.shape().size(), {pad_width, pad_width});

        return pad(e, pw, mode, constant_value);
    }
}

#endif
