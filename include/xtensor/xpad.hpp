/****************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay,  Wolf Vollprecht and  *
 * Martin Renou                                                             *
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
     * @brief Defines different algorithms to be used in ``xt::pad``
     */
    enum class pad_mode
    {
        constant,
        symmetric,
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

        auto out = e;

        for (size_type axis = 0; axis < e.shape().size(); ++axis)
        {
            size_type nb = static_cast<size_type>(pad_width[axis][0]);
            size_type ne = static_cast<size_type>(pad_width[axis][1]);

            if (mode == pad_mode::constant)
            {
                auto shape_bgn = out.shape();
                auto shape_end = out.shape();
                shape_bgn[axis] = nb;
                shape_end[axis] = ne;
                auto bgn = constant_value * xt::ones<value_type>(shape_bgn);
                auto end = constant_value * xt::ones<value_type>(shape_end);
                out = xt::concatenate(xt::xtuple(bgn, out, end), axis);
            }
            else
            {
                xt::xstrided_slice_vector sv_bgn(e.shape().size(), xt::all());
                xt::xstrided_slice_vector sv_end(e.shape().size(), xt::all());

                if (mode == pad_mode::periodic)
                {
                    XTENSOR_ASSERT(nb <= out.shape()[axis]);
                    XTENSOR_ASSERT(ne <= out.shape()[axis]);
                    sv_bgn[axis] = xt::range(out.shape()[axis] - nb, out.shape()[axis]);
                    sv_end[axis] = xt::range(0, ne);
                }
                else if (mode == pad_mode::symmetric)
                {
                    XTENSOR_ASSERT(nb <= out.shape()[axis]);
                    XTENSOR_ASSERT(ne <= out.shape()[axis]);
                    if (nb == (size_type) 0)
                    {
                        sv_bgn[axis] = xt::range(0, 0, -1);
                    }
                    else
                    {
                        sv_bgn[axis] = xt::range(nb - 1, _, -1);
                    }
                    if (ne == out.shape()[axis])
                    {
                        sv_end[axis] = xt::range(out.shape()[axis], _, -1);
                    }
                    else
                    {
                        sv_end[axis] = xt::range(out.shape()[axis], out.shape()[axis] - ne - 1, -1);
                    }
                }

                auto bgn = xt::strided_view(out, sv_bgn);
                auto end = xt::strided_view(out, sv_end);

                out = xt::concatenate(xt::xtuple(bgn, out, end), axis);
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
    inline auto pad(E&& e, S pad_width, pad_mode mode = pad_mode::constant, V constant_value = 0)
    {
        std::vector<std::vector<S>> pw(e.shape().size(), { pad_width, pad_width });

        return pad(e, pw, mode, constant_value);
    }
}

#endif
