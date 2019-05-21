/****************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

/**
 * @brief construct histogram
 */

#ifndef XTENSOR_HISTOGRAM_HPP
#define XTENSOR_HISTOGRAM_HPP

#include "xtensor.hpp"
#include "xsort.hpp"

namespace xt
{
    /**
     * @ingroup histogram
     * @brief Compute the histogram of a set of data.
     *
     * @param data The data.
     * @param bin_edges The bin-edges.
     * @param weights Weight factors corresponding to each data-point.
     * @param density If true the resulting integral is normalized to 1. [default: false]
     * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
     */
    template <class R = double, class E1, class E2, class E3>
    inline auto histogram(E1&& data, E2&& bin_edges, E3&& weights, bool density = false)
    {
        // alias counter and value type
        using size_type = common_size_type_t<std::decay_t<E1>, std::decay_t<E2>, std::decay_t<E3>>;
        using value_type = typename std::decay_t<E3>::value_type;

        // basic checks
        // - rank
        XTENSOR_ASSERT(data.dimension() == 1);
        XTENSOR_ASSERT(weights.dimension() == 1);
        XTENSOR_ASSERT(bin_edges.dimension() == 1);
        // - size
        XTENSOR_ASSERT(weights.size() == data.size());
        XTENSOR_ASSERT(bin_edges.size() >= 2);
        // - bin-edges must be sorted
        XTENSOR_ASSERT(std::is_sorted(bin_edges.cbegin(), bin_edges.cend()));
        // - data must be enclosed in the bins
        XTENSOR_ASSERT(xt::amin(data)[0] >= bin_edges[0]);
        XTENSOR_ASSERT(xt::amax(data)[0] <= bin_edges[bin_edges.size() - 1]);

        // initialize output
        xt::xtensor<value_type, 1> count = xt::zeros<value_type>({ bin_edges.size() - 1 });

        // indices that sort "data"
        auto isort = xt::argsort(data);

        // index of the current bin
        size_type ibin = 0;

        // fill the histogram: loop over (sorted) data
        for (auto& idx : isort)
        {
            // - proceed to the relevant bin
            while (data[idx] >= bin_edges[ibin + 1] && ibin < bin_edges.size() - 2)
            {
                ++ibin;
            }
            // - update the count
            count[ibin] += weights[idx];
        }

        // cast type
        xt::xtensor<R, 1> prob = xt::cast<R>(count);

        // normalize
        if (density)
        {
            // - size as doubles
            R n = static_cast<R>(data.size());
            // - apply normalization
            for (size_type i = 0; i < bin_edges.size() - 1; ++i)
            {
                prob[i] /= (static_cast<R>(bin_edges[i + 1] - bin_edges[i]) * n);
            }
        }

        return prob;
    }

    /**
     * @ingroup histogram
     * @brief Compute the histogram of a set of data.
     *
     * @param data The data.
     * @param bin_edges The bin-edges.
     * @param density If true the resulting integral is normalized to 1. [default: false]
     * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
     */
    template <class E1, class E2>
    inline auto histogram(E1&& data, E2&& bin_edges, bool density = false)
    {
        using value_type = typename std::decay_t<E1>::value_type;

        auto n = data.size();

        return histogram(std::forward<E1>(data),
                         std::forward<E2>(bin_edges),
                         xt::ones<value_type>({ n }),
                         density);
    }

    /**
     * @ingroup histogram
     * @brief Compute the histogram of a set of data.
     *
     * @param data The data.
     * @param bins The number of bins. [default: 10]
     * @param density If true the resulting integral is normalized to 1. [default: false]
     * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
     */
    template <class E1>
    inline auto histogram(E1&& data, std::size_t bins = 10, bool density = false)
    {
        using value_type = typename std::decay_t<E1>::value_type;

        auto n = data.size();

        auto bin_edges = histogram_bin_edges(data, xt::ones<value_type>({ n }), bins);

        return histogram(std::forward<E1>(data),
                         std::forward<E1>(bin_edges),
                         xt::ones<value_type>({ n }),
                         density);
    }

    /**
     * @ingroup histogram
     * @brief Compute the histogram of a set of data.
     *
     * @param data The data.
     * @param bins The number of bins.
     * @param weights Weight factors corresponding to each data-point.
     * @param density If true the resulting integral is normalized to 1. [default: false]
     * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
     */
    template <class E1, class E2>
    inline auto histogram(E1&& data, std::size_t bins, E2&& weights, bool density = false)
    {
        auto bin_edges = histogram_bin_edges(data, weights, bins);

        return histogram(std::forward<E1>(data),
                         std::forward<E1>(bin_edges),
                         std::forward<E2>(weights),
                         density);
    }

    /**
     * @ingroup histogram
     * @brief Defines different algorithms to be used in "histogram_bin_edges"
     */
    enum class histogram_algorithm
    {
        automatic,
        linspace,
        logspace,
        uniform
    };

    /**
     * @ingroup histogram
     * @brief Compute the bin-edges of a histogram of a set of data using different algorithms.
     *
     * @param data The data.
     * @param weights Weight factors corresponding to each data-point.
     * @param left The lower-most edge.
     * @param right The upper-most edge.
     * @param bins The number of bins. [default: 10]
     * @param mode The type of algorithm to use. [default: "auto"]
     * @return An one-dimensional xarray<double>, length: bins+1.
     */
    template <class E1, class E2, class E3>
    inline auto histogram_bin_edges(E1&& data,
                                    E2&& weights,
                                    E3 left,
                                    E3 right,
                                    std::size_t bins = 10,
                                    histogram_algorithm mode = histogram_algorithm::automatic)
    {
        // counter and return type
        using size_type = common_size_type_t<std::decay_t<E1>, std::decay_t<E2>>;
        using value_type = typename std::decay_t<E1>::value_type;
        using weights_type = typename std::decay_t<E2>::value_type;

        // basic checks
        // - rank
        XTENSOR_ASSERT(data.dimension() == 1);
        XTENSOR_ASSERT(weights.dimension() == 1);
        // - size
        XTENSOR_ASSERT(weights.size() == data.size());
        // - bounds
        XTENSOR_ASSERT(left <= xt::amin(data)[0]);
        XTENSOR_ASSERT(right >= xt::amax(data)[0]);
        // - non-empty
        XTENSOR_ASSERT(bins > std::size_t(0));

        // act on different modes
        switch (mode)
        {
            // bins of equal width
            case histogram_algorithm::automatic:
            {
                xt::xtensor<value_type, 1> bin_edges
                    = xt::linspace<value_type>(left, right, bins + 1);
                return bin_edges;
            }

            // bins of equal width
            case histogram_algorithm::linspace:
            {
                xt::xtensor<value_type, 1> bin_edges
                    = xt::linspace<value_type>(left, right, bins + 1);
                return bin_edges;
            }

            // bins of logarithmically increasing size
            case histogram_algorithm::logspace:
            {
                using rhs_value_type
                    = std::conditional_t<std::is_integral<value_type>::value, double, value_type>;

                xtensor<value_type, 1> bin_edges = xt::cast<value_type>(
                    xt::logspace<rhs_value_type>(std::log10(left), std::log10(right), bins + 1));

                // TODO: replace above with below after 'xsimd' fix
                // xt::xtensor<value_type,1> bin_edges = xt::logspace<value_type>(
                //     std::log10(left), std::log10(right), bins+1);

                return bin_edges;
            }

            // same amount of data in each bin
            case histogram_algorithm::uniform:
            {
                // indices that sort "data"
                auto isort = xt::argsort(data);

                // histogram: all of equal 'height'
                // - height
                weights_type w
                    = xt::sum<weights_type>(weights)[0] / static_cast<weights_type>(bins);
                // - apply to all bins
                xt::xtensor<weights_type, 1> count = w * xt::ones<weights_type>({ bins });

                // take cumulative sum, to act as easy look-up
                count = xt::cumsum(count);

                // edges
                // - allocate
                std::vector<size_t> shape = { bins + 1 };
                xt::xtensor<value_type, 1> bin_edges = xtensor<value_type, 1>::from_shape(shape);
                // - first/last edge
                bin_edges[0] = left;
                bin_edges[bins] = right;
                // - cumulative weight
                weights_type cum_weight = static_cast<weights_type>(0);
                // - current bin
                size_type ibin = 0;
                // - loop to find interior bin-edges
                for (size_type i = 0; i < weights.size(); ++i)
                {
                    if (cum_weight >= count[ibin])
                    {
                        bin_edges[ibin + 1] = data[isort[i]];
                        ++ibin;
                    }
                    cum_weight += weights[isort[i]];
                }
                return bin_edges;
            }
            default:
            {
                xt::xtensor<value_type, 1> bin_edges
                    = xt::linspace<value_type>(left, right, bins + 1);
                return bin_edges;
            }
        }
    }

    /**
     * @ingroup histogram
     * @brief Compute the bin-edges of a histogram of a set of data using different algorithms.
     *
     * @param data The data.
     * @param weights Weight factors corresponding to each data-point.
     * @param bins The number of bins. [default: 10]
     * @param mode The type of algorithm to use. [default: "auto"]
     * @return An one-dimensional xarray<double>, length: bins+1.
     */
    template <class E1, class E2>
    inline auto histogram_bin_edges(E1&& data,
                                    E2&& weights,
                                    std::size_t bins = 10,
                                    histogram_algorithm mode = histogram_algorithm::automatic)
    {
        using value_type = typename std::decay_t<E1>::value_type;
        std::array<value_type, 2> left_right;
        left_right = xt::minmax(data)();

        return histogram_bin_edges(std::forward<E1>(data),
                                   std::forward<E2>(weights),
                                   left_right[0],
                                   left_right[1],
                                   bins,
                                   mode);
    }

    /**
     * @ingroup histogram
     * @brief Compute the bin-edges of a histogram of a set of data using different algorithms.
     *
     * @param data The data.
     * @param bins The number of bins. [default: 10]
     * @param mode The type of algorithm to use. [default: "auto"]
     * @return An one-dimensional xarray<double>, length: bins+1.
     */
    template <class E1>
    inline auto histogram_bin_edges(E1&& data,
                                    std::size_t bins = 10,
                                    histogram_algorithm mode = histogram_algorithm::automatic)
    {
        using value_type = typename std::decay_t<E1>::value_type;

        auto n = data.size();
        std::array<value_type, 2> left_right;
        left_right = xt::minmax(data)();

        return histogram_bin_edges(std::forward<E1>(data),
                                   xt::ones<value_type>({ n }),
                                   left_right[0],
                                   left_right[1],
                                   bins,
                                   mode);
    }

    /**
     * @ingroup histogram
     * @brief Compute the bin-edges of a histogram of a set of data using different algorithms.
     *
     * @param data The data.
     * @param left The lower-most edge.
     * @param right The upper-most edge.
     * @param bins The number of bins. [default: 10]
     * @param mode The type of algorithm to use. [default: "auto"]
     * @return An one-dimensional xarray<double>, length: bins+1.
     */
    template <class E1, class E2>
    inline auto histogram_bin_edges(E1&& data,
                                    E2 left,
                                    E2 right,
                                    std::size_t bins = 10,
                                    histogram_algorithm mode = histogram_algorithm::automatic)
    {
        using value_type = typename std::decay_t<E1>::value_type;

        auto n = data.size();

        return histogram_bin_edges(
            std::forward<E1>(data), xt::ones<value_type>({ n }), left, right, bins, mode);
    }

    /**
     * Count number of occurrences of each value in array of non-negative ints.
     *
     * The number of bins (of size 1) is one larger than the largest value in x.
     * If minlength is specified, there will be at least this number of bins in
     * the output array (though it will be longer if necessary, depending on the
     * contents of x). Each bin gives the number of occurrences of its index
     * value in x. If weights is specified the input array is weighted by it,
     * i.e. if a value ``n`` is found at position ``i``, ``out[n] += weight[i]``
     * instead of ``out[n] += 1``.
     *
     * @param data the 1D container with integers to count into bins
     * @param weights a 1D container with the same number of elements as ``data``
     * @param minlength The minlength
     *
     * @return 1D container with the bincount
     */
    template <class E1, class E2, XTL_REQUIRES(is_xexpression<std::decay_t<E2>>)>
    inline auto bincount(E1&& data, E2&& weights, std::size_t minlength = 0)
    {
        using result_value_type = typename std::decay_t<E2>::value_type;
        using input_value_type = typename std::decay_t<E1>::value_type;
        using size_type = typename std::decay_t<E1>::size_type;

        static_assert(std::is_integral<typename std::decay_t<E1>::value_type>::value,
                      "Bincount data has to be integral type.");
        XTENSOR_ASSERT(data.dimension() == 1);
        XTENSOR_ASSERT(weights.dimension() == 1);

        std::array<input_value_type, 2> left_right;
        left_right = xt::minmax(data)();

        if (left_right[0] < input_value_type(0))
        {
            throw std::runtime_error(
                "Data argument for bincount can only contain positive integers!");
        }

        xt::xtensor<result_value_type, 1> res = xt::zeros<result_value_type>(
            { (std::max)(minlength, std::size_t(left_right[1] + 1)) });

        for (size_type i = 0; i < data.size(); ++i)
        {
            res(data(i)) += weights(i);
        }

        return res;
    }

    template <class E1>
    inline auto bincount(E1&& data, std::size_t minlength = 0)
    {
        return bincount(std::forward<E1>(data),
                        xt::ones<typename std::decay_t<E1>::value_type>(data.shape()),
                        minlength);
    }
}

#endif
