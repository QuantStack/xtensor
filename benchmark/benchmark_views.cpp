/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <cstddef>
#include <chrono>
#include <string>

#include <benchmark/benchmark.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

namespace xt
{
    // Thanks to Ullrich Koethe for these benchmarks
    // https://github.com/QuantStack/xtensor/issues/695
    namespace view_benchmarks
    {
        constexpr int SIZE = 1000;

        template <class V>
        void dynamic_iterator(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::strided_view(data, xt::slice_vector{xt::all(), SIZE/2});
            for (auto _ : state)
            {
                std::copy(v.begin(), v.end(), res.begin());
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void iterator(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::view(data, xt::all(), SIZE/2);
            for (auto _ : state)
            {
                std::copy(v.begin(), v.end(), res.begin());
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void loop(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::strided_view(data, xt::slice_vector{xt::all(), SIZE/2});
            for (auto _ : state)
            {
                for(std::size_t k = 0; k < v.shape()[0]; ++k)
                {
                    res(k) = v(k);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void loop_view(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::view(data, xt::all(), SIZE / 2);
            for (auto _ : state)
            {
                for(std::size_t k = 0; k < v.shape()[0]; ++k)
                {
                    res(k) = v(k);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void loop_raw(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            for (auto _ : state)
            {
                std::size_t j = SIZE / 2;
                for(std::size_t k = 0; k < SIZE; ++k)
                {
                    res(k) = data(k, j);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void assign(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::strided_view(data, xt::slice_vector{xt::all(), SIZE/2});
            for (auto _ : state)
            {
                xt::noalias(res) = v;
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void assign_view(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::view(data, xt::all(), SIZE/2);
            auto r = xt::view(res, xt::all());
            for (auto _ : state)
            {
                r = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void assign_strided_view(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::strided_view(data, xt::slice_vector{xt::all(), SIZE/2});
            auto r = xt::strided_view(res, xt::slice_vector{xt::all()});

            for (auto _ : state)
            {
                r = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void assign_view_noalias(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::view(data, xt::all(), SIZE/2);
            auto r = xt::view(res, xt::all());
            for (auto _ : state)
            {
                xt::noalias(r) = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void assign_strided_view_noalias(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::strided_view(data, xt::slice_vector{xt::all(), SIZE/2});
            auto r = xt::strided_view(res, xt::slice_vector{xt::all()});

            for (auto _ : state)
            {
                xt::noalias(r) = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        BENCHMARK_TEMPLATE(dynamic_iterator, float);
        BENCHMARK_TEMPLATE(iterator, float);
        BENCHMARK_TEMPLATE(loop, float);
        BENCHMARK_TEMPLATE(loop_view, float);
        BENCHMARK_TEMPLATE(loop_raw, float);
        BENCHMARK_TEMPLATE(assign, float);
        BENCHMARK_TEMPLATE(assign_view, float);
        BENCHMARK_TEMPLATE(assign_strided_view, float);
        BENCHMARK_TEMPLATE(assign_view_noalias, float);
        BENCHMARK_TEMPLATE(assign_strided_view_noalias, float);
    }


    namespace stridedview
    {

        template <layout_type L1, layout_type L2>
        inline auto transpose_transpose(benchmark::State& state, std::vector<std::size_t> shape)
        {
            xarray<double, L1> x = xt::arange<double>(compute_size(shape));
            x.resize(shape);

            xarray<double, L2> res;
            res.resize(std::vector<std::size_t>(shape.rbegin(), shape.rend()));

            while (state.KeepRunning())
            {
                res = transpose(x);
            }
        }

        auto transpose_transpose_rm_rm = transpose_transpose<layout_type::row_major, layout_type::row_major>;
        auto transpose_transpose_cm_cm = transpose_transpose<layout_type::column_major, layout_type::column_major>;
        auto transpose_transpose_rm_cm = transpose_transpose<layout_type::row_major, layout_type::column_major>;
        auto transpose_transpose_cm_rm = transpose_transpose<layout_type::column_major, layout_type::row_major>;

        BENCHMARK_CAPTURE(transpose_transpose_rm_rm, 10x20x500, {10, 20, 500});
        BENCHMARK_CAPTURE(transpose_transpose_cm_cm, 10x20x500, {10, 20, 500});
        BENCHMARK_CAPTURE(transpose_transpose_rm_cm, 10x20x500, {10, 20, 500});
        BENCHMARK_CAPTURE(transpose_transpose_cm_rm, 10x20x500, {10, 20, 500});
    }
}
