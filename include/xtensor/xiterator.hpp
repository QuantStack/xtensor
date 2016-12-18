/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XITERATOR_HPP
#define XITERATOR_HPP

#include <utility>
#include <tuple>
#include <type_traits>
#include <iterator>
#include <array>
#include <algorithm>

#include "xutils.hpp"
#include "xexception.hpp"

namespace xt
{

    /***********************
     * broadcast functions *
     ***********************/

    template <class S1, class S2>
    bool broadcast_shape(const S1& input, S2& output);

    template <class S1, class S2>
    bool broadcastable(const S1& s1, S2& s2);

    /************
     * xstepper *
     ************/

    namespace detail
    {
        template <class C>
        struct get_storage_iterator_impl
        {
            using type = typename C::storage_iterator;
        };

        template <class C>
        struct get_storage_iterator_impl<const C>
        {
            using type = typename C::const_storage_iterator;
        };
    }

    template <class C>
    using get_storage_iterator = typename detail::get_storage_iterator_impl<C>::type;

    template <class C>
    class xstepper
    {

    public:

        using container_type = C;
        using subiterator_type = get_storage_iterator<C>;
        using subiterator_traits = std::iterator_traits<subiterator_type>;
        using value_type = typename subiterator_traits::value_type;
        using reference = typename subiterator_traits::reference;
        using pointer = typename subiterator_traits::pointer;
        using difference_type = typename subiterator_traits::difference_type;
        using size_type = typename container_type::size_type;
        using shape_type = typename container_type::shape_type;

        xstepper(container_type* c, subiterator_type it, size_type offset) noexcept;

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);

        void to_end();

        bool equal(const xstepper& rhs) const;

    private:

        container_type* p_c;
        subiterator_type m_it;
        size_type m_offset;
    };

    template <class C>
    bool operator==(const xstepper<C>& lhs,
                    const xstepper<C>& rhs);

    template <class C>
    bool operator!=(const xstepper<C>& lhs,
                    const xstepper<C>& rhs);

    template <class S, class ST>
    void increment_stepper(S& stepper,
                           ST& index,
                           const ST& shape);

    /*************
     * xiterator *
     *************/

    template <class It, class S>
    class xiterator
    {

    public:

        using self_type = xiterator<It, S>;

        using subiterator_type = It;
        using value_type = typename subiterator_type::value_type;
        using reference = typename subiterator_type::reference;
        using pointer = typename subiterator_type::pointer;
        using difference_type = typename subiterator_type::difference_type;
        using size_type = typename subiterator_type::size_type;
        using iterator_category = std::forward_iterator_tag;

        using shape_type = S;

        xiterator(It it, const shape_type& shape);
        xiterator(It it, const size_type* shape, size_type dimension);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;

        bool equal(const xiterator& rhs) const;

    private:

        subiterator_type m_it;
        shape_type m_shape;
        shape_type m_index;
    };

    template <class It, class S>
    bool operator==(const xiterator<It, S>& lhs,
                    const xiterator<It, S>& rhs);

    template <class It, class S>
    bool operator!=(const xiterator<It, S>& lhs,
                    const xiterator<It, S>& rhs);

    /**************************************
     * broadcast functions implementation *
     **************************************/

    template <class S1, class S2>
    inline bool broadcast_shape(const S1& input, S2& output)
    {
        bool trivial_broadcast = (input.size() == output.size());
        auto input_iter = input.crbegin();
        auto output_iter = output.rbegin();
        for(;input_iter != input.crend(); ++input_iter, ++output_iter)
        {
            if(*output_iter == 1)
            {
                *output_iter = *input_iter;
            }
            else if((*input_iter != 1) && (*output_iter != *input_iter))
            {
                throw broadcast_error(output, input);
            }
            trivial_broadcast = trivial_broadcast && (*output_iter == *input_iter);
        }
        return trivial_broadcast;
    }

    template <class S1, class S2>
    inline bool broadcastable(const S1& s1, const S2& s2)
    {
        auto iter1 = s1.crbegin();
        auto iter2 = s2.crbegin();
        for(;iter1 != s1.crend() && iter2 != s2.crend(); ++iter1, ++iter2)
        {
            if((*iter2 != 1) && (*iter1 != 1) && (*iter2 != *iter1))
            {
                return false;
            }
        }
        return true;
    }

    /***************************
     * xstepper implementation *
     ***************************/

    template <class C>
    inline xstepper<C>::xstepper(container_type* c, subiterator_type it, size_type offset) noexcept
        : p_c(c), m_it(it), m_offset(offset)
    {
    }

    template <class C>
    inline auto xstepper<C>::operator*() const -> reference
    {
        return *m_it;
    }

    template <class C>
    inline void xstepper<C>::step(size_type dim, size_type n)
    {
        if(dim >= m_offset)
            m_it += n * p_c->strides()[dim - m_offset];
    }

    template <class C>
    inline void xstepper<C>::step_back(size_type dim, size_type n)
    {
        if(dim >= m_offset)
            m_it -= n * p_c->strides()[dim - m_offset];
    }

    template <class C>
    inline void xstepper<C>::reset(size_type dim)
    {
        if(dim >= m_offset)
            m_it -= p_c->backstrides()[dim - m_offset];
    }

    template <class C>
    inline void xstepper<C>::to_end()
    {
        m_it = p_c->storage_end();
    }

    template <class C>
    inline bool xstepper<C>::equal(const xstepper& rhs) const
    {
        return p_c == rhs.p_c && m_it == rhs.m_it && m_offset == rhs.m_offset;
    }

    template <class C>
    inline bool operator==(const xstepper<C>& lhs,
                           const xstepper<C>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class C>
    inline bool operator!=(const xstepper<C>& lhs,
                           const xstepper<C>& rhs)
    {
        return !(lhs.equal(rhs));
    }

    template <class S, class ST>
    void increment_stepper(S& stepper,
                           ST& index,
                           const ST& shape)
    {
        using size_type = typename S::size_type;
        size_type i = index.size();
        while(i != 0)
        {
            --i;
            if(++index[i] != shape[i])
            {
                stepper.step(i);
                return;
            }
            else if(i != 0)
            {
                index[i] = 0;
                stepper.reset(i);
            }
        }
        if(i == 0)
        {
            stepper.to_end();
        }
    }

    /****************************
     * xiterator implementation *
     ****************************/

    template <class It, class S>
    inline xiterator<It, S>::xiterator(It it, const shape_type& shape)
        : m_it(it), m_shape(shape),
          m_index(make_shape<shape_type>(shape.size(), size_type(0)))
    {
    }

    template <class It, class S>
    inline xiterator<It, S>::xiterator(It it, const size_type* shape, size_type dimension)
        : m_it(it), m_shape(shape, shape + dimension),
          m_index(make_shape<shape_type>(dimension, size_type(0)))
    {
    }

    template <class It, class S>
    inline auto xiterator<It, S>::operator++() -> self_type&
    {
        increment_stepper(m_it, m_index, m_shape);
        return *this;
    }

    template <class It, class S>
    inline auto xiterator<It, S>::operator++(int) -> self_type
    {
        self_type tmp(*this);
        ++(*this);
        return tmp;
    }

    template <class It, class S>
    inline auto xiterator<It, S>::operator*() const -> reference
    {
        return *m_it;
    }

    template <class It, class S>
    inline bool xiterator<It, S>::equal(const xiterator& rhs) const
    {
        return m_it == rhs.m_it && m_shape == rhs.m_shape;
    }

    template <class It, class S>
    inline bool operator==(const xiterator<It, S>& lhs,
                           const xiterator<It, S>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class It, class S>
    inline bool operator!=(const xiterator<It, S>& lhs,
                           const xiterator<It, S>& rhs)
    {
        return !(lhs.equal(rhs));
    }
}

#endif

