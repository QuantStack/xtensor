/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_BUFFER_ADAPTOR_HPP
#define XTENSOR_BUFFER_ADAPTOR_HPP

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <stdexcept>

#include <xtl/xclosure.hpp>

#include "xstorage.hpp"
#include "xtensor_config.hpp"

namespace xt
{

    struct no_ownership
    {
    };

    using smart_ownership = no_ownership;

    struct acquire_ownership
    {
    };

    template <class CP, class O = no_ownership, class A = std::allocator<std::remove_pointer_t<std::remove_reference_t<CP>>>>
    class xbuffer_adaptor;

    /********************
     * buffer_storage_t *
     ********************/

    namespace detail
    {
        template <class CP, class A>
        class xbuffer_storage
        {
        public:

            using self_type = xbuffer_storage<CP, A>;
            using allocator_type = A;
            using destructor_type = allocator_type;
            using value_type = typename allocator_type::value_type;
            using reference = std::conditional_t<std::is_const<std::remove_pointer_t<std::remove_reference_t<CP>>>::value,
                                  typename allocator_type::const_reference,
                                  typename allocator_type::reference>;
            using const_reference = typename allocator_type::const_reference;
            using pointer = std::conditional_t<std::is_const<std::remove_pointer_t<std::remove_reference_t<CP>>>::value,
                                  typename allocator_type::const_pointer,
                                  typename allocator_type::pointer>;
            using const_pointer = typename allocator_type::const_pointer;
            using size_type = typename allocator_type::size_type;
            using difference_type = typename allocator_type::difference_type;

            xbuffer_storage();

            template <class P>
            xbuffer_storage(P&& data, size_type size, const allocator_type& alloc = allocator_type());

            size_type size() const noexcept;
            void resize(size_type size);

            pointer data() noexcept;
            const_pointer data() const noexcept;

            void swap(self_type& rhs) noexcept;

        private:

            pointer p_data;
            size_type m_size;
        };

        template <class CP, class D>
        class xbuffer_smart_pointer
        {
        public:

            using self_type = xbuffer_storage<CP, D>;
            using destructor_type = D;
            using value_type = std::remove_const_t<std::remove_pointer_t<std::remove_reference_t<CP>>>;
            using allocator_type = std::allocator<value_type>;
            using reference = std::conditional_t<std::is_const<std::remove_pointer_t<std::remove_reference_t<CP>>>::value,
                                  typename allocator_type::const_reference,
                                  typename allocator_type::reference>;
            using const_reference = typename allocator_type::const_reference;
            using pointer = std::conditional_t<std::is_const<std::remove_pointer_t<std::remove_reference_t<CP>>>::value,
                                  typename allocator_type::const_pointer,
                                  typename allocator_type::pointer>;
            using const_pointer = typename allocator_type::const_pointer;
            using size_type = typename allocator_type::size_type;
            using difference_type = typename allocator_type::difference_type;

            xbuffer_smart_pointer();

            template <class P, class DT>
            xbuffer_smart_pointer(P&& data_ptr, size_type size, DT&& destruct);

            size_type size() const noexcept;
            void resize(size_type size);

            pointer data() noexcept;
            const_pointer data() const noexcept;

            void swap(self_type& rhs) noexcept;

        private:

            pointer p_data;
            size_type m_size;
            destructor_type m_destruct;
        };

        template <class CP, class A>
        class xbuffer_owner_storage
        {
        public:

            using self_type = xbuffer_owner_storage<CP, A>;
            using allocator_type = A;
            using destructor_type = allocator_type;
            using value_type = typename allocator_type::value_type;
            using reference = std::conditional_t<std::is_const<std::remove_pointer_t<std::remove_reference_t<CP>>>::value,
                                  typename allocator_type::const_reference,
                                  typename allocator_type::reference>;
            using const_reference = typename allocator_type::const_reference;
            using pointer = std::conditional_t<std::is_const<std::remove_pointer_t<std::remove_reference_t<CP>>>::value,
                                  typename allocator_type::const_pointer,
                                  typename allocator_type::pointer>;
            using const_pointer = typename allocator_type::const_pointer;
            using size_type = typename allocator_type::size_type;
            using difference_type = typename allocator_type::difference_type;

            xbuffer_owner_storage() = default;

            template <class P>
            xbuffer_owner_storage(P&& data, size_type size, const allocator_type& alloc = allocator_type());

            ~xbuffer_owner_storage();

            xbuffer_owner_storage(const self_type&) = delete;
            self_type& operator=(const self_type&);

            xbuffer_owner_storage(self_type&&);
            self_type& operator=(self_type&&);

            size_type size() const noexcept;
            void resize(size_type size);

            pointer data() noexcept;
            const_pointer data() const noexcept;

            allocator_type get_allocator() const noexcept;

            void swap(self_type& rhs) noexcept;

        private:

            xtl::xclosure_wrapper<CP> m_data;
            size_type m_size;
            bool m_moved_from;
            allocator_type m_allocator;
        };

        // Workaround for MSVC2015: using void_t results in some
        // template instantiation caching that leads to wrong 
        // type deduction later in xfunction.
        template <class T>
        struct msvc2015_void
        {
            using type = void;
        };

        template <class T>
        using msvc2015_void_t = typename msvc2015_void<T>::type;

        template <class E, class = void>
        struct is_lambda_type : std::false_type
        {
        };

        // check if operator() is available
        template <class E>
        struct is_lambda_type<E, msvc2015_void_t<decltype(&E::operator())>>
            : std::true_type
        {
        };

        template <class T>
        struct self_type
        {
            using type = T;
        };
        template <class CP, class A, class O>
        struct get_buffer_storage
        {
            using type = xtl::mpl::eval_if_t<is_lambda_type<A>,
                                             self_type<xbuffer_smart_pointer<CP, A>>,
                                             self_type<xbuffer_storage<CP, A>>>;
        };

        template <class CP, class A>
        struct get_buffer_storage<CP, A, acquire_ownership>
        {
            using type = xbuffer_owner_storage<CP, A>;
        };

        template <class CP, class T>
        struct get_buffer_storage<CP, std::shared_ptr<T>, no_ownership>
        {
            using type = xbuffer_smart_pointer<CP, std::shared_ptr<T>>;
        };

        template <class CP, class T>
        struct get_buffer_storage<CP, std::unique_ptr<T>, no_ownership>
        {
            using type = xbuffer_smart_pointer<CP, std::unique_ptr<T>>;
        };

        template <class CP, class A, class O>
        using buffer_storage_t = typename get_buffer_storage<CP, A, O>::type;
    }

    /************************
     * xbuffer_adaptor_base *
     ************************/

    template <class D>
    struct buffer_inner_types;

    template <class D>
    class xbuffer_adaptor_base
    {
    public:

        using self_type = xbuffer_adaptor_base<D>;
        using derived_type = D;
        using inner_types = buffer_inner_types<D>;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename inner_types::pointer;
        using const_pointer = typename inner_types::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename inner_types::difference_type;
        using iterator = typename inner_types::iterator;
        using const_iterator = typename inner_types::const_iterator;
        using reverse_iterator = typename inner_types::reverse_iterator;
        using const_reverse_iterator = typename inner_types::const_reverse_iterator;
        using index_type = typename inner_types::index_type;

        bool empty() const noexcept;

        reference operator[](size_type i);
        const_reference operator[](size_type i) const;

        reference front();
        const_reference front() const;

        reference back();
        const_reference back() const;

        iterator begin() noexcept;
        iterator end() noexcept;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        reverse_iterator rbegin() noexcept;
        reverse_iterator rend() noexcept;

        const_reverse_iterator rbegin() const noexcept;
        const_reverse_iterator rend() const noexcept;
        const_reverse_iterator crbegin() const noexcept;
        const_reverse_iterator crend() const noexcept;

        derived_type& derived_cast() noexcept;
        const derived_type& derived_cast() const noexcept;

    protected:

        xbuffer_adaptor_base() = default;
        ~xbuffer_adaptor_base() = default;

        xbuffer_adaptor_base(const self_type&) = default;
        self_type& operator=(const self_type&) = default;

        xbuffer_adaptor_base(self_type&&) = default;
        self_type& operator=(self_type&&) = default;
    };

    template <class D>
    bool operator==(const xbuffer_adaptor_base<D>& lhs,
                    const xbuffer_adaptor_base<D>& rhs);

    template <class D>
    bool operator!=(const xbuffer_adaptor_base<D>& lhs,
                    const xbuffer_adaptor_base<D>& rhs);

    template <class D>
    bool operator<(const xbuffer_adaptor_base<D>& lhs,
                   const xbuffer_adaptor_base<D>& rhs);

    template <class D>
    bool operator<=(const xbuffer_adaptor_base<D>& lhs,
                    const xbuffer_adaptor_base<D>& rhs);

    template <class D>
    bool operator>(const xbuffer_adaptor_base<D>& lhs,
                   const xbuffer_adaptor_base<D>& rhs);

    template <class D>
    bool operator>=(const xbuffer_adaptor_base<D>& lhs,
                    const xbuffer_adaptor_base<D>& rhs);

    /*******************
     * xbuffer_adaptor *
     *******************/

    template <class CP, class O, class A>
    struct buffer_inner_types<xbuffer_adaptor<CP, O, A>>
    {
        using base_type = detail::buffer_storage_t<CP, A, O>;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using size_type = typename base_type::size_type;
        using difference_type = typename base_type::difference_type;
        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        using index_type = size_type;
    };

    template <class CP, class O, class A>
    class xbuffer_adaptor : private detail::buffer_storage_t<CP, A, O>,
                            public xbuffer_adaptor_base<xbuffer_adaptor<CP, O, A>>
    {
    public:

        using self_type = xbuffer_adaptor<CP, O, A>;
        using base_type = detail::buffer_storage_t<CP, A, O>;
        using buffer_base_type = xbuffer_adaptor_base<self_type>;
        using allocator_type = typename base_type::allocator_type;
        using destructor_type = typename base_type::destructor_type;
        using value_type = typename buffer_base_type::value_type;
        using reference = typename buffer_base_type::reference;
        using const_reference = typename buffer_base_type::const_reference;
        using pointer = typename buffer_base_type::pointer;
        using const_pointer = typename buffer_base_type::const_pointer;
        using size_type = typename buffer_base_type::size_type;
        using difference_type = typename buffer_base_type::difference_type;
        using iterator = typename buffer_base_type::iterator;
        using const_iterator = typename buffer_base_type::const_iterator;
        using reverse_iterator = typename buffer_base_type::reverse_iterator;
        using const_reverse_iterator = typename buffer_base_type::const_reverse_iterator;
        using temporary_type = uvector<value_type, allocator_type>;

        xbuffer_adaptor() = default;

        using base_type::base_type;

        ~xbuffer_adaptor() = default;

        xbuffer_adaptor(const self_type&) = default;
        self_type& operator=(const self_type&) = default;

        xbuffer_adaptor(self_type&&) = default;
        xbuffer_adaptor& operator=(self_type&&) = default;

        self_type& operator=(temporary_type&&);

        using base_type::size;
        using base_type::resize;
        using base_type::data;
        using base_type::swap;
    };

    template <class CP, class O, class A>
    void swap(xbuffer_adaptor<CP, O, A>& lhs,
              xbuffer_adaptor<CP, O, A>& rhs) noexcept;

    /*********************
     * xiterator_adaptor *
     *********************/

    template <class I, class CI>
    class xiterator_adaptor;

    template <class I, class CI>
    struct buffer_inner_types<xiterator_adaptor<I, CI>>
    {
        using traits = std::iterator_traits<I>;
        using const_traits = std::iterator_traits<CI>;

        using value_type = std::common_type_t<typename traits::value_type,
                                              typename const_traits::value_type>;
        using reference = typename traits::reference;
        using const_reference = typename const_traits::reference;
        using pointer = typename traits::pointer;
        using const_pointer = typename const_traits::pointer;
        using difference_type = std::common_type_t<typename traits::difference_type,
                                                   typename const_traits::difference_type>;
        using size_type = std::make_unsigned_t<difference_type>;
        
        using iterator = I;
        using const_iterator = CI;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        using index_type = difference_type;
    };

    template <class I, class CI>
    class xiterator_adaptor : public xbuffer_adaptor_base<xiterator_adaptor<I, CI>>
    {
    public:

        using self_type = xiterator_adaptor<I, CI>;
        using base_type = xbuffer_adaptor_base<self_type>;
        using value_type = typename base_type::value_type;
        using allocator_type = std::allocator<value_type>;
        using size_type = typename base_type::size_type;
        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator; 
        using temporary_type = uvector<value_type, allocator_type>;

        xiterator_adaptor() = default;
        xiterator_adaptor(I it, CI cit, size_type size);

        ~xiterator_adaptor() = default;

        xiterator_adaptor(const self_type&) = default;
        xiterator_adaptor& operator=(const self_type&) = default;

        xiterator_adaptor(self_type&&) = default;
        xiterator_adaptor& operator=(self_type&&) = default;

        xiterator_adaptor& operator=(const temporary_type& rhs);
        xiterator_adaptor& operator=(temporary_type&& rhs);

        size_type size() const noexcept;
        void resize(size_type size);
        
        iterator data() noexcept;
        const_iterator data() const noexcept;

        void swap(self_type& rhs) noexcept;
    
    private:

        I m_it;
        CI m_cit;
        size_type m_size;
    };

    template <class I, class CI>
    void swap(xiterator_adaptor<I, CI>& lhs,
              xiterator_adaptor<I, CI>& rhs) noexcept;

    /************************************
     * temporary_container metafunction *
     ************************************/

    template <class C>
    struct temporary_container
    {
        using type = C;
    };

    template <class CP, class O, class A>
    struct temporary_container<xbuffer_adaptor<CP, O, A>>
    {
        using type = typename xbuffer_adaptor<CP, O, A>::temporary_type;
    };

    template <class I, class CI>
    struct temporary_container<xiterator_adaptor<I, CI>>
    {
        using type = typename xiterator_adaptor<I, CI>::temporary_type;
    };

    template <class C>
    using temporary_container_t = typename temporary_container<C>::type;

    /**********************************
     * xbuffer_storage implementation *
     **********************************/

    namespace detail
    {
        template <class CP, class A>
        inline xbuffer_storage<CP, A>::xbuffer_storage()
            : p_data(nullptr), m_size(0)
        {
        }

        template <class CP, class A>
        template <class P>
        inline xbuffer_storage<CP, A>::xbuffer_storage(P&& data, size_type size, const allocator_type&)
            : p_data(std::forward<P>(data)), m_size(size)
        {
        }

        template <class CP, class A>
        inline auto xbuffer_storage<CP, A>::size() const noexcept -> size_type
        {
            return m_size;
        }

        template <class CP, class A>
        inline void xbuffer_storage<CP, A>::resize(size_type size)
        {
            if (size != m_size)
            {
                throw std::runtime_error("xbuffer_storage not resizable");
            }
        }

        template <class CP, class A>
        inline auto xbuffer_storage<CP, A>::data() noexcept -> pointer
        {
            return p_data;
        }

        template <class CP, class A>
        inline auto xbuffer_storage<CP, A>::data() const noexcept -> const_pointer
        {
            return p_data;
        }

        template <class CP, class A>
        inline void xbuffer_storage<CP, A>::swap(self_type& rhs) noexcept
        {
            using std::swap;
            swap(p_data, rhs.p_data);
            swap(m_size, rhs.m_size);
        }
    }

    /****************************************
     * xbuffer_owner_storage implementation *
     ****************************************/

    namespace detail
    {
        template <class CP, class A>
        template <class P>
        inline xbuffer_owner_storage<CP, A>::xbuffer_owner_storage(P&& data, size_type size, const allocator_type& alloc)
            : m_data(std::forward<P>(data)), m_size(size), m_moved_from(false), m_allocator(alloc)
        {
        }

        template <class CP, class A>
        inline xbuffer_owner_storage<CP, A>::~xbuffer_owner_storage()
        {
            if (!m_moved_from)
            {
                safe_destroy_deallocate(m_allocator, m_data.get(), m_size);
                m_size = 0;
            }
        }

        template <class CP, class A>
        inline auto xbuffer_owner_storage<CP, A>::operator=(const self_type& rhs) -> self_type&
        {
            using std::swap;
            if (this != &rhs)
            {
                allocator_type al = std::allocator_traits<allocator_type>::select_on_container_copy_construction(rhs.get_allocator());
                pointer tmp = safe_init_allocate(al, rhs.m_size);
                if (xtrivially_default_constructible<value_type>::value)
                {
                    std::uninitialized_copy(rhs.m_data.get(), rhs.m_data.get() + rhs.m_size, tmp);
                }
                else
                {
                    std::copy(rhs.m_data.get(), rhs.m_data.get() + rhs.m_size, tmp);
                }
                swap(m_data.get(), tmp);
                m_size = rhs.m_size;
                swap(m_allocator, al);
                safe_destroy_deallocate(al, tmp, m_size);
            }
            return *this;
        }

        template <class CP, class A>
        inline xbuffer_owner_storage<CP, A>::xbuffer_owner_storage(self_type&& rhs)
            : m_data(std::move(rhs.m_data)), m_size(std::move(rhs.m_size)), m_moved_from(std::move(rhs.m_moved_from)), m_allocator(std::move(rhs.m_allocator))
        {
            rhs.m_moved_from = true;
            rhs.m_size = 0;
        }

        template <class CP, class A>
        inline auto xbuffer_owner_storage<CP, A>::operator=(self_type&& rhs) -> self_type&
        {
            swap(rhs);
            return *this;
        }

        template <class CP, class A>
        inline auto xbuffer_owner_storage<CP, A>::size() const noexcept -> size_type
        {
            return m_size;
        }

        template <class CP, class A>
        void xbuffer_owner_storage<CP, A>::resize(size_type size)
        {
            using std::swap;
            if (size != m_size)
            {
                pointer tmp = safe_init_allocate(m_allocator, size);
                swap(m_data.get(), tmp);
                swap(m_size, size);
                safe_destroy_deallocate(m_allocator, tmp, size);
            }
        }

        template <class CP, class A>
        inline auto xbuffer_owner_storage<CP, A>::data() noexcept -> pointer
        {
            return m_data.get();
        }

        template <class CP, class A>
        inline auto xbuffer_owner_storage<CP, A>::data() const noexcept -> const_pointer
        {
            return m_data.get();
        }

        template <class CP, class A>
        inline auto xbuffer_owner_storage<CP, A>::get_allocator() const noexcept -> allocator_type
        {
            return allocator_type(m_allocator);
        }

        template <class CP, class A>
        inline void xbuffer_owner_storage<CP, A>::swap(self_type& rhs) noexcept
        {
            using std::swap;
            swap(m_data, rhs.m_data);
            swap(m_size, rhs.m_size);
            swap(m_allocator, rhs.m_allocator);
        }
    }

    /****************************************
     * xbuffer_smart_pointer implementation *
     ****************************************/

    namespace detail
    {
        template <class CP, class D>
        template <class P, class DT>
        xbuffer_smart_pointer<CP, D>::xbuffer_smart_pointer(P&& data_ptr, size_type size, DT&& destruct)
            : p_data(data_ptr), m_size(size), m_destruct(std::forward<DT>(destruct))
        {
        }

        template <class CP, class D>
        auto xbuffer_smart_pointer<CP, D>::size() const noexcept -> size_type
        {
            return m_size;
        }

        template <class CP, class D>
        void xbuffer_smart_pointer<CP, D>::resize(size_type size)
        {
            if (m_size != size)
            {
                throw std::runtime_error("xbuffer_storage not resizable");
            }
        }

        template <class CP, class D>
        auto xbuffer_smart_pointer<CP, D>::data() noexcept -> pointer
        {
            return p_data;
        }
        template <class CP, class D>
        auto xbuffer_smart_pointer<CP, D>::data() const noexcept -> const_pointer
        {
            return p_data;
        }

        template <class CP, class D>
        void xbuffer_smart_pointer<CP, D>::swap(self_type& rhs) noexcept
        {
            using std::swap;
            swap(p_data, rhs.p_data);
            swap(m_size, rhs.m_size);
            swap(m_destruct, rhs.m_destruct);
        }
    }

    /***************************************
     * xbuffer_adaptor_base implementation *
     ***************************************/

    template <class D>
    inline bool xbuffer_adaptor_base<D>::empty() const noexcept
    {
        return derived_cast().size() == size_type(0);
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::operator[](size_type i) -> reference
    {
        return derived_cast().data()[static_cast<index_type>(i)];
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::operator[](size_type i) const -> const_reference
    {
        return derived_cast().data()[static_cast<index_type>(i)];
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::front() -> reference
    {
        return this->operator[](0);
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::front() const -> const_reference
    {
        return this->operator[](0);
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::back() -> reference
    {
        return this->operator[](derived_cast().size() - 1);
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::back() const -> const_reference
    {
        return this->operator[](derived_cast().size() - 1);
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::begin() noexcept -> iterator
    {
        return derived_cast().data();
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::end() noexcept-> iterator
    {
        return derived_cast().data() + static_cast<index_type>(derived_cast().size());
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::begin() const noexcept -> const_iterator
    {
        return derived_cast().data();
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::end() const noexcept -> const_iterator
    {
        return derived_cast().data() + static_cast<index_type>(derived_cast().size());
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::cbegin() const noexcept -> const_iterator
    {
        return begin();
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::cend() const noexcept -> const_iterator
    {
        return end();
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::rbegin() noexcept-> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::rend() noexcept -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::rbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(end());
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::rend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(begin());
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::crbegin() const noexcept -> const_reverse_iterator
    {
        return rbegin();
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::crend() const noexcept -> const_reverse_iterator
    {
        return rend();
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::derived_cast() noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xbuffer_adaptor_base<D>::derived_cast() const noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D>
    inline bool operator==(const xbuffer_adaptor_base<D>& lhs,
                           const xbuffer_adaptor_base<D>& rhs)
    {
        return lhs.derived_cast().size() == rhs.derived_cast().size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class D>
    inline bool operator!=(const xbuffer_adaptor_base<D>& lhs,
                           const xbuffer_adaptor_base<D>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class D>
    inline bool operator<(const xbuffer_adaptor_base<D>& lhs,
                          const xbuffer_adaptor_base<D>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                            rhs.begin(), rhs.end(),
                                            std::less<typename D::value_type>());
    }

    template <class D>
    inline bool operator<=(const xbuffer_adaptor_base<D>& lhs,
                           const xbuffer_adaptor_base<D>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                            rhs.begin(), rhs.end(),
                                            std::less_equal<typename D::value_type>());
    }

    template <class D>
    inline bool operator>(const xbuffer_adaptor_base<D>& lhs,
                          const xbuffer_adaptor_base<D>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                            rhs.begin(), rhs.end(),
                                            std::greater<typename D::value_type>());
    }

    template <class D>
    inline bool operator>=(const xbuffer_adaptor_base<D>& lhs,
                           const xbuffer_adaptor_base<D>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                            rhs.begin(), rhs.end(),
                                            std::greater_equal<typename D::value_type>());
    }

    /**********************************
     * xbuffer_adaptor implementation *
     **********************************/

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::operator=(temporary_type&& tmp) -> self_type&
    {
        base_type::resize(tmp.size());
        std::copy(tmp.cbegin(), tmp.cend(), this->begin());
        return *this;
    }

    template <class CP, class O, class A>
    inline void swap(xbuffer_adaptor<CP, O, A>& lhs,
                     xbuffer_adaptor<CP, O, A>& rhs) noexcept
    {
        lhs.swap(rhs);
    }

    /************************************
     * xiterator_adaptor implementation *
     ************************************/
    
    template <class I, class CI>
    inline xiterator_adaptor<I, CI>::xiterator_adaptor(I it, CI cit, size_type size)
        : m_it(it), m_cit(cit), m_size(size)
    {
    }

    template <class I, class CI>
    inline auto xiterator_adaptor<I, CI>::operator=(const temporary_type& rhs) -> self_type&
    {
        resize(rhs.size());
        std::copy(rhs.cbegin(), rhs.cend(), m_it);
        return *this;
    }

    template <class I, class CI>
    inline auto xiterator_adaptor<I, CI>::operator=(temporary_type&& rhs) -> self_type&
    {
        return (*this = rhs);
    }
    
    template <class I, class CI>
    inline auto xiterator_adaptor<I, CI>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class I, class CI>
    inline void xiterator_adaptor<I, CI>::resize(size_type size)
    {
        if (m_size != size)
        {
            throw std::runtime_error("xiterator_adaptor not resizable");
        }
    }

    template <class I, class CI>
    inline auto xiterator_adaptor<I, CI>::data() noexcept -> iterator
    {
        return m_it;
    }

    template <class I, class CI>
    inline auto xiterator_adaptor<I, CI>::data() const noexcept -> const_iterator
    {
        return m_cit;
    }

    template <class I, class CI>
    inline void xiterator_adaptor<I, CI>::swap(self_type& rhs) noexcept
    {
        using std::swap;
        swap(m_it, rhs.m_it);
        swap(m_cit, rhs.m_cit);
        swap(m_size, rhs.m_size);
    }
    
    template <class I, class CI>
    inline void swap(xiterator_adaptor<I, CI>& lhs,
                     xiterator_adaptor<I, CI>& rhs) noexcept
    {
        lhs.swap(rhs);
    }
}

#endif
