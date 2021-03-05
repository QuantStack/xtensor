/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_VECTOR_HPP
#define XTENSOR_VECTOR_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "xbuffer_adaptor.hpp"
#include "xcontainer.hpp"
#include "xsemantic.hpp"

namespace xt
{

    /***********************
     * xtensor declaration *
     ***********************/

    namespace extension
    {
        template <class EC, class Tag>
        struct xvector_container_base;

        template <class EC>
        struct xvector_container_base<EC, xtensor_expression_tag>
        {
            using type = xtensor_empty_base;
        };

        template <class EC, class Tag>
        using xvector_container_base_t = typename xvector_container_base<EC, Tag>::type;
    }

    template <class EC, class Tag>
    struct xcontainer_inner_types<xvector_container<EC, Tag>>
    {
        using storage_type = EC;
        using reference = inner_reference_t<storage_type>;
        using const_reference = typename storage_type::const_reference;
        using size_type = typename storage_type::size_type;
        using shape_type = std::array<typename storage_type::size_type, 1ul>;
        using strides_type = get_strides_t<shape_type>;
        using backstrides_type = get_strides_t<shape_type>;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xvector_container<EC, Tag>;
        static constexpr layout_type layout = XTENSOR_DEFAULT_LAYOUT;
    };

    template <class EC, class Tag>
    struct xiterable_inner_types<xvector_container<EC, Tag>>
        : xcontainer_iterable_types<xvector_container<EC, Tag>>
    {
    };

    /**
     * @class xvector_container
     * @brief Dense multidimensional container with tensor semantic and fixed
     * dimension.
     *
     * The xvector_container class implements a dense multidimensional container
     * with tensor semantics and fixed dimension
     *
     * @tparam EC The type of the container holding the elements.
     * @tparam N The dimension of the container.
     * @tparam L The layout_type of the tensor.
     * @tparam Tag The expression tag.
     * @sa xtensor, xcontainer, xcontainer
     */
    template <class EC, class Tag>
    class xvector_container : public xcontainer<xvector_container<EC, Tag>>,
                              public xcontainer_semantic<xvector_container<EC, Tag>>,
                              public extension::xvector_container_base_t<EC, Tag>
    {
    public:

        using self_type = xvector_container<EC, Tag>;
        using base_type = xcontainer<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using extension_base = extension::xvector_container_base_t<EC, Tag>;
        using storage_type = typename base_type::storage_type;
        using allocator_type = typename base_type::allocator_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using inner_shape_type = typename base_type::inner_shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using inner_backstrides_type = typename base_type::inner_backstrides_type;
        using inner_strides_type = typename base_type::inner_strides_type;
        using temporary_type = typename semantic_base::temporary_type;
        using expression_tag = Tag;

        xvector_container();
        xvector_container(std::initializer_list<value_type> t);
        explicit xvector_container(const shape_type& shape);
        explicit xvector_container(const shape_type& shape, const_reference value);
        explicit xvector_container(const shape_type& shape, const strides_type& strides);
        explicit xvector_container(const shape_type& shape, const strides_type& strides, const_reference value);
        explicit xvector_container(storage_type&& storage, inner_shape_type&& shape, inner_strides_type&& strides);

        template <class S = shape_type>
        static xvector_container from_shape(S&& s);

        ~xvector_container() = default;

        xvector_container(const xvector_container&) = default;
        xvector_container& operator=(const xvector_container&) = default;

        xvector_container(xvector_container&&) = default;
        xvector_container& operator=(xvector_container&&) = default;

        template <class SC>
        explicit xvector_container(xarray_container<EC, XTENSOR_DEFAULT_LAYOUT, SC, Tag>&&);
        template <class SC>
        xvector_container& operator=(xarray_container<EC, XTENSOR_DEFAULT_LAYOUT, SC, Tag>&&);

        template <class E>
        xvector_container(const xexpression<E>& e);

        template <class E>
        xvector_container& operator=(const xexpression<E>& e);

    private:

        storage_type m_storage;

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        friend class xcontainer<xvector_container<EC, Tag>>;
    };

    /*****************************************
     * xvector_container_adaptor declaration *
     *****************************************/

    namespace extension
    {
        template <class EC, class Tag>
        struct xvector_adaptor_base;

        template <class EC>
        struct xvector_adaptor_base<EC, xtensor_expression_tag>
        {
            using type = xtensor_empty_base;
        };

        template <class EC, class Tag>
        using xvector_adaptor_base_t = typename xvector_adaptor_base<EC, Tag>::type;
    }

    template <class EC, class Tag>
    struct xcontainer_inner_types<xvector_adaptor<EC, Tag>>
    {
        using storage_type = std::remove_reference_t<EC>;
        using reference = inner_reference_t<storage_type>;
        using const_reference = typename storage_type::const_reference;
        using size_type = typename storage_type::size_type;
        using shape_type = std::array<typename storage_type::size_type, 1ul>;
        using strides_type = get_strides_t<shape_type>;
        using backstrides_type = get_strides_t<shape_type>;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xvector_container<temporary_container_t<storage_type>, Tag>;
    };

    template <class EC, class Tag>
    struct xiterable_inner_types<xvector_adaptor<EC, Tag>>
        : xcontainer_iterable_types<xvector_adaptor<EC, Tag>>
    {
    };

    /**
     * @class xvector_adaptor
     * @brief Dense multidimensional container adaptor with tensor
     * semantics and fixed dimension.
     *
     * The xvector_adaptor class implements a dense multidimensional
     * container adaptor with tensor semantics and fixed dimension. It
     * is used to provide a multidimensional container semantic and a
     * tensor semantic to stl-like containers.
     *
     * @tparam EC The closure for the container type to adapt.
     * @tparam N The dimension of the adaptor.
     * @tparam L The layout_type of the adaptor.
     * @tparam Tag The expression tag.
     * @sa xcontainer, xcontainer
     */
    template <class EC, class Tag>
    class xvector_adaptor : public xcontainer<xvector_adaptor<EC, Tag>>,
                            public xcontainer_semantic<xvector_adaptor<EC, Tag>>,
                            public extension::xvector_adaptor_base_t<EC, Tag>
    {
    public:

        using container_closure_type = EC;

        using self_type = xvector_adaptor<EC, Tag>;
        using base_type = xcontainer<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using extension_base = extension::xvector_adaptor_base_t<EC, Tag>;
        using storage_type = typename base_type::storage_type;
        using allocator_type = typename base_type::allocator_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using temporary_type = typename semantic_base::temporary_type;
        using expression_tag = Tag;

        xvector_adaptor(storage_type&& storage);
        xvector_adaptor(const storage_type& storage);

        template <class D>
        xvector_adaptor(D&& storage, const shape_type& shape);

        template <class D>
        xvector_adaptor(D&& storage, const shape_type& shape, const strides_type& strides);

        ~xvector_adaptor() = default;

        xvector_adaptor(const xvector_adaptor&) = default;
        xvector_adaptor& operator=(const xvector_adaptor&);

        xvector_adaptor(xvector_adaptor&&) = default;
        xvector_adaptor& operator=(xvector_adaptor&&);
        xvector_adaptor& operator=(temporary_type&&);

        template <class E>
        xvector_adaptor& operator=(const xexpression<E>& e);

    private:

        container_closure_type m_storage;

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        friend class xcontainer<xvector_adaptor<EC, Tag>>;
    };

    /****************************
     * xvector_view declaration *
     ****************************/

    template <class EC, class Tag>
    class xvector_view;

    namespace extension
    {
        template <class EC, class Tag>
        struct xvector_view_base;

        template <class EC>
        struct xvector_view_base<EC, xtensor_expression_tag>
        {
            using type = xtensor_empty_base;
        };

        template <class EC, class Tag>
        using xvector_view_base_t = typename xvector_view_base<EC, Tag>::type;
    }

    template <class EC, class Tag>
    struct xcontainer_inner_types<xvector_view<EC, Tag>>
    {
        using storage_type = std::remove_reference_t<EC>;
        using reference = inner_reference_t<storage_type>;
        using const_reference = typename storage_type::const_reference;
        using size_type = typename storage_type::size_type;
        using shape_type = std::array<typename storage_type::size_type, 1ul>;
        using strides_type = get_strides_t<shape_type>;
        using backstrides_type = get_strides_t<shape_type>;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xvector_container<temporary_container_t<storage_type>, Tag>;
    };

    template <class EC, class Tag>
    struct xiterable_inner_types<xvector_view<EC, Tag>>
        : xcontainer_iterable_types<xvector_view<EC, Tag>>
    {
    };

    /**
     * @class xvector_view
     * @brief Dense multidimensional container adaptor with view
     * semantics and fixed dimension.
     *
     * The xvector_view class implements a dense multidimensional
     * container adaptor with viewsemantics and fixed dimension. It
     * is used to provide a multidimensional container semantic and a
     * view semantic to stl-like containers.
     *
     * @tparam EC The closure for the container type to adapt.
     * @tparam N The dimension of the view.
     * @tparam L The layout_type of the view.
     * @tparam Tag The expression tag.
     * @sa xcontainer, xcontainer
     */
    template <class EC, class Tag>
    class xvector_view : public xcontainer<xvector_view<EC, Tag>>,
                         public xview_semantic<xvector_view<EC, Tag>>,
                         public extension::xvector_view_base_t<EC, Tag>
    {
    public:

        using container_closure_type = EC;

        using self_type = xvector_view<EC, Tag>;
        using base_type = xcontainer<self_type>;
        using semantic_base = xview_semantic<self_type>;
        using extension_base = extension::xvector_adaptor_base_t<EC, Tag>;
        using storage_type = typename base_type::storage_type;
        using allocator_type = typename base_type::allocator_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using temporary_type = typename semantic_base::temporary_type;
        using expression_tag = Tag;

        xvector_view(storage_type&& storage);
        xvector_view(const storage_type& storage);

        template <class D>
        xvector_view(D&& storage, const shape_type& shape);

        template <class D>
        xvector_view(D&& storage, const shape_type& shape, const strides_type& strides);

        ~xvector_view() = default;

        xvector_view(const xvector_view&) = default;
        xvector_view& operator=(const xvector_view&);

        xvector_view(xvector_view&&) = default;
        xvector_view& operator=(xvector_view&&);

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

    private:

        container_closure_type m_storage;

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        void assign_temporary_impl(temporary_type&& tmp);

        friend class xcontainer<xvector_view<EC, Tag>>;
        friend class xview_semantic<xvector_view<EC, Tag>>;
    };

    /************************************
     * xvector_container implementation *
     ************************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Allocates an uninitialized xvector_container that holds 0 elements.
     */
    template <class EC, class Tag>
    inline xvector_container<EC, Tag>::xvector_container()
        : base_type(), m_storage(1, value_type())
    {
    }

    /**
     * Allocates an xvector_container with nested initializer lists.
     */
    template <class EC, class Tag>
    inline xvector_container<EC, Tag>::xvector_container(std::initializer_list<value_type> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t), true);
        nested_copy(m_storage.begin(), t);
    }

    /**
     * Allocates an uninitialized xvector_container with the specified shape and
     * layout_type.
     * @param shape the shape of the xvector_container
     * @param l the layout_type of the xvector_container
     */
    template <class EC, class Tag>
    inline xvector_container<EC, Tag>::xvector_container(const shape_type& shape)
        : base_type()
    {
        base_type::resize(shape);
    }

    /**
     * Allocates an xvector_container with the specified shape and layout_type. Elements
     * are initialized to the specified value.
     * @param shape the shape of the xvector_container
     * @param value the value of the elements
     * @param l the layout_type of the xvector_container
     */
    template <class EC, class Tag>
    inline xvector_container<EC, Tag>::xvector_container(const shape_type& shape, const_reference value)
        : base_type()
    {
        base_type::resize(shape);
        std::fill(m_storage.begin(), m_storage.end(), value);
    }

    /**
     * Allocates an uninitialized xvector_container with the specified shape and strides.
     * @param shape the shape of the xvector_container
     * @param strides the strides of the xvector_container
     */
    template <class EC, class Tag>
    inline xvector_container<EC, Tag>::xvector_container(const shape_type& shape, const strides_type& strides)
        : base_type()
    {
        base_type::resize(shape, strides);
    }

    /**
     * Allocates an uninitialized xvector_container with the specified shape and strides.
     * Elements are initialized to the specified value.
     * @param shape the shape of the xvector_container
     * @param strides the strides of the xvector_container
     * @param value the value of the elements
     */
    template <class EC, class Tag>
    inline xvector_container<EC, Tag>::xvector_container(const shape_type& shape, const strides_type& strides, const_reference value)
        : base_type()
    {
        base_type::resize(shape, strides);
        std::fill(m_storage.begin(), m_storage.end(), value);
    }

    /**
     * Allocates an xvector_container by moving specified data, shape and strides
     *
     * @param storage the data for the xvector_container
     * @param shape the shape of the xvector_container
     * @param strides the strides of the xvector_container
     */
    template <class EC, class Tag>
    inline xvector_container<EC, Tag>::xvector_container(storage_type&& storage, inner_shape_type&& shape, inner_strides_type&& strides)
        : base_type(std::move(shape), std::move(strides)), m_storage(std::move(storage))
    {
    }

    template <class EC, class Tag>
    template <class SC>
    inline xvector_container<EC, Tag>::xvector_container(xarray_container<EC, XTENSOR_DEFAULT_LAYOUT, SC, Tag>&& rhs)
        : base_type(xtl::forward_sequence<inner_shape_type, decltype(rhs.shape())>(rhs.shape()),
                    xtl::forward_sequence<inner_strides_type, decltype(rhs.strides())>(rhs.strides()),
                    xtl::forward_sequence<inner_backstrides_type, decltype(rhs.backstrides())>(rhs.backstrides()),
                    std::move(rhs.layout())),
          m_storage(std::move(rhs.storage()))
    {
    }

    template <class EC, class Tag>
    template <class SC>
    inline xvector_container<EC, Tag>& xvector_container<EC, Tag>::operator=(xarray_container<EC, XTENSOR_DEFAULT_LAYOUT, SC, Tag>&& rhs)
    {
        XTENSOR_ASSERT_MSG(N == rhs.dimension(), "Cannot change dimension of xtensor.");
        std::copy(rhs.shape().begin(), rhs.shape().end(), this->shape_impl().begin());
        std::copy(rhs.strides().cbegin(), rhs.strides().cend(), this->strides_impl().begin());
        std::copy(rhs.backstrides().cbegin(), rhs.backstrides().cend(), this->backstrides_impl().begin());
        this->mutable_layout() = std::move(rhs.layout());
        m_storage = std::move(std::move(rhs.storage()));
        return *this;
    }


    template <class EC, class Tag>
    template <class S>
    inline xvector_container<EC, Tag> xvector_container<EC, Tag>::from_shape(S&& s)
    {
        XTENSOR_ASSERT_MSG(s.size() == N, "Cannot change dimension of xtensor.");
        shape_type shape = xtl::forward_sequence<shape_type, S>(s);
        return self_type(shape);
    }
    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended copy constructor.
     */
    template <class EC, class Tag>
    template <class E>
    inline xvector_container<EC, Tag>::xvector_container(const xexpression<E>& e)
        : base_type()
    {
        XTENSOR_ASSERT_MSG(N == e.derived_cast().dimension(), "Cannot change dimension of xtensor.");
        // Avoids uninitialized data because of (m_shape == shape) condition
        // in resize (called by assign), which is always true when dimension() == 0.
        if (e.derived_cast().dimension() == 0)
        {
            detail::resize_data_container(m_storage, std::size_t(1));
        }
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class EC, class Tag>
    template <class E>
    inline auto xvector_container<EC, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class EC, class Tag>
    inline auto xvector_container<EC, Tag>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }

    template <class EC, class Tag>
    inline auto xvector_container<EC, Tag>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    /**********************************
     * xvector_adaptor implementation *
     **********************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xvector_adaptor of the given stl-like container.
     * @param storage the container to adapt
     */
    template <class EC, class Tag>
    inline xvector_adaptor<EC, Tag>::xvector_adaptor(storage_type&& storage)
        : base_type(), m_storage(std::move(storage))
    {
    }

    /**
     * Constructs an xvector_adaptor of the given stl-like container.
     * @param storage the container to adapt
     */
    template <class EC, class Tag>
    inline xvector_adaptor<EC, Tag>::xvector_adaptor(const storage_type& storage)
        : base_type(), m_storage(storage)
    {
    }

    /**
     * Constructs an xvector_adaptor of the given stl-like container,
     * with the specified shape and layout_type.
     * @param storage the container to adapt
     * @param shape the shape of the xvector_adaptor
     * @param l the layout_type of the xvector_adaptor
     */
    template <class EC, class Tag>
    template <class D>
    inline xvector_adaptor<EC, Tag>::xvector_adaptor(D&& storage, const shape_type& shape)
        : base_type(), m_storage(std::forward<D>(storage))
    {
        base_type::resize(shape);
    }

    /**
     * Constructs an xvector_adaptor of the given stl-like container,
     * with the specified shape and strides.
     * @param storage the container to adapt
     * @param shape the shape of the xvector_adaptor
     * @param strides the strides of the xvector_adaptor
     */
    template <class EC, class Tag>
    template <class D>
    inline xvector_adaptor<EC, Tag>::xvector_adaptor(D&& storage, const shape_type& shape, const strides_type& strides)
        : base_type(), m_storage(std::forward<D>(storage))
    {
        base_type::resize(shape, strides);
    }
    //@}

    template <class EC, class Tag>
    inline auto xvector_adaptor<EC, Tag>::operator=(const xvector_adaptor& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_storage = rhs.m_storage;
        return *this;
    }

    template <class EC, class Tag>
    inline auto xvector_adaptor<EC, Tag>::operator=(xvector_adaptor&& rhs) -> self_type&
    {
        base_type::operator=(std::move(rhs));
        m_storage = rhs.m_storage;
        return *this;
    }

    template <class EC, class Tag>
    inline auto xvector_adaptor<EC, Tag>::operator=(temporary_type&& rhs) -> self_type&
    {
        base_type::shape_impl() = std::move(const_cast<shape_type&>(rhs.shape()));
        base_type::strides_impl() = std::move(const_cast<strides_type&>(rhs.strides()));
        base_type::backstrides_impl() = std::move(const_cast<backstrides_type&>(rhs.backstrides()));
        m_storage = std::move(rhs.storage());
        return *this;
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class EC, class Tag>
    template <class E>
    inline auto xvector_adaptor<EC, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class EC, class Tag>
    inline auto xvector_adaptor<EC, Tag>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }

    template <class EC, class Tag>
    inline auto xvector_adaptor<EC, Tag>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    /*******************************
     * xvector_view implementation *
     *******************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xvector_view of the given stl-like container.
     * @param storage the container to adapt
     */
    template <class EC, class Tag>
    inline xvector_view<EC, Tag>::xvector_view(storage_type&& storage)
        : base_type(), m_storage(std::move(storage))
    {
    }

    /**
     * Constructs an xvector_view of the given stl-like container.
     * @param storage the container to adapt
     */
    template <class EC, class Tag>
    inline xvector_view<EC, Tag>::xvector_view(const storage_type& storage)
        : base_type(), m_storage(storage)
    {
    }

    /**
     * Constructs an xvector_view of the given stl-like container,
     * with the specified shape and layout_type.
     * @param storage the container to adapt
     * @param shape the shape of the xvector_view
     * @param l the layout_type of the xvector_view
     */
    template <class EC, class Tag>
    template <class D>
    inline xvector_view<EC, Tag>::xvector_view(D&& storage, const shape_type& shape)
        : base_type(), m_storage(std::forward<D>(storage))
    {
        base_type::resize(shape);
    }

    /**
     * Constructs an xvector_view of the given stl-like container,
     * with the specified shape and strides.
     * @param storage the container to adapt
     * @param shape the shape of the xvector_view
     * @param strides the strides of the xvector_view
     */
    template <class EC, class Tag>
    template <class D>
    inline xvector_view<EC, Tag>::xvector_view(D&& storage, const shape_type& shape, const strides_type& strides)
        : base_type(), m_storage(std::forward<D>(storage))
    {
        base_type::resize(shape, strides);
    }
    //@}

    template <class EC, class Tag>
    inline auto xvector_view<EC, Tag>::operator=(const xvector_view& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_storage = rhs.m_storage;
        return *this;
    }

    template <class EC, class Tag>
    inline auto xvector_view<EC, Tag>::operator=(xvector_view&& rhs) -> self_type&
    {
        base_type::operator=(std::move(rhs));
        m_storage = rhs.m_storage;
        return *this;
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class EC, class Tag>
    template <class E>
    inline auto xvector_view<EC, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class EC, class Tag>
    template <class E>
    inline auto xvector_view<EC, Tag>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(m_storage.begin(), m_storage.end(), e);
        return *this;
    }

    template <class EC, class Tag>
    inline auto xvector_view<EC, Tag>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }

    template <class EC, class Tag>
    inline auto xvector_view<EC, Tag>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    template <class EC, class Tag>
    inline void xvector_view<EC, Tag>::assign_temporary_impl(temporary_type&& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), m_storage.begin());
    }

}

#endif
