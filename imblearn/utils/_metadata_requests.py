"""
This is a copy of sklearn/utils/_metadata_requests.py. It can be removed once
we support scikit-learn >= 1.4.

Metadata Routing Utility

In order to better understand the components implemented in this file, one
needs to understand their relationship to one another.

The only relevant public API for end users are the ``set_{method}_request``,
e.g. ``estimator.set_fit_request(sample_weight=True)``. However, third-party
developers and users who implement custom meta-estimators, need to deal with
the objects implemented in this file.

All estimators (should) implement a ``get_metadata_routing`` method, returning
the routing requests set for the estimator. This method is automatically
implemented via ``BaseEstimator`` for all simple estimators, but needs a custom
implementation for meta-estimators.

In non-routing consumers, i.e. the simplest case, e.g. ``SVM``,
``get_metadata_routing`` returns a ``MetadataRequest`` object.

In routers, e.g. meta-estimators and a multi metric scorer,
``get_metadata_routing`` returns a ``MetadataRouter`` object.

An object which is both a router and a consumer, e.g. a meta-estimator which
consumes ``sample_weight`` and routes ``sample_weight`` to its sub-estimators,
routing information includes both information about the object itself (added
via ``MetadataRouter.add_self_request``), as well as the routing information
for its sub-estimators.

A ``MetadataRequest`` instance includes one ``MethodMetadataRequest`` per
method in ``METHODS``, which includes ``fit``, ``score``, etc.

Request values are added to the routing mechanism by adding them to
``MethodMetadataRequest`` instances, e.g.
``metadatarequest.fit.add(param="sample_weight", alias="my_weights")``. This is
used in ``set_{method}_request`` which are automatically generated, so users
and developers almost never need to directly call methods on a
``MethodMetadataRequest``.

The ``alias`` above in the ``add`` method has to be either a string (an alias),
or a {True (requested), False (unrequested), None (error if passed)}``. There
are some other special values such as ``UNUSED`` and ``WARN`` which are used
for purposes such as warning of removing a metadata in a child class, but not
used by the end users.

``MetadataRouter`` includes information about sub-objects' routing and how
methods are mapped together. For instance, the information about which methods
of a sub-estimator are called in which methods of the meta-estimator are all
stored here. Conceptually, this information looks like:

```
{
    "sub_estimator1": (
        mapping=[(caller="fit", callee="transform"), ...],
        router=MetadataRequest(...),  # or another MetadataRouter
    ),
    ...
}
```

To give the above representation some structure, we use the following objects:

- ``(caller, callee)`` is a namedtuple called ``MethodPair``

- The list of ``MethodPair`` stored in the ``mapping`` field is a
  ``MethodMapping`` object

- ``(mapping=..., router=...)`` is a namedtuple called ``RouterMappingPair``

The ``set_{method}_request`` methods are dynamically generated for estimators
which inherit from the ``BaseEstimator``. This is done by attaching instances
of the ``RequestMethod`` descriptor to classes, which is done in the
``_MetadataRequester`` class, and ``BaseEstimator`` inherits from this mixin.
This mixin also implements the ``get_metadata_routing``, which meta-estimators
need to override, but it works for simple consumers as is.
"""
import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from sklearn import __version__, get_config
from sklearn.utils import Bunch
from sklearn.utils.fixes import parse_version
sklearn_version = parse_version(__version__)
if parse_version(sklearn_version.base_version) < parse_version('1.4'):
    SIMPLE_METHODS = ['fit', 'partial_fit', 'predict', 'predict_proba', 'predict_log_proba', 'decision_function', 'score', 'split', 'transform', 'inverse_transform']
    COMPOSITE_METHODS = {'fit_transform': ['fit', 'transform'], 'fit_predict': ['fit', 'predict']}
    METHODS = SIMPLE_METHODS + list(COMPOSITE_METHODS.keys())

    def _routing_enabled():
        """Return whether metadata routing is enabled.

        .. versionadded:: 1.3

        Returns
        -------
        enabled : bool
            Whether metadata routing is enabled. If the config is not set, it
            defaults to False.
        """
        return get_config().get("enable_metadata_routing", False)

    def _raise_for_params(params, owner, method):
        """Raise an error if metadata routing is not enabled and params are passed.

        .. versionadded:: 1.4

        Parameters
        ----------
        params : dict
            The metadata passed to a method.

        owner : object
            The object to which the method belongs.

        method : str
            The name of the method, e.g. "fit".

        Raises
        ------
        ValueError
            If metadata routing is not enabled and params are passed.
        """
        if not _routing_enabled() and params:
            raise ValueError(
                f"Metadata routing is not enabled, but {method} of {owner} "
                f"received metadata: {list(params.keys())}. "
                "Enable it using sklearn.set_config(enable_metadata_routing=True)."
            )

    def _raise_for_unsupported_routing(obj, method, **kwargs):
        """Raise when metadata routing is enabled and metadata is passed.

        This is used in meta-estimators which have not implemented metadata routing
        to prevent silent bugs. There is no need to use this function if the
        meta-estimator is not accepting any metadata, especially in `fit`, since
        if a meta-estimator accepts any metadata, they would do that in `fit` as
        well.

        Parameters
        ----------
        obj : estimator
            The estimator for which we're raising the error.

        method : str
            The method where the error is raised.

        **kwargs : dict
            The metadata passed to the method.
        """
        if _routing_enabled() and kwargs:
            raise ValueError(
                f"Metadata routing is enabled, but {method} of {obj.__class__.__name__} "
                f"received metadata: {list(kwargs.keys())}. "
                f"This {obj.__class__.__name__} does not support metadata routing."
            )

    class _RoutingNotSupportedMixin:
        """A mixin to be used to remove the default `get_metadata_routing`.

        This is used in meta-estimators where metadata routing is not yet
        implemented.

        This also makes it clear in our rendered documentation that this method
        cannot be used.
        """

        def get_metadata_routing(self):
            """Raise `NotImplementedError`.

            This estimator does not support metadata routing yet."""
            pass
    UNUSED = '$UNUSED$'
    WARN = '$WARN$'
    UNCHANGED = '$UNCHANGED$'
    VALID_REQUEST_VALUES = [False, True, None, UNUSED, WARN]

    def request_is_alias(item):
        """Check if an item is a valid alias.

        Values in ``VALID_REQUEST_VALUES`` are not considered aliases in this
        context. Only a string which is a valid identifier is.

        Parameters
        ----------
        item : object
            The given item to be checked if it can be an alias.

        Returns
        -------
        result : bool
            Whether the given item is a valid alias.
        """
        pass

    def request_is_valid(item):
        """Check if an item is a valid request value (and not an alias).

        Parameters
        ----------
        item : object
            The given item to be checked.

        Returns
        -------
        result : bool
            Whether the given item is valid.
        """
        pass

    class MethodMetadataRequest:
        """A prescription of how metadata is to be passed to a single method.

        Refer to :class:`MetadataRequest` for how this class is used.

        .. versionadded:: 1.3

        Parameters
        ----------
        owner : str
            A display name for the object owning these requests.

        method : str
            The name of the method to which these requests belong.

        requests : dict of {str: bool, None or str}, default=None
            The initial requests for this method.
        """

        def __init__(self, owner, method, requests=None):
            self._requests = requests or dict()
            self.owner = owner
            self.method = method

        @property
        def requests(self):
            """Dictionary of the form: ``{key: alias}``."""
            pass

        def add_request(self, *, param, alias):
            """Add request info for a metadata.

            Parameters
            ----------
            param : str
                The property for which a request is set.

            alias : str, or {True, False, None}
                Specifies which metadata should be routed to `param`

                - str: the name (or alias) of metadata given to a meta-estimator that
                should be routed to this parameter.

                - True: requested

                - False: not requested

                - None: error if passed
            """
            pass

        def _get_param_names(self, return_alias):
            """Get names of all metadata that can be consumed or routed by this method.

            This method returns the names of all metadata, even the ``False``
            ones.

            Parameters
            ----------
            return_alias : bool
                Controls whether original or aliased names should be returned. If
                ``False``, aliases are ignored and original names are returned.

            Returns
            -------
            names : set of str
                A set of strings with the names of all parameters.
            """
            pass

        def _check_warnings(self, *, params):
            """Check whether metadata is passed which is marked as WARN.

            If any metadata is passed which is marked as WARN, a warning is raised.

            Parameters
            ----------
            params : dict
                The metadata passed to a method.
            """
            pass

        def _route_params(self, params):
            """Prepare the given parameters to be passed to the method.

            The output of this method can be used directly as the input to the
            corresponding method as extra props.

            Parameters
            ----------
            params : dict
                A dictionary of provided metadata.

            Returns
            -------
            params : Bunch
                A :class:`~sklearn.utils.Bunch` of {prop: value} which can be given to
                the corresponding method.
            """
            pass

        def _consumes(self, params):
            """Check whether the given parameters are consumed by this method.

            Parameters
            ----------
            params : iterable of str
                An iterable of parameters to check.

            Returns
            -------
            consumed : set of str
                A set of parameters which are consumed by this method.
            """
            pass

        def _serialize(self):
            """Serialize the object.

            Returns
            -------
            obj : dict
                A serialized version of the instance in the form of a dictionary.
            """
            pass

        def __repr__(self):
            return str(self._serialize())

        def __str__(self):
            return str(repr(self))

    class MetadataRequest:
        """Contains the metadata request info of a consumer.

        Instances of `MethodMetadataRequest` are used in this class for each
        available method under `metadatarequest.{method}`.

        Consumer-only classes such as simple estimators return a serialized
        version of this class as the output of `get_metadata_routing()`.

        .. versionadded:: 1.3

        Parameters
        ----------
        owner : str
            The name of the object to which these requests belong.
        """
        _type = 'metadata_request'

        def __init__(self, owner):
            self.owner = owner
            for method in SIMPLE_METHODS:
                setattr(self, method, MethodMetadataRequest(owner=owner, method=method))

        def consumes(self, method, params):
            """Check whether the given parameters are consumed by the given method.

            .. versionadded:: 1.4

            Parameters
            ----------
            method : str
                The name of the method to check.

            params : iterable of str
                An iterable of parameters to check.

            Returns
            -------
            consumed : set of str
                A set of parameters which are consumed by the given method.
            """
            pass

        def __getattr__(self, name):
            if name not in COMPOSITE_METHODS:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            requests = {}
            for method in COMPOSITE_METHODS[name]:
                mmr = getattr(self, method)
                existing = set(requests.keys())
                upcoming = set(mmr.requests.keys())
                common = existing & upcoming
                conflicts = [key for key in common if requests[key] != mmr._requests[key]]
                if conflicts:
                    raise ValueError(f'Conflicting metadata requests for {', '.join(conflicts)} while composing the requests for {name}. Metadata with the same name for methods {', '.join(COMPOSITE_METHODS[name])} should have the same request value.')
                requests.update(mmr._requests)
            return MethodMetadataRequest(owner=self.owner, method=name, requests=requests)

        def _get_param_names(self, method, return_alias, ignore_self_request=None):
            """Get names of all metadata that can be consumed or routed by specified                 method.

            This method returns the names of all metadata, even the ``False``
            ones.

            Parameters
            ----------
            method : str
                The name of the method for which metadata names are requested.

            return_alias : bool
                Controls whether original or aliased names should be returned. If
                ``False``, aliases are ignored and original names are returned.

            ignore_self_request : bool
                Ignored. Present for API compatibility.

            Returns
            -------
            names : set of str
                A set of strings with the names of all parameters.
            """
            pass

        def _route_params(self, *, method, params):
            """Prepare the given parameters to be passed to the method.

            The output of this method can be used directly as the input to the
            corresponding method as extra keyword arguments to pass metadata.

            Parameters
            ----------
            method : str
                The name of the method for which the parameters are requested and
                routed.

            params : dict
                A dictionary of provided metadata.

            Returns
            -------
            params : Bunch
                A :class:`~sklearn.utils.Bunch` of {prop: value} which can be given to
                the corresponding method.
            """
            pass

        def _check_warnings(self, *, method, params):
            """Check whether metadata is passed which is marked as WARN.

            If any metadata is passed which is marked as WARN, a warning is raised.

            Parameters
            ----------
            method : str
                The name of the method for which the warnings should be checked.

            params : dict
                The metadata passed to a method.
            """
            pass

        def _serialize(self):
            """Serialize the object.

            Returns
            -------
            obj : dict
                A serialized version of the instance in the form of a dictionary.
            """
            pass

        def __repr__(self):
            return str(self._serialize())

        def __str__(self):
            return str(repr(self))
    RouterMappingPair = namedtuple('RouterMappingPair', ['mapping', 'router'])
    MethodPair = namedtuple('MethodPair', ['callee', 'caller'])

    class MethodMapping:
        """Stores the mapping between callee and caller methods for a router.

        This class is primarily used in a ``get_metadata_routing()`` of a router
        object when defining the mapping between a sub-object (a sub-estimator or a
        scorer) to the router's methods. It stores a collection of ``Route``
        namedtuples.

        Iterating through an instance of this class will yield named
        ``MethodPair(callee, caller)`` tuples.

        .. versionadded:: 1.3
        """

        def __init__(self):
            self._routes = []

        def __iter__(self):
            return iter(self._routes)

        def add(self, *, callee, caller):
            """Add a method mapping.

            Parameters
            ----------
            callee : str
                Child object's method name. This method is called in ``caller``.

            caller : str
                Parent estimator's method name in which the ``callee`` is called.

            Returns
            -------
            self : MethodMapping
                Returns self.
            """
            pass

        def _serialize(self):
            """Serialize the object.

            Returns
            -------
            obj : list
                A serialized version of the instance in the form of a list.
            """
            pass

        @classmethod
        def from_str(cls, route):
            """Construct an instance from a string.

            Parameters
            ----------
            route : str
                A string representing the mapping, it can be:

                - `"one-to-one"`: a one to one mapping for all methods.
                - `"method"`: the name of a single method, such as ``fit``,
                    ``transform``, ``score``, etc.

            Returns
            -------
            obj : MethodMapping
                A :class:`~sklearn.utils.metadata_routing.MethodMapping` instance
                constructed from the given string.
            """
            pass

        def __repr__(self):
            return str(self._serialize())

        def __str__(self):
            return str(repr(self))

    class MetadataRouter:
        """Stores and handles metadata routing for a router object.

        This class is used by router objects to store and handle metadata routing.
        Routing information is stored as a dictionary of the form ``{"object_name":
        RouteMappingPair(method_mapping, routing_info)}``, where ``method_mapping``
        is an instance of :class:`~sklearn.utils.metadata_routing.MethodMapping` and
        ``routing_info`` is either a
        :class:`~sklearn.utils.metadata_routing.MetadataRequest` or a
        :class:`~sklearn.utils.metadata_routing.MetadataRouter` instance.

        .. versionadded:: 1.3

        Parameters
        ----------
        owner : str
            The name of the object to which these requests belong.
        """
        _type = 'metadata_router'

        def __init__(self, owner):
            self._route_mappings = dict()
            self._self_request = None
            self.owner = owner

        def add_self_request(self, obj):
            """Add `self` (as a consumer) to the routing.

            This method is used if the router is also a consumer, and hence the
            router itself needs to be included in the routing. The passed object
            can be an estimator or a
            :class:`~sklearn.utils.metadata_routing.MetadataRequest`.

            A router should add itself using this method instead of `add` since it
            should be treated differently than the other objects to which metadata
            is routed by the router.

            Parameters
            ----------
            obj : object
                This is typically the router instance, i.e. `self` in a
                ``get_metadata_routing()`` implementation. It can also be a
                ``MetadataRequest`` instance.

            Returns
            -------
            self : MetadataRouter
                Returns `self`.
            """
            pass

        def add(self, *, method_mapping, **objs):
            """Add named objects with their corresponding method mapping.

            Parameters
            ----------
            method_mapping : MethodMapping or str
                The mapping between the child and the parent's methods. If str, the
                output of :func:`~sklearn.utils.metadata_routing.MethodMapping.from_str`
                is used.

            **objs : dict
                A dictionary of objects from which metadata is extracted by calling
                :func:`~sklearn.utils.metadata_routing.get_routing_for_object` on them.

            Returns
            -------
            self : MetadataRouter
                Returns `self`.
            """
            pass

        def consumes(self, method, params):
            """Check whether the given parameters are consumed by the given method.

            .. versionadded:: 1.4

            Parameters
            ----------
            method : str
                The name of the method to check.

            params : iterable of str
                An iterable of parameters to check.

            Returns
            -------
            consumed : set of str
                A set of parameters which are consumed by the given method.
            """
            pass

        def _get_param_names(self, *, method, return_alias, ignore_self_request):
            """Get names of all metadata that can be consumed or routed by specified                 method.

            This method returns the names of all metadata, even the ``False``
            ones.

            Parameters
            ----------
            method : str
                The name of the method for which metadata names are requested.

            return_alias : bool
                Controls whether original or aliased names should be returned,
                which only applies to the stored `self`. If no `self` routing
                object is stored, this parameter has no effect.

            ignore_self_request : bool
                If `self._self_request` should be ignored. This is used in
                `_route_params`. If ``True``, ``return_alias`` has no effect.

            Returns
            -------
            names : set of str
                A set of strings with the names of all parameters.
            """
            pass

        def _route_params(self, *, params, method):
            """Prepare the given parameters to be passed to the method.

            This is used when a router is used as a child object of another router.
            The parent router then passes all parameters understood by the child
            object to it and delegates their validation to the child.

            The output of this method can be used directly as the input to the
            corresponding method as extra props.

            Parameters
            ----------
            method : str
                The name of the method for which the parameters are requested and
                routed.

            params : dict
                A dictionary of provided metadata.

            Returns
            -------
            params : Bunch
                A :class:`~sklearn.utils.Bunch` of {prop: value} which can be given to
                the corresponding method.
            """
            pass

        def route_params(self, *, caller, params):
            """Return the input parameters requested by child objects.

            The output of this method is a bunch, which includes the inputs for all
            methods of each child object that are used in the router's `caller`
            method.

            If the router is also a consumer, it also checks for warnings of
            `self`'s/consumer's requested metadata.

            Parameters
            ----------
            caller : str
                The name of the method for which the parameters are requested and
                routed. If called inside the :term:`fit` method of a router, it
                would be `"fit"`.

            params : dict
                A dictionary of provided metadata.

            Returns
            -------
            params : Bunch
                A :class:`~sklearn.utils.Bunch` of the form
                ``{"object_name": {"method_name": {prop: value}}}`` which can be
                used to pass the required metadata to corresponding methods or
                corresponding child objects.
            """
            pass

        def validate_metadata(self, *, method, params):
            """Validate given metadata for a method.

            This raises a ``TypeError`` if some of the passed metadata are not
            understood by child objects.

            Parameters
            ----------
            method : str
                The name of the method for which the parameters are requested and
                routed. If called inside the :term:`fit` method of a router, it
                would be `"fit"`.

            params : dict
                A dictionary of provided metadata.
            """
            pass

        def _serialize(self):
            """Serialize the object.

            Returns
            -------
            obj : dict
                A serialized version of the instance in the form of a dictionary.
            """
            pass

        def __iter__(self):
            if self._self_request:
                yield ('$self_request', RouterMappingPair(mapping=MethodMapping.from_str('one-to-one'), router=self._self_request))
            for name, route_mapping in self._route_mappings.items():
                yield (name, route_mapping)

        def __repr__(self):
            return str(self._serialize())

        def __str__(self):
            return str(repr(self))

    def get_routing_for_object(obj=None):
        """Get a ``Metadata{Router, Request}`` instance from the given object.

        This function returns a
        :class:`~sklearn.utils.metadata_routing.MetadataRouter` or a
        :class:`~sklearn.utils.metadata_routing.MetadataRequest` from the given input.

        This function always returns a copy or an instance constructed from the
        input, such that changing the output of this function will not change the
        original object.

        .. versionadded:: 1.3

        Parameters
        ----------
        obj : object
            - If the object is already a
                :class:`~sklearn.utils.metadata_routing.MetadataRequest` or a
                :class:`~sklearn.utils.metadata_routing.MetadataRouter`, return a copy
                of that.
            - If the object provides a `get_metadata_routing` method, return a copy
                of the output of that method.
            - Returns an empty :class:`~sklearn.utils.metadata_routing.MetadataRequest`
                otherwise.

        Returns
        -------
        obj : MetadataRequest or MetadataRouting
            A ``MetadataRequest`` or a ``MetadataRouting`` taken or created from
            the given object.
        """
        pass
    REQUESTER_DOC = '        Request metadata passed to the ``{method}`` method.\n\n            Note that this method is only relevant if\n            ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).\n            Please see :ref:`User Guide <metadata_routing>` on how the routing\n            mechanism works.\n\n            The options for each parameter are:\n\n            - ``True``: metadata is requested, and     passed to ``{method}`` if provided. The request is ignored if     metadata is not provided.\n\n            - ``False``: metadata is not requested and the meta-estimator     will not pass it to ``{method}``.\n\n            - ``None``: metadata is not requested, and the meta-estimator     will raise an error if the user provides it.\n\n            - ``str``: metadata should be passed to the meta-estimator with     this given alias instead of the original name.\n\n            The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the\n            existing request. This allows you to change the request for some\n            parameters and not others.\n\n            .. versionadded:: 1.3\n\n            .. note::\n                This method is only relevant if this estimator is used as a\n                sub-estimator of a meta-estimator, e.g. used inside a\n                :class:`~sklearn.pipeline.Pipeline`. Otherwise it has no effect.\n\n            Parameters\n            ----------\n    '
    REQUESTER_DOC_PARAM = '        {metadata} : str, True, False, or None,                         default=sklearn.utils.metadata_routing.UNCHANGED\n                Metadata routing for ``{metadata}`` parameter in ``{method}``.\n\n    '
    REQUESTER_DOC_RETURN = '        Returns\n            -------\n            self : object\n                The updated object.\n    '

    class RequestMethod:
        """
        A descriptor for request methods.

        .. versionadded:: 1.3

        Parameters
        ----------
        name : str
            The name of the method for which the request function should be
            created, e.g. ``"fit"`` would create a ``set_fit_request`` function.

        keys : list of str
            A list of strings which are accepted parameters by the created
            function, e.g. ``["sample_weight"]`` if the corresponding method
            accepts it as a metadata.

        validate_keys : bool, default=True
            Whether to check if the requested parameters fit the actual parameters
            of the method.

        Notes
        -----
        This class is a descriptor [1]_ and uses PEP-362 to set the signature of
        the returned function [2]_.

        References
        ----------
        .. [1] https://docs.python.org/3/howto/descriptor.html

        .. [2] https://www.python.org/dev/peps/pep-0362/
        """

        def __init__(self, name, keys, validate_keys=True):
            self.name = name
            self.keys = keys
            self.validate_keys = validate_keys

        def __get__(self, instance, owner):

            def func(*args, **kw):
                """Updates the request for provided parameters

                This docstring is overwritten below.
                See REQUESTER_DOC for expected functionality
                """
                if not _routing_enabled():
                    raise RuntimeError('This method is only available when metadata routing is enabled. You can enable it using sklearn.set_config(enable_metadata_routing=True).')
                if self.validate_keys and set(kw) - set(self.keys):
                    raise TypeError(f'Unexpected args: {set(kw) - set(self.keys)}. Accepted arguments are: {set(self.keys)}')
                if instance is None:
                    _instance = args[0]
                    args = args[1:]
                else:
                    _instance = instance
                if args:
                    raise TypeError(f'set_{self.name}_request() takes 0 positional argument but {len(args)} were given')
                requests = _instance._get_metadata_request()
                method_metadata_request = getattr(requests, self.name)
                for prop, alias in kw.items():
                    if alias is not UNCHANGED:
                        method_metadata_request.add_request(param=prop, alias=alias)
                _instance._metadata_request = requests
                return _instance
            func.__name__ = f'set_{self.name}_request'
            params = [inspect.Parameter(name='self', kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=owner)]
            params.extend([inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=UNCHANGED, annotation=Optional[Union[bool, None, str]]) for k in self.keys])
            func.__signature__ = inspect.Signature(params, return_annotation=owner)
            doc = REQUESTER_DOC.format(method=self.name)
            for metadata in self.keys:
                doc += REQUESTER_DOC_PARAM.format(metadata=metadata, method=self.name)
            doc += REQUESTER_DOC_RETURN
            func.__doc__ = doc
            return func

    class _MetadataRequester:
        """Mixin class for adding metadata request functionality.

        ``BaseEstimator`` inherits from this Mixin.

        .. versionadded:: 1.3
        """
        if TYPE_CHECKING:

        def __init_subclass__(cls, **kwargs):
            """Set the ``set_{method}_request`` methods.

            This uses PEP-487 [1]_ to set the ``set_{method}_request`` methods. It
            looks for the information available in the set default values which are
            set using ``__metadata_request__*`` class attributes, or inferred
            from method signatures.

            The ``__metadata_request__*`` class attributes are used when a method
            does not explicitly accept a metadata through its arguments or if the
            developer would like to specify a request value for those metadata
            which are different from the default ``None``.

            References
            ----------
            .. [1] https://www.python.org/dev/peps/pep-0487
            """
            try:
                requests = cls._get_default_requests()
            except Exception:
                super().__init_subclass__(**kwargs)
                return
            for method in SIMPLE_METHODS:
                mmr = getattr(requests, method)
                if not len(mmr.requests):
                    continue
                setattr(cls, f'set_{method}_request', RequestMethod(method, sorted(mmr.requests.keys())))
            super().__init_subclass__(**kwargs)

        @classmethod
        def _build_request_for_signature(cls, router, method):
            """Build the `MethodMetadataRequest` for a method using its signature.

            This method takes all arguments from the method signature and uses
            ``None`` as their default request value, except ``X``, ``y``, ``Y``,
            ``Xt``, ``yt``, ``*args``, and ``**kwargs``.

            Parameters
            ----------
            router : MetadataRequest
                The parent object for the created `MethodMetadataRequest`.
            method : str
                The name of the method.

            Returns
            -------
            method_request : MethodMetadataRequest
                The prepared request using the method's signature.
            """
            pass

        @classmethod
        def _get_default_requests(cls):
            """Collect default request values.

            This method combines the information present in ``__metadata_request__*``
            class attributes, as well as determining request keys from method
            signatures.
            """
            pass

        def _get_metadata_request(self):
            """Get requested data properties.

            Please check :ref:`User Guide <metadata_routing>` on how the routing
            mechanism works.

            Returns
            -------
            request : MetadataRequest
                A :class:`~sklearn.utils.metadata_routing.MetadataRequest` instance.
            """
            pass

        def get_metadata_routing(self):
            """Get metadata routing of this object.

            Please check :ref:`User Guide <metadata_routing>` on how the routing
            mechanism works.

            Returns
            -------
            routing : MetadataRequest
                A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
                routing information.
            """
            pass

    def process_routing(_obj, _method, /, **kwargs):
        """Validate and route input parameters.

        This function is used inside a router's method, e.g. :term:`fit`,
        to validate the metadata and handle the routing.

        Assuming this signature: ``fit(self, X, y, sample_weight=None, **fit_params)``,
        a call to this function would be:
        ``process_routing(self, sample_weight=sample_weight, **fit_params)``.

        Note that if routing is not enabled and ``kwargs`` is empty, then it
        returns an empty routing where ``process_routing(...).ANYTHING.ANY_METHOD``
        is always an empty dictionary.

        .. versionadded:: 1.3

        Parameters
        ----------
        _obj : object
            An object implementing ``get_metadata_routing``. Typically a
            meta-estimator.

        _method : str
            The name of the router's method in which this function is called.

        **kwargs : dict
            Metadata to be routed.

        Returns
        -------
        routed_params : Bunch
            A :class:`~sklearn.utils.Bunch` of the form ``{"object_name":
            {"method_name": {prop: value}}}`` which can be used to pass the required
            metadata to corresponding methods or corresponding child objects. The object
            names are those defined in `obj.get_metadata_routing()`.
        """
        pass
else:
    from sklearn.exceptions import UnsetMetadataPassedError
    from sklearn.utils._metadata_requests import COMPOSITE_METHODS, METHODS, SIMPLE_METHODS, UNCHANGED, UNUSED, WARN, MetadataRequest, MetadataRouter, MethodMapping, _MetadataRequester, _raise_for_params, _raise_for_unsupported_routing, _routing_enabled, _RoutingNotSupportedMixin, get_routing_for_object, process_routing
