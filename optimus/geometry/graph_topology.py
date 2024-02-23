"""Graph topology of nested domains."""


class Graph:
    def __init__(self, label="graph"):
        """
        Create a graph topology for nested domains.

        Parameters
        ----------
        label : str
            The label of the graph topology.
        """

        self.label = label

        self.subdomain_nodes = []
        self.interface_nodes = []
        self.edges = []
        self.interface_connectors = []

        self._has_exterior = False

        return

    def number_subdomain_nodes(self):
        """Returns the number of active subdomain nodes in the graph."""
        count = 0
        for node in self.subdomain_nodes:
            if node.is_active():
                count += 1
        return count

    def number_interface_nodes(self):
        """Returns the number of active interface nodes in the graph."""
        count = 0
        for node in self.interface_nodes:
            if node.is_active():
                count += 1
        return count

    def number_edges(self):
        """Returns the number of active edges in the graph."""
        count = 0
        for edge in self.edges:
            if edge.is_active():
                count += 1
        return count

    def number_interface_connectors(self):
        """Returns the number of active interface connectors in the graph."""
        count = 0
        for connector in self.interface_connectors:
            if connector.is_active():
                count += 1
        return count

    def print_graph_information(self):
        """Display information about the graph topology."""

        print("Label of graph:", self.label)

        print("Number of subdomain nodes:", self.number_subdomain_nodes())
        print("Number of interface nodes:", self.number_interface_nodes())
        print("Number of edges:", self.number_edges())
        print("Number of interface connectors:", self.number_interface_connectors())

        print("\nTopology of subdomain nodes:")
        for node in self.subdomain_nodes:
            node.print_topology()

        print("\nTopology of interface nodes:")
        for node in self.interface_nodes:
            node.print_topology()

        print("\nTopology of edges:")
        for edge in self.edges:
            edge.print_topology()

        print("\nTopology of interface connectors:")
        for connector in self.interface_connectors:
            connector.print_topology()

        print("\nMaterials:")
        for node in self.subdomain_nodes:
            node.print_material()

        print("\nGeometry:")
        for node in self.interface_nodes:
            node.print_geometry()

        print("\nSources:")
        for node in self.subdomain_nodes:
            node.print_sources()

        return

    def plot_graph(self):
        """
        Plot the graph topology.

        Currently, visualisation of the graph is only possible with the
        treelib package.
        """

        try:
            import treelib
        except ImportError:
            print(
                "The graph cannot be plotted without the 'treelib' package. "
                "Please install the library manually. "
                "This issue will be solved in future releases."
            )
            return

        from optimus.geometry.common import Geometry

        subdomain_nodes_list = []
        for node in self.subdomain_nodes:
            if node.is_active():
                parent_interface = self.interface_nodes[node.parent_interface_id]
                subdomain_nodes_list.append(
                    (
                        str(node.identifier) + "." + node.material.name,
                        node.identifier,
                        parent_interface.parent_subdomain_id,
                    )
                )
        tree_subdomain = self._create_tree(subdomain_nodes_list)

        interface_nodes_list = []
        for node in self.interface_nodes:
            if node.is_active():
                node_label = str(node.identifier) + "."
                if not node.bounded:
                    node_label += "exterior"
                elif isinstance(node.geometry, Geometry):
                    node_label += node.geometry.label
                else:
                    node_label += "unknown"
                interface_nodes_list.append(
                    (node_label, node.identifier, node.parent_interface_id)
                )
        tree_interfaces = self._create_tree(interface_nodes_list)

        print("Subdomain tree:")
        tree_subdomain.show()
        print("Interface tree:")
        tree_interfaces.show()

        return

    @staticmethod
    def _create_tree(nodes_list):
        """
        Create tree with the nodes specified.

        Create a tree graph for the specified nodes. Notice that the nodes
        may come in arbitrary order while the tree graph is created in a
        recurrent way specifying the parent node. Hence, the list may need to
        be traversed multiple times.

        Parameters
        ----------
        nodes_list : list[tuple[int]]
            The list of nodes to be added to the tree graph.

        Returns
        -------
        tree : treelib.Tree
            The tree graph to be created.
        """

        from treelib import Tree
        from treelib.exceptions import NodeIDAbsentError

        tree = Tree()

        for counter in range(len(nodes_list) ** 2):
            if len(nodes_list) == 0:
                break
            else:
                active_node = nodes_list[0]
            try:
                tree.create_node(active_node[0], active_node[1], parent=active_node[2])
            except NodeIDAbsentError:
                nodes_list.append(active_node)
            finally:
                nodes_list.pop(0)

        return tree

    def create_exterior_domain(
        self,
        material,
        source=None,
        verbose=False,
    ):
        """
        Create the only exterior domain in the topology, which will be the root
        node of the tree graph.

        Parameters
        ----------
        material : optimus.material.common.Material
            The homogeneous propagating material of the subdomain.
        source : None, optimus.source.common.Source
            The source of the incident wave field inside the subdomain.
        verbose : bool
            Display information.
        """

        if self._has_exterior:
            raise AssertionError(
                "The topology already has an exterior. "
                "Only one exterior domain can be defined."
            )
        else:
            self._has_exterior = True

        new_subdomain_node_id = len(self.subdomain_nodes)
        new_interface_node_id = len(self.interface_nodes)

        self.subdomain_nodes.append(
            SubdomainNode(
                new_subdomain_node_id,
                new_interface_node_id,
                [],
                False,
                material,
                source,
            )
        )
        if verbose:
            print("Created new subdomain node " + str(new_subdomain_node_id) + ".")

        self.interface_nodes.append(
            InterfaceNode(
                new_interface_node_id,
                None,
                new_subdomain_node_id,
                None,
                [],
                [],
                False,
                None,
            )
        )
        if verbose:
            print("Created new interface node " + str(new_interface_node_id) + ".")

        return

    def create_interface(
        self,
        geometry,
        exterior_subdomain_id,
        interior_subdomain_material,
        interior_interface_ids=None,
        source=None,
        verbose=False,
    ):
        """
        Create a new interface in the topology.

        A new interface splits an existing subdomain. The identifier for the existing
        subdomain exterior to the new interface needs to be specified. The subdomain
        interior to the interface will be filled with the material type provided.
        The identifiers to all existing interfaces that will be inside the new material
        have to be specified. If not, they are assumed to be exterior to the new
        interface.
        Optionally, a new source inside the new material can be specified.

        Parameters
        ----------
        geometry : None, optimus.geometry.common.Geometry
            The geometry of the interface, which includes the surface mesh.
        exterior_subdomain_id : int
            The identifier of the subdomain exterior to the new interface.
        interior_subdomain_material : optimus.material.common.Material
            The homogeneous propagating material of the subdomain interior
            to the new interface.
        interior_interface_ids : None, list[int]
            The identifiers of the interface nodes interior to the new interface.
        source : None, optimus.source.common.Source
            The source of the incident wave field inside the subdomain interior
            to the new interface.
        verbose : bool
            Display information.
        """

        from copy import deepcopy as _deepcopy

        # Split the subdomain based on the new interface.

        exterior_subdomain = self._check_subdomain(exterior_subdomain_id)
        self._check_new_geometry(geometry)

        if interior_interface_ids is None:
            interior_interface_ids = []
        new_child_ids, new_sibling_ids = self._split_subdomain(
            exterior_subdomain,
            interior_interface_ids,
        )

        # Create the new interface and subdomain nodes.

        new_interface_node_id = len(self.interface_nodes)
        new_subdomain_node_id = len(self.subdomain_nodes)

        self.subdomain_nodes.append(
            SubdomainNode(
                new_subdomain_node_id,
                new_interface_node_id,
                interior_interface_ids,
                True,
                interior_subdomain_material,
                source,
            )
        )
        if verbose:
            print("Created new subdomain node " + str(new_subdomain_node_id) + ".")

        self.interface_nodes.append(
            InterfaceNode(
                new_interface_node_id,
                exterior_subdomain_id,
                new_subdomain_node_id,
                exterior_subdomain.parent_interface_id,
                new_sibling_ids,
                new_child_ids,
                True,
                geometry,
            )
        )
        if verbose:
            print("Created new interface node " + str(new_interface_node_id) + ".")

        # Update the topology of the exterior subdomain

        exterior_subdomain.update_topology(
            None,
            new_sibling_ids + [new_interface_node_id],
        )
        if verbose:
            print(
                "Updated topology of exterior subdomain node "
                + str(exterior_subdomain.identifier)
                + "."
            )

        # Update the topology of the interfaces belonging to the old boundary
        # of the exterior subdomain

        exterior_interface_id = exterior_subdomain.parent_interface_id
        self.interface_nodes[exterior_interface_id].update_topology(
            None,
            None,
            None,
            None,
            new_sibling_ids + [new_interface_node_id],
        )
        if verbose:
            print(
                "Updated topology of exterior interface node "
                + str(exterior_interface_id)
                + "."
            )

        for sibling_id in new_sibling_ids:
            dummy_sibling_ids = new_sibling_ids + [new_interface_node_id]
            dummy_sibling_ids.remove(sibling_id)
            self.interface_nodes[sibling_id].update_topology(
                None,
                None,
                None,
                dummy_sibling_ids,
                None,
            )
            if verbose:
                print(
                    "Updated topology of sibling interface node "
                    + str(sibling_id)
                    + "."
                )

        for child_id in new_child_ids:
            dummy_sibling_ids = _deepcopy(new_child_ids)
            dummy_sibling_ids.remove(child_id)
            self.interface_nodes[child_id].update_topology(
                new_subdomain_node_id,
                None,
                new_interface_node_id,
                dummy_sibling_ids,
                None,
            )
            if verbose:
                print("Updated topology of child interface node " + str(child_id) + ".")

        # Update edges and interface connectors that have their propagating
        # medium changed, and remove redundant interface connectors

        for child_id in new_child_ids:
            for edge in self.edges:
                if (
                    edge.subdomain_id == exterior_subdomain_id
                    and edge.interface_id == child_id
                ):
                    edge.update_topology(new_subdomain_node_id, None, None)
                    if verbose:
                        print(
                            "Updated subdomain topology of edge "
                            + str(edge.identifier)
                            + "."
                        )

            for connector in self.interface_connectors:
                if (
                    connector.subdomain_id == exterior_subdomain_id
                    and connector.interfaces_ids == (child_id, child_id)
                    and connector.topology == "self-exterior"
                ):
                    connector.update_topology(None, new_subdomain_node_id, None)
                    if verbose:
                        print(
                            "Updated subdomain topology of interface connector "
                            + str(connector.identifier)
                            + "."
                        )

                if (
                    connector.subdomain_id == exterior_subdomain_id
                    and child_id in connector.interfaces_ids
                    and connector.topology in ("parent-child", "sibling")
                ):
                    connector.inactivate()
                    if verbose:
                        print(
                            "Inactivated interface connector "
                            + str(connector.identifier)
                            + "."
                        )

        # Create new edges and interface connectors

        self.edges.append(
            Edge(
                len(self.edges),
                exterior_subdomain_id,
                new_interface_node_id,
                "interface_interior_to_subdomain",
            )
        )
        if verbose:
            print("Created new edge " + str(len(self.edges) - 1) + ".")

        self.edges.append(
            Edge(
                len(self.edges),
                new_subdomain_node_id,
                new_interface_node_id,
                "interface_exterior_to_subdomain",
            )
        )
        if verbose:
            print("Created new edge " + str(len(self.edges) - 1) + ".")

        self.interface_connectors.append(
            InterfaceConnector(
                len(self.interface_connectors),
                (new_interface_node_id, new_interface_node_id),
                exterior_subdomain_id,
                "self-exterior",
            )
        )
        if verbose:
            print(
                "Created new self-exterior interface connector "
                + str(len(self.interface_connectors) - 1)
                + "."
            )

        self.interface_connectors.append(
            InterfaceConnector(
                len(self.interface_connectors),
                (new_interface_node_id, new_interface_node_id),
                new_subdomain_node_id,
                "self-interior",
            )
        )
        if verbose:
            print(
                "Created new self-interior interface connector "
                + str(len(self.interface_connectors) - 1)
                + "."
            )

        if self.interface_nodes[exterior_interface_id].bounded:
            self.interface_connectors.append(
                InterfaceConnector(
                    len(self.interface_connectors),
                    (exterior_interface_id, new_interface_node_id),
                    exterior_subdomain_id,
                    "parent-child",
                )
            )
            if verbose:
                print(
                    "Created new parent-child interface connector "
                    + str(len(self.interface_connectors) - 1)
                    + "."
                )

        for sibling_id in new_sibling_ids:
            self.interface_connectors.append(
                InterfaceConnector(
                    len(self.interface_connectors),
                    (sibling_id, new_interface_node_id),
                    exterior_subdomain_id,
                    "sibling",
                )
            )
            if verbose:
                print(
                    "Created new sibling interface connector "
                    + str(len(self.interface_connectors) - 1)
                    + "."
                )

        for child_id in new_child_ids:
            self.interface_connectors.append(
                InterfaceConnector(
                    len(self.interface_connectors),
                    (new_interface_node_id, child_id),
                    new_subdomain_node_id,
                    "parent-child",
                )
            )
            if verbose:
                print(
                    "Created new parent-child interface connector "
                    + str(len(self.interface_connectors) - 1)
                    + "."
                )

    def inactivate_interface(self, interface_id, verbose=False):
        """
        Remove an interface from the topology.

        Interfaces are only removed from the topology, but the node object is
        not removed. Hence, the memory will not be cleared. The topology
        will be updated accordingly.

        Parameters
        ----------
        interface_id : int
            The identifier of the interface.
        verbose : bool
            Display information.
        """

        if interface_id < 0 or interface_id >= len(self.interface_nodes):
            raise ValueError(
                "The interface "
                + str(interface_id)
                + " does not exist in the topology."
            )

        if not self.interface_nodes[interface_id].is_active():
            raise ValueError(
                "The interface " + str(interface_id) + " is not active in the topology."
            )

        inactivated_interface = self.interface_nodes[interface_id]

        # Inactivate the interface, and the corresponding interior subdomain,
        # all edges, and all interface connectors.

        inactivated_interface.inactivate()
        if verbose:
            print("Inactivated interface node " + str(interface_id) + ".")

        interior_subdomain_id = inactivated_interface.child_subdomain_id
        self.subdomain_nodes[interior_subdomain_id].inactivate()
        for edge in self.edges:
            if edge.interface_id == interface_id:
                edge.inactivate()
                if verbose:
                    print("Inactivated edge " + str(edge.identifier) + ".")
        for connector in self.interface_connectors:
            if interface_id in connector.interfaces_ids:
                connector.inactivate()
                if verbose:
                    print(
                        "Inactivated interface connector "
                        + str(connector.identifier)
                        + "."
                    )

        # Update the exterior subdomain of the child interfaces, its edges
        # and self-interacting interface connectors

        for interface in inactivated_interface.child_interfaces_ids:
            self.interface_nodes[interface].update_topology(
                inactivated_interface.parent_subdomain_id,
                None,
                inactivated_interface.parent_interface_id,
                None,
                None,
            )
            if verbose:
                print(
                    "Updated parent topology of interface node " + str(interface) + "."
                )
        for edge in self.edges:
            if edge.subdomain_id == interface_id:
                edge.update_topology(
                    inactivated_interface.parent_subdomain_id,
                    None,
                    None,
                )
                if verbose:
                    print(
                        "Updated subdomain topology of edge "
                        + str(edge.identifier)
                        + "."
                    )
        for connector in self.interface_connectors:
            if (
                connector.subdomain_id == interface_id
                and connector.topology == "self-exterior"
            ):
                connector.update_topology(
                    None,
                    inactivated_interface.parent_subdomain_id,
                    None,
                )
                if verbose:
                    print(
                        "Updated topology of self-exterior interface connector "
                        + str(connector.identifier)
                        + "."
                    )

        # Create new interface connectors

        for interface in inactivated_interface.child_interfaces_ids:
            self.interface_connectors.append(
                InterfaceConnector(
                    len(self.interface_connectors),
                    (interface, inactivated_interface.parent_interface_id),
                    inactivated_interface.parent_subdomain_id,
                    "parent-child",
                )
            )
            if verbose:
                print(
                    "Created new parent-child interface connector "
                    + str(len(self.interface_connectors) - 1)
                    + "."
                )

            for sibling in inactivated_interface.sibling_interfaces_ids:
                self.interface_connectors.append(
                    InterfaceConnector(
                        len(self.interface_connectors),
                        (interface, sibling),
                        inactivated_interface.parent_subdomain_id,
                        "sibling",
                    )
                )
            if verbose:
                print(
                    "Created new sibling interface connector "
                    + str(len(self.interface_connectors) - 1)
                    + "."
                )

        return

    def change_material(self, subdomain_id, material, verbose=False):
        """
        Change the material of a subdomain.

        Parameters
        ----------
        subdomain_id : int
            The identifier of the subdomain.
        material : optimus.material.common.Material
            The new material of the subdomain.
        verbose : bool
            Display information.
        """

        if subdomain_id < 0 or subdomain_id >= len(self.subdomain_nodes):
            raise ValueError(
                "The subdomain "
                + str(subdomain_id)
                + " does not exist in the topology."
            )
        subdomain_node = self.subdomain_nodes[subdomain_id]
        if not subdomain_node.is_active():
            raise ValueError(
                "The subdomain " + str(subdomain_id) + " is not active in the topology."
            )

        if verbose:
            print(
                "Change material of subdomain "
                + str(subdomain_id)
                + " from "
                + subdomain_node.material.name
                + " to "
                + material.name
                + "."
            )

        subdomain_node.update_material(material)

        return

    def change_geometry(self, interface_id, geometry, verbose=False):
        """
        Change the geometry of an interface.

        Parameters
        ----------
        interface_id : int
            The identifier of the interface.
        geometry : optimus.geometry.common.Geometry
            The new geometry of the interface.
        verbose : bool
            Display information.
        """

        if interface_id < 0 or interface_id >= len(self.interface_nodes):
            raise ValueError(
                "The interface "
                + str(interface_id)
                + " does not exist in the topology."
            )
        interface_node = self.interface_nodes[interface_id]
        if not interface_node.is_active():
            raise ValueError(
                "The interface " + str(interface_id) + " is not active in the topology."
            )

        if verbose:
            print(
                "Change geometry of interface "
                + str(interface_id)
                + " from "
                + interface_node.geometry.label
                + " to "
                + geometry.label
                + "."
            )

        interface_node.update_geometry(geometry)

        return

    def _check_subdomain(self, subdomain_id):
        """
        Check validity of subdomain node.

        Check if the provided identifier points to an existing and active subdomain.

        Parameters
        ----------
        subdomain_id : int
            The identifier of the subdomain node.

        Returns
        -------
        subdomain : optimus.geometry.graph_topology.SubdomainNode
            The subdomain node.
        """

        if subdomain_id >= len(self.subdomain_nodes):
            raise ValueError(
                "The subdomain "
                + str(subdomain_id)
                + " does not exist in the topology."
            )

        subdomain = self.subdomain_nodes[subdomain_id]

        if not subdomain.is_active():
            raise ValueError(
                "The subdomain " + str(subdomain_id) + " is not active in the topology."
            )

        return subdomain

    def _check_new_geometry(self, geometry):
        """
        Check validity of a new geometry.

        Throw a warning if the geometry is already present in the topology.

        Parameters
        ----------
        geometry : optimus.geometry.common.Geometry
            The new geometry.
        """

        import warnings as _warnings

        active_geometry_labels = []
        for interface in self.interface_nodes:
            if interface.is_active() and interface.geometry is not None:
                active_geometry_labels.append(interface.geometry.label)

        if geometry is not None:
            if geometry.label in active_geometry_labels:
                _warnings.warn(
                    "Another geometry with label "
                    + geometry.label
                    + " already exists in the topology.",
                    UserWarning,
                )

        return

    def _split_subdomain(self, subdomain, child_ids):
        """
        Split subdomain and place the child interfaces interior to the new subdomain.

        Parameters
        ----------
        subdomain : optimus.geometry.graph_topology.SubdomainNode
            The subdomain to split.
        child_ids : list[int]
            The identifiers of the child interfaces.

        Returns
        -------
        new_child_ids, new_sibling_ids : tuple[list[int], list[int]]
            The identifierss of the new child and sibling interface nodes.
        """

        for node in child_ids:
            if node >= len(self.interface_nodes):
                raise ValueError(
                    "The interface node " + str(node) + " does not exist in "
                    "the topology and cannot be added interior to the new subdomain."
                )
            if not self.interface_nodes[node].is_active():
                raise ValueError(
                    "The interface node " + str(node) + " is not active in "
                    "the topology and cannot be added interior to the new subdomain."
                )

        existing_child_interfaces = subdomain.child_interfaces_ids

        if len(child_ids) == 0:
            new_child_ids = []
            new_sibling_ids = existing_child_interfaces
        else:
            for node in child_ids:
                if node not in existing_child_interfaces:
                    raise ValueError(
                        "Interface" + str(node) + " cannot be placed interior "
                        "to the new subdmain because it is not in the existing "
                        "subdomain to be split."
                    )
            new_child_ids = []
            new_sibling_ids = []
            for node in existing_child_interfaces:
                if node in child_ids:
                    new_child_ids.append(node)
                else:
                    new_sibling_ids.append(node)

        return new_child_ids, new_sibling_ids

    def find_interface_connectors_of_interface(self, interface_id, topology=None):
        """
        Find the interface connectors of an interface.

        Parameters
        ----------
        interface_id : int
            The identifier of the interface.
        topology : None, str
            The topology of the connector:
             - "self-exterior": self interaction via exterior subdomain
             - "self-interior": self interaction via interior subdomain
             - "parent-child": interaction from parent to child
             - "sibling": interaction between siblings
            If None, all connectors are returned.

        Returns
        -------
        interface_connectors : list[InterfaceConnector]
            The identifiers of the interface connectors.
        """

        interface_connectors = []
        for connector in self.interface_connectors:
            if connector.is_active() and interface_id in connector.interfaces_ids:
                if topology is None or connector.topology == topology:
                    interface_connectors.append(connector)

        return interface_connectors


class _GraphComponent:
    def __init__(self, identifier):
        """
        Create a graph component.

        Parameters
        ----------
        identifier : int
            The unique identifier of the graph component.
        """

        from optimus.utils.conversions import convert_to_positive_int

        self.identifier = convert_to_positive_int(
            identifier, "graph component identifier"
        )
        self.active = True

        return

    def is_active(self):
        """Check if the component is active in the graph."""
        return self.active

    def inactivate(self):
        """Inactivate the graph component."""
        self.active = False
        return


class _GraphNode(_GraphComponent):
    def __init__(self, identifier, bounded):
        """
        Create a graph node.

        Parameters
        ----------
        identifier : int
            The unique identifier of the graph component.
        bounded : bool
            Unbounded interfaces and subdomains are the root nodes of the topology.
        """

        super().__init__(identifier)

        self.bounded = bounded

        return

    def inactivate(self):
        """Inactivate the node."""

        if self.bounded:
            super().inactivate()
        else:
            raise AssertionError(
                "The unbounded regions cannot be inactivated. "
                "They are the root node of the topology."
            )

        return


class SubdomainNode(_GraphNode):
    def __init__(
        self,
        identifier,
        parent_interface,
        child_interfaces,
        bounded,
        material,
        source=None,
    ):
        """
        Create a subdomain node in a graph topology.

        A subdomain node represents a volumetric subdomain of the three-dimensional
        space. Its boundary consists of a number of interfaces, stored in interface
        nodes, and linked together with edges.

        Parameters
        ----------
        identifier : int
            The unique identifier of the subdomain node in the graph.
        parent_interface : None, int
            The identifier of the parent interface, i.e., the exterior boundary
            of the subdomain. Is None for the root node.
        child_interfaces : list[int]
            The identifiers of the child interfaces, i.e., the interior boundaries
            of the subdomain. Is empty for child nodes.
        bounded : bool
            True if the subdomain is bounded, False if it is the unbounded exterior.
        material : optimus.material.common.Material
            The homogeneous propagating material of the subdomain.
        source : None, optimus.source.common.Source
            The source of the incident wave field inside the subdomain.
        """

        super().__init__(identifier, bounded)

        self.parent_interface_id = parent_interface
        self.child_interfaces_ids = child_interfaces

        self.material = material

        self.sources = []
        if source is not None:
            self.add_source(source)

        return

    def update_topology(self, new_parent, new_children):
        """
        Update the node topology.

        Parameters
        ----------
        new_parent : None, int
            The identifier of the parent interface, i.e., the exterior boundary
            of the subdomain. Is None for the root node.
        new_children : list[int]
            The identifiers of the chile interfaces, i.e., the interior boundaries
            of the subdomain. Is empty for child nodes.
        """

        if new_parent is not None:
            self.parent_interface_id = new_parent
        if new_children is not None:
            self.child_interfaces_ids = new_children

        return

    def print_topology(self):
        """
        Print the topology of the subdomain node.
        """

        if self.is_active():
            print(" Subdomain node " + str(self.identifier) + ":")
            print("  Parent interface: " + str(self.parent_interface_id))
            print("  Child interfaces: " + str(self.child_interfaces_ids))
            print("  Bounded: " + str(self.bounded))
        else:
            print(" Subdomain node " + str(self.identifier) + " is inactive.")

        return

    def update_material(self, material):
        """
        Update the material of the subdomain node.

        Parameters
        ----------
        material : optimus.material.common.Material
            The homogeneous propagating material of the subdomain.
        """

        self.material = material

        return

    def print_material(self):
        """
        Print the material of the subdomain node.
        """

        if self.is_active():
            print(" Subdomain node " + str(self.identifier) + ":")
            print("  Material: " + self.material.name)

        return

    def add_source(self, source):
        """
        Add a source definition to the subdomain.

        Parameters
        ----------
        source : None, optimus.source.common.Source
            The source of the incident wave field.
        """

        self.sources.append(source)

        return

    def print_sources(self):
        """
        Print the sources of the subdomain node.
        """

        if self.is_active():
            print(" Subdomain node " + str(self.identifier) + ":")
            if len(self.sources) == 0:
                print("  No sources have been specified")
            else:
                print("  Sources: ")
                for source in self.sources:
                    print("   " + source.type)

        return


class InterfaceNode(_GraphNode):
    def __init__(
        self,
        identifier,
        parent_subdomain,
        child_subdomain,
        parent_interface,
        sibling_interfaces,
        child_interfaces,
        bounded,
        geometry,
    ):
        """
        Create an interface node in a graph topology.

        An interface node represents a surface between two volumetric subdomains.
        It is connected via edges to its exterior and interior subdomain node.

        Parameters
        ----------
        identifier : int
            The unique identifier of the interface node in the graph.
        parent_subdomain : None, int
            The identifier of the parent subdomain, i.e., exterior to the interface.
            Is None for the root node.
        child_subdomain : int
            The identifier of the parent subdomain, i.e., interior to the interface.
        parent_interface : None, int
            The identifier of the parent interface.
            Is None for the root node.
        sibling_interfaces : list[int]
            The identifiers of the sibling interfaces, if present in the graph.
        child_interfaces : list[int]
            The identifiers of the child interfaces. Is empty for the leaf node.
        bounded : bool
            True if the interface is bounded, False if it is the unbounded exterior.
        geometry : None, optimus.geometry.common.Geometry
            The geometry of the interface, which includes the surface mesh.
            Is None for the root node.
        """

        super().__init__(identifier, bounded)

        self.parent_subdomain_id = parent_subdomain
        self.child_subdomain_id = child_subdomain
        self.parent_interface_id = parent_interface
        self.sibling_interfaces_ids = sibling_interfaces
        self.child_interfaces_ids = child_interfaces

        self.geometry = geometry

        return

    def update_topology(
        self,
        new_parent_subdomain,
        new_child_subdomain,
        new_parent_interface,
        new_sibling_interfaces,
        new_child_interfaces,
    ):
        """
        Update the node topology.

        Parameters
        ----------
        new_parent_subdomain : None, int
            The identifier of the parent subdomain, i.e., exterior to the interface.
            Is None for the root node.
        new_child_subdomain : int
            The identifier of the parent subdomain, i.e., interior to the interface.
        new_parent_interface : None, int
            The identifier of the parent interface.
            Is None for the root node.
        new_sibling_interfaces : list[int]
            The identifiers of the sibling interfaces, if present in the graph.
        new_child_interfaces : list[int]
            The identifiers of the child interfaces. Is empty for the leaf node.
        """

        if new_parent_subdomain is not None:
            self.parent_subdomain_id = new_parent_subdomain
        if new_child_subdomain is not None:
            self.child_subdomain_id = new_child_subdomain
        if new_parent_interface is not None:
            self.parent_interface_id = new_parent_interface
        if new_sibling_interfaces is not None:
            self.sibling_interfaces_ids = new_sibling_interfaces
        if new_child_interfaces is not None:
            self.child_interfaces_ids = new_child_interfaces

        return

    def print_topology(self):
        """
        Print the topology of the interface node.
        """

        if self.is_active():
            print(" Interface node " + str(self.identifier) + ":")
            print("  Parent subdomain: " + str(self.parent_subdomain_id))
            print("  Child subdomain: " + str(self.child_subdomain_id))
            print("  Parent interface: " + str(self.parent_interface_id))
            print("  Sibling interfaces: " + str(self.sibling_interfaces_ids))
            print("  Child interfaces: " + str(self.child_interfaces_ids))
            print("  Bounded: " + str(self.bounded))
        else:
            print(" Interface node " + str(self.identifier) + " is inactive.")

        return

    def update_geometry(self, geometry):
        """
        Update the geometry of the interface node.

        Parameters
        ----------
        geometry : optimus.geometry.common.Geometry
            The geometry of the interface, which includes the surface mesh.
        """

        self.geometry = geometry

        return

    def print_geometry(self):
        """
        Print the geometry of the interface node.
        """

        if self.is_active():
            print(" Interface node " + str(self.identifier) + ":")
            if not self.bounded:
                print("  Unbounded exterior interface")
            elif self.geometry is not None:
                print("  Geometry: " + self.geometry.label)
                print(
                    "  Number of vertices: " + str(self.geometry.number_of_vertices())
                )
            else:
                print("  No geometry has been specified")

        return


class Edge(_GraphComponent):
    def __init__(
        self,
        identifier,
        subdomain_node,
        interface_node,
        orientation,
    ):
        """
        Create an edge in the graph topology.

        An edge represents the connection between one interface and one subdomain.
        They represent potential integral operators in the BEM.

        Parameters
        ----------
        identifier : int
            The unique identifier of the edge in the graph.
        subdomain_node : int
            The identifier of the subdomain.
        interface_node : int
            The identifier of the interface.
        orientation : str
            The orientation of the edge:
             - "interface_interior_to_subdomain"
             - "interface_exterior_to_subdomain"
        """

        super().__init__(identifier)

        self.subdomain_id = subdomain_node
        self.interface_id = interface_node
        self.orientation = self.check_topology(orientation)

        return

    @staticmethod
    def check_topology(topology):
        """
        Check the validity of the topology.

        Parameters
        ----------
        topology : str
            The orientation of the edge:
             - "interface_interior_to_subdomain"
             - "interface_exterior_to_subdomain"
        """

        topology = topology.lower()

        if topology in (
            "interface_interior_to_subdomain",
            "interface_exterior_to_subdomain",
        ):
            return topology
        else:
            raise ValueError("Topology of edge is not correctly defined.")

    def update_topology(self, subdomain, interface, topology):
        """
        Update the edge topology.

        Parameters
        ----------
        subdomain : int
            The identifier of the subdomain.
        interface : int
            The identifier of the interface.
        topology : str
            The orientation of the edge:
             - "interface_interior_to_subdomain"
             - "interface_exterior_to_subdomain"
        """

        if subdomain is not None:
            self.subdomain_id = subdomain
        if interface is not None:
            self.interface_id = interface
        if topology is not None:
            self.orientation = self.check_topology(topology)

        return

    def print_topology(self):
        """
        Print the topology of the edge.
        """

        if self.is_active():
            print(" Edge " + str(self.identifier) + ":")
            print("  Subdomain node: " + str(self.subdomain_id))
            print("  Interface node: " + str(self.interface_id))
            print("  Orientation: " + str(self.orientation))
        else:
            print(" Edge " + str(self.identifier) + " is inactive.")

        return


class InterfaceConnector(_GraphComponent):
    def __init__(
        self,
        identifier,
        interface_nodes,
        subdomain_node,
        topology,
    ):
        """
        Create an interface connector in the graph topology.

        An interface connector is an interaction between two adjacent interface,
        which share the same propagating subdomain.
        They represent potential integral operators in the BEM.

        Parameters
        ----------
        identifier : int
            The unique identifier of the interface connector in the graph.
        interface_nodes : tuple[int]
            The identifiers of the two interface nodes.
        subdomain_node : int
            The identifier of the subdomain node.
        topology : str
            The topology of the connector:
             - "self-exterior": self interaction via exterior subdomain
             - "self-interior": self interaction via interior subdomain
             - "parent-child": interaction from parent to child
             - "sibling": interaction between siblings
        """

        super().__init__(identifier)

        self.topology, self.interfaces_ids = self.check_topology(
            topology, interface_nodes
        )
        self.subdomain_id = subdomain_node

        return

    @staticmethod
    def check_topology(topology, interface_nodes):
        """
        Check the validity of the topology.

        Parameters
        ----------
        topology : str
            The topology of the connector:
             - "self-exterior": self interaction via exterior subdomain
             - "self-interior": self interaction via interior subdomain
             - "parent-child": interaction from parent to child
             - "sibling": interaction between siblings
        interface_nodes : tuple[int]
            The identifiers of the two interface nodes.
        """

        if not isinstance(topology, str):
            raise TypeError("Topology of interface connector has to be a string.")
        else:
            topology = topology.lower()

        if not isinstance(interface_nodes, tuple):
            raise TypeError("Interface nodes of interface connector has to be a tuple.")
        elif len(interface_nodes) != 2:
            raise ValueError(
                "Interface nodes of interface connector has to be a tuple of length 2."
            )
        elif not isinstance(interface_nodes[0], int) or not isinstance(
            interface_nodes[1], int
        ):
            raise TypeError(
                "Interface nodes of interface connector has to be a tuple of integers."
            )

        if topology in ("self-exterior", "self-interior"):
            if interface_nodes[0] != interface_nodes[1]:
                raise AssertionError(
                    "Self interactions has to be between identical interfaces."
                )
        elif topology in ("parent-child", "sibling"):
            if interface_nodes[0] == interface_nodes[1]:
                raise AssertionError(
                    "Parent-child and sibling interactions have to be between "
                    "different interfaces."
                )
        else:
            raise AssertionError("Unknown topology of interface connector: " + topology)

        return topology, interface_nodes

    def update_topology(self, interface_nodes, subdomain_node, topology):
        """
        Update the connector topology.

        Parameters
        ----------
        interface_nodes : None, tuple[int]
            The identifiers of the two interface nodes.
        subdomain_node : None, int
            The identifier of the subdomain node.
        topology : None, str
            The topology of the connector:
             - "self-exterior": self interaction via exterior subdomain
             - "self-interior": self interaction via interior subdomain
             - "parent-child": interaction from parent to child
             - "sibling": interaction between siblings
        """

        if interface_nodes is not None:
            self.interfaces_ids = interface_nodes
        if subdomain_node is not None:
            self.subdomain_id = subdomain_node
        if topology is not None:
            self.topology = topology

        _ = self.check_topology(self.topology, self.interfaces_ids)

        return

    def print_topology(self):
        """
        Print the topology of the connector.
        """

        if self.is_active():
            print(" Interface connector " + str(self.identifier) + ":")
            print("  Interface nodes: " + str(self.interfaces_ids))
            print("  Subdomain node: " + str(self.subdomain_id))
            print("  Topology: " + str(self.topology))
        else:
            print(" Interface connector " + str(self.identifier) + " is inactive.")

        return
