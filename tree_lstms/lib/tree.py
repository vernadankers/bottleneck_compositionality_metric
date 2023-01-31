class CustomTree:
    def __init__(self, tree, string_representation, sentiment_tree=False):
        self.tree = tree

        self.sentiment_tree = sentiment_tree
        self.string_repr = string_representation.replace(
            "(", "").replace(")", "").split()
        self.children = []
        if len(str(self.tree).split()) > 1:
            for child in self.tree:
                child = CustomTree(child, string_representation,
                                   sentiment_tree=sentiment_tree)
                self.children.append(child)
        self.leaves = tuple(self.get_leaves())
        self.unrolled = self.unroll(self, [], 0)

    def flatten(self):
        if self.sentiment_tree:
            if not self.children[0].children:
                #assert
                return [self]
            return [self] + [x for c in self.children for x in c.flatten()]
        return [self]

    def is_leaf(self):
        return len(str(self.tree).split()) == 1

    def idx(self):
        """
        Return the index of the current root node. For sentiment trees,
        return the target (= sentiment) of the current subtree.
        Returns:
            integer representing the label
            None / floating point target
        """
        if len(str(self.tree).split()) == 1:
            return int(self.tree), None
        if self.sentiment_tree:
            # Get the target of this subtree
            sub_target = float(self.string_repr[int(self.tree._label)])
            return int(self.tree.label()), sub_target
        #print(len(str(self.tree).split()))
        #print(self.tree.label())
        return int(self.tree.label()), None

    def __str__(self):
        """
        Create a string representation by printing children & root of the tree.
        Returns:
            str
        """
        if isinstance(self.tree, str):
            return self.string_repr[int(self.tree)]
        child1 = str(self.children[0])
        root = self.string_repr[self.idx()[0]]

        if len(self.children) == 1:
            return f"( {root} {child1} )"
        child2 = self.children[1]
        if self.sentiment_tree:
            return f"( {root} {child1} {child2} )"
        return f"( {child1} {root} {child2} )"

    def get_leaves(self):
        """
        Create a list of the leave nodes that are at any point below the
        current (root) node.
        For a leave node, return a list with only that leave node.
        Returns:
            list of str
        """
        if isinstance(self.tree, str):
            return [self.string_repr[int(self.tree)]]
        return [l for c in self.children for l in c.get_leaves()]

    def unroll(self, tree, trace=[], depth=0):
        """
        Function that recursively collects depths, and indices of child nodes.
        Args:
            tree (CustomTree)
            trace (set)
            depth (int)
        Returns:
            updated trace, with the current node added
            depth + 1
        """
        children = tree.children
        idx, target = tree.idx()
        if not children:
            trace.append((depth, (idx,), None))
        else:
            depths = []
            indices = [idx]
            for child in children:
                trace, depth_ = self.unroll(child, trace, depth)
                depths.append(depth_)
                indices.append(child.idx()[0])
            depth = max(depths)
            trace.append(
                (depth, tuple(indices), target))
        return trace, depth + 1
