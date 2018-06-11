import numpy as np


class BinaryTree:

    root = None
    node_index = None

    def __init__(self, index):
        self.root = BinaryTreeNode(index)
        self.node_index = {index: self.root}

    def add_left_descendant(self, index, parent_index):
        parent = self.node_index[parent_index]
        new_node = BinaryTreeNode(index, parent)
        self.node_index[index] = new_node
        parent.add_left_descendant(new_node)

    def has_left_descendant_at_node(self, index):
        return self.node_index[index].has_left_descendant()

    def add_right_descendant(self, index, parent_index):
        parent = self.node_index[parent_index]
        new_node = BinaryTreeNode(index, parent)
        self.node_index[index] = new_node
        parent.add_right_descendant(new_node)

    def has_right_descendant_at_node(self, index):
        return self.node_index[index].has_right_descendant()

    def set_word(self, index, word):
        self.node_index[index].set_word(word)

    def print_tree(self):
        self.root.recursive_print()

    def get_sentence(self):
        sentence = ' '.join([n.word for n in self.node_index.values() if n.word != '_PAD_'])
        return sentence

    def get_words(self):
        return [n.word for n in self.node_index.values()]

    def convert_to_ptb_format(self):
        return self.root.convert_to_ptb()

    def convert_to_sequence_and_masks(self, head_node):
        """
        Convert a subtree into a sequence of words, and corresponding masks of the root
        :param head_node: the node to treat as the root of the (sub)tree
        :return words: list of words in tree order
        :return left_mask, right_mask: masks denoting the tree structure
        """
        sequence = head_node.get_children_in_sequence()
        sequence.reverse()
        sequence_map = {}
        for s_i, s in enumerate(sequence):
            sequence_map[s] = s_i
        # sequence_map = {s: s_i for s_i, s in enumerate(sequence)}
        n_elements = len(sequence)
        left_mask = np.zeros([n_elements, n_elements], dtype=np.float32)
        right_mask = np.zeros([n_elements, n_elements], dtype=np.float32)
        for s_i, n_i in enumerate(sequence):
            node = self.node_index[n_i]
            if node.has_left_descendant():
                left_mask[s_i, sequence_map[node.left_descendant.index]] = 1.
            if node.has_right_descendant():
                right_mask[s_i, sequence_map[node.right_descendant.index]] = 1.
        words = [self.node_index[n_i].word for n_i in sequence]
        return words, left_mask, right_mask

    def get_parents(self):
        sequence = self.root.get_children_in_sequence()
        words = [self.node_index[n_i].word for n_i in sequence]
        parents = [self.node_index[n_i].parent.index + 1 if self.node_index[n_i].parent else 0 for n_i in sequence]
        indexs = [self.node_index[n_i].index+1 for n_i in sequence]

        return words, parents, indexs




class BinaryTreeNode:

    word = None
    index = None
    parent = None
    left_descendant = None
    right_descendant = None

    def __init__(self, index, parent=None):
        self.index = index
        self.word = '_PAD_'
        if parent is not None:
            self.parent = parent

    def __str__(self):
        return '(%s %s)' % (self.word, self.index)

    def set_word(self, word):
        self.word = word

    def add_left_descendant(self, new_node):
        self.left_descendant = new_node

    def has_left_descendant(self):
        return self.left_descendant is not None

    def add_right_descendant(self, new_node):
        self.right_descendant = new_node

    def has_right_descendant(self):
        return self.right_descendant is not None

    def recursive_print(self, depth=0):
        print '  '*depth, self.word, self.index
        if self.left_descendant is not None:
            self.left_descendant.recursive_print(depth+1)
        if self.right_descendant is not None:
            self.right_descendant.recursive_print(depth+1)

    def convert_to_ptb(self):
        ptb_string = '(' 
        if self.word != '_PAD_':
            ptb_string += ' ' + self.word
        if self.left_descendant is not None:
            ptb_string += ' ' + self.left_descendant.convert_to_ptb()
        if self.right_descendant is not None:
            ptb_string += ' ' + self.right_descendant.convert_to_ptb()
        ptb_string += ')'
        return ptb_string

    def get_leaf_nodes(self):
        leaves = []
        if self.left_descendant is None and self.right_descendant is None:
            leaves.append(self.word)
        else:
            if self.left_descendant is not None:
                leaves.extend(self.left_descendant.get_leaf_nodes())
            if self.right_descendant is not None:
                leaves.extend(self.right_descendant.get_leaf_nodes())
        return leaves

    def get_children_in_sequence(self):
        sequence = []
        if self.left_descendant is None and self.right_descendant is None:
            sequence.append(self.index)
        else:
            if self.left_descendant is not None:
                sequence.extend(self.left_descendant.get_children_in_sequence())
            if self.right_descendant is not None:
                sequence.extend(self.right_descendant.get_children_in_sequence())
            sequence = [self.index] + sequence
        return sequence


