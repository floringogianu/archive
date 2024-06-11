""" Stats of the TinyImagenet dataset used to inform the creation of a
    continual learning setup.

    We are interested in balancing the number of tasks (datasets) and the no
    of classes, where a class in a dataset is in the hierarchy of a
    superclass (bighorn in dataset 0 is part of ungulate in the table below).

    Some ASCII art is more than a thousand words:

                |  D0           |  D1               |  D2          |  D3
    dog         | shepherd dog  | golden retriever  | labrador     | terrier
    ungulate    | bighorn       | pig               | ox           | camel
    insect      | fly           | sulphur butterfly | bee          | dragonfly
    instruments | binoculars    | hourglass         | syringe      | stopwatch
    vehicle     | sports car    | waggon            | convertible  | limo
    ...
    misc        | ufo           | J. Bieber memes   | polenta      | idk...

    A node in the graph would be a superclass. For examples: superclass:
    `insect` with leafs `fly`, `butterfly`, `bee`, `dragonfly`, etc.

    Some more terminology I might use below, sorry for the mess:
    node == superclass
    leaf == class
    task == dataset
    random == miscellanous, refers to classes belonging to superclasses not
        meeting some constraints refering to the no of leafs they have.
        Maybe these could be useful too in a miscellanous/random superclass.
    
    A `wnid` is a WordNet ID and is used to identify both classes and
    superclasses in the ImageNet hierarchy.
"""
import csv
from collections import defaultdict


def consolidate_parents(p2c, c2p):
    """ Keeps only the parents with the largest number of children.
    """
    c2p_ = {}
    p2c_ = defaultdict(list)
    for child, parents in c2p.items():
        parent_, children_no = None, 0
        for parent in parents:
            if len(p2c[parent]) > children_no:
                parent_ = parent
        c2p_[child] = parent_
        p2c_[parent_].append(child)
    return p2c_, c2p_


def get_hierarchy():
    """ Each line in `is_a.txt` is a parent -> child relation.
        Parsed 75850 rows.
        parents:   75850 ->  16693 unique.
        children:  75850 ->  74389 unique.
    """
    p2c = defaultdict(list)
    c2p = defaultdict(list)
    with open("data/tiny-imagenet-200/is_a.txt") as csv_file:
        for row_cnt, row in enumerate(csv.reader(csv_file, delimiter=" ")):
            p2c[row[0]].append(row[1])
            c2p[row[1]].append(row[0])
    print(f"Parsed {row_cnt + 1} lines from imagenet `is_a.txt`.")
    print(f"p2c: {len(p2c)} parents.")
    print(f"c2p: {len(c2p)} children.")
    return p2c, c2p


def get_all_parents(wnids, c2p, wnid2words=None):
    all_parents = defaultdict(list)
    for cnt, wnid in enumerate(wnids):
        if wnid2words is not None:
            print(cnt, wnid, wnid2words[wnid])
        parent = c2p[wnid]
        while parent in c2p:
            all_parents[wnid].append(parent)
            parent = c2p[parent]
    return all_parents


def display_parents(all_parents, wnid2words):
    print("\nAncestors:")
    for cnt, (child, ancestors) in enumerate(all_parents.items()):
        # some_ancestor = wnid2words[ancestors[-4]]
        words = wnid2words[child]
        print(f"{cnt} {child} with {len(ancestors):2d} parents. ({words})")
        for ancestor in ancestors:
            print(f"  -->  {wnid2words[ancestor]} [{ancestor}]")


class DirectedGraph:
    def __init__(self):
        self._adj = defaultdict(list)
        self._root = None

    def add_edge(self, v, w):
        if w not in self._adj[v]:
            self._adj[v].append(w)
        # add the leafs also
        if w not in self._adj:
            self._adj[w]

    def get_adj(self, v):
        return self._adj[v]

    def reversed(self):
        reverse_digraph = DirectedGraph()
        for parent in self._adj.keys():
            for child in self.get_adj(parent):
                reverse_digraph.add_edge(child, parent)
        return reverse_digraph

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, v):
        self._root = v

    def get_vertices_cnt(self):
        return len(self._adj)

    def get_edges_cnt(self):
        return sum(len(v) for v in self._adj.values())

    def __str__(self):
        s = f"size: {len(self)}, with root in {self._root}\n"
        for parent, children in self._adj.items():
            s += f"{parent}({len(children):2d})  -->  {children}\n"
        return s

    def __len__(self):
        return self.get_vertices_cnt()


class DFSLeafCount:
    def __init__(self, digraph, source):
        self._digraph = digraph
        self._source = source
        self._marked = defaultdict(bool)
        self._leafs = defaultdict(set)
        self._edge_to = {}
        self.dfs(source)

    def dfs(self, v):
        self._marked[v] = True

        if not self._digraph.get_adj(v):
            # if is leaf, update the leaf set of each ancestor of `v`
            self.update_node_info(v)

        for child in self._digraph.get_adj(v):
            if not self._marked[child]:
                self._edge_to[child] = v
                self.dfs(child)

    def update_node_info(self, v):
        """ Update leaf set of each ancestor of `v`. """
        x = self._edge_to[v]
        while x != self._source:
            self._leafs[x].update({v})
            x = self._edge_to[x]
        self._leafs[self._source].update({v})

    def get_leaf_no(self, v):
        return len(self._leafs[v])

    def get_leafs(self, v):
        return self._leafs[v]

    def has_path_to(self, v):
        return self._marked[v]

    def get_path_to(self, v):
        if not self.has_path_to(v):
            return None
        path = []
        x = v
        while x != self._source:
            path.append(x)
            x = self._edge_to[x]
        path.append(self._source)
        return path


def dfs(tree, v, marked, superclasses, min_leafs=7, max_leafs=11):
    """ Walks the tree in depth and marks the nodes that can be used as
    superclasses based on the number of leafs (classes) they have.
    """
    marked[v] = True

    leaf_cnt = tree.get_leaf_no(v)
    children = tree._digraph.get_adj(v)
    clc = [tree.get_leaf_no(c) for c in children]

    # decide to pick the node as a superclass
    if min_leafs <= leaf_cnt <= max_leafs:
        superclasses.append(v)
        return
    elif (leaf_cnt > max_leafs) and all(map(lambda x: x < min_leafs, clc)):
        # if the node has many leafs but the children can't pass the constraint
        # also pick the node
        superclasses.append(v)
        return

    # continue walking
    for child in children:
        if not marked[child]:
            dfs(tree, child, marked, superclasses, min_leafs, max_leafs)


def find_tasks(dfs_counts, tree, min_leafs, max_leafs):
    marked = defaultdict(bool)
    superclasses = []
    dfs(dfs_counts, tree.root, marked, superclasses, min_leafs, max_leafs)
    return superclasses


def dump_dataset(dfsl, k, cls_no, misc_no, task_no, superclasses):
    """ Dump the wnids for a given dataset.
        
        First line contains:
            - no of classes,
            - no of classes that didn't get into a superclass meeting the
            constraints
            - no of tasks

        Each additional line contains *all* the wnids that are part of a
        superclass. So not just N wnids where N is the no of tasks but all of
        them.
    """
    file_name = f"cl_t{task_no}_c{cls_no}.txt"
    with open(file_name, "w+") as f:
        f.write(",".join([str(x) for x in [cls_no, misc_no, task_no]]) + "\n")
        for supercls in superclasses:
            row = [supercls] + list(dfsl.get_leafs(supercls))
            f.write(",".join(row) + "\n")


def main():
    _, c2p = consolidate_parents(*get_hierarchy())
    wnids = [r.rstrip() for r in open("data/tiny-imagenet-200/wnids.txt", "r")]
    words_file = open("data/tiny-imagenet-200/words.txt", "r")
    wnid2words = {r[0]: r[1] for r in csv.reader(words_file, delimiter="\t")}
    # words2wnid = {r[1]: r[0] for r in csv.reader(words_file, delimiter="\t")}

    # c2p_tiny = {wnid: c2p[wnid] for wnid in wnids}
    all_parents = get_all_parents(wnids, c2p)
    # display_parents(all_parents, wnid2words)

    print("\nBuild the Imagenet tree...")
    tree = DirectedGraph()
    for child, ancestors in all_parents.items():
        for ancestor in ancestors:
            tree.add_edge(ancestor, child)
            child = ancestor

    tree.root = "n00001930"

    # print the no of leafs of each node
    # in case of TinyImagenet, that is the no of classes under each node
    print("\nLeaf set for each node: ")
    dfsl = DFSLeafCount(tree, tree.root)
    for node, leafs in sorted(dfsl._leafs.items(), key=lambda kv: len(kv[1])):
        print(f"node {wnid2words[node]} has {len(leafs)} leafs.")

    # look at the path from a give class to a node
    print(f"\nFrom `{wnid2words['n02504458']}` to `{wnid2words[tree.root]}`:")
    ancestors = dfsl.get_path_to("n02504458")  # african elephant to root
    for ancestor in ancestors:
        print(f"  -->  {wnid2words[ancestor]} [{ancestor}]")

    # Set some constraints, the minimum and the maximum no of leafs for which
    # a superclass is a valide pick.
    print("\nTry to get superclasses.")
    constraints = []
    for cmin in range(5, 17):
        for cadd in range(2, 16):
            constraints.append((cmin, cmin + cadd))

    # Get the maximum no of superclasses given the constraint.
    results = {}
    for constraint in constraints:
        # return the superclasses
        superclasses = find_tasks(dfsl, tree, *constraint)
        # count how many leafs (instances) each superclass has
        instance_no = [dfsl.get_leaf_no(sc) for sc in superclasses]
        # compute how many uncategorized classes are left
        leftovers = 200 - sum(instance_no)
        # compute how many miscelaneous or random classes we can create
        class_no, task_no = len(superclasses), min(instance_no)
        misc_no = leftovers // task_no
        results[constraint] = [class_no, misc_no, task_no, superclasses]

    results_ = sorted(results.items(), key=lambda kv: kv[1][2], reverse=True)
    for k, values in results_:
        res = "{:>2} + {:>2} classes, {:>2} tasks".format(*values[:3])
        cst = "min={:>2}, max={:>2}".format(*k)
        print(f"  {res}  |  {cst}.")

    print("\nPossible picks:")
    good_constraints = [(10, 22), (10, 12), (7, 9), (5, 7)]
    picked = {k: v for k, v in results.items() if k in good_constraints}
    for k, values in picked.items():
        res = "{:>2} classes + {:>2} leftovers, {:>2} tasks".format(*values[:3])
        cst = "min={:>2}, max={:>2}".format(*k)
        header = f"  {res}  |  {cst}."
        print(f"{'-' * len(header)}\n{header}\n{'-' * len(header)}")
        for i, supercls in enumerate(values[-1]):
            print(f"[{i:2d}][{supercls}] {wnid2words[supercls]}.")
            for leaf in dfsl.get_leafs(supercls):
                print(f"  |  [{leaf}] {wnid2words[leaf]}")

    for k, values in picked.items():
        dump_dataset(dfsl, k, *values)


def test():
    digraph = DirectedGraph()
    with open("./test/tree_test.conf", "r") as f:
        rows = csv.reader(f, delimiter=" ")
        V, E = next(rows), next(rows)
        for (v, w) in rows:
            digraph.add_edge(int(v), int(w))
    print(digraph)

    dfsl = DFSLeafCount(digraph, 0)
    print("Leaf count for each node: ")
    for node, leafs in dfsl._leafs.items():
        print(f"node {node} has {len(leafs)} leafs.")


if __name__ == "__main__":
    # test()
    main()
