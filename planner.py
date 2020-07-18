import utils
from queue import Queue


class Node:
    def __init__(self, sg):
        self.sg = sg
        self.key = hash_sg(sg)
        self.visited = False
        self.parent = None
        self.act = None

    def get_key(self):
        return self.key

    def get_sg(self):
        return self.sg

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.key == other.key


def find_top(up_rel, ob_start):
    rel_res = None
    while True:
        done = True
        for rel in up_rel:
            if rel[2] == ob_start:
                ob_start = rel[0]
                done = False
                rel_res = rel
                break
        if done:
            return ob_start, rel_res


def action_model(sg_from, action):
    assert action in ["12", "13", "21", "23", "31", "33"]
    base_objs = ['brown', 'purple', 'cyan']
    relations_from = sg_from[:]
    block_from = int(action[0])-1
    block_to = int(action[1])-1

    # check valid action
    up_rel = [rel for rel in relations_from if rel[1] == "up"]
    ob_to = [rel[2] for rel in up_rel]
    if base_objs[block_from] not in ob_to:
        return None

    # modify "from" block
    top_block_from, to_be_removed_rel = find_top(up_rel, base_objs[block_from])
    relations_from.remove(to_be_removed_rel)

    # modify "to" block
    top_block_to, _ = find_top(up_rel, base_objs[block_to])
    relations_from.append([top_block_from, "up", top_block_to])

    return relations_from


def possible_next_states(current_state):
    predefined_actions = ["12", "13", "21", "23", "31", "33"]
    res = []
    for action in predefined_actions:
        next_state = action_model(current_state, action)
        if next_state is not None:
            res.append((next_state, action))
    return res


def hash_sg(relationships):
    a_key = [0]*64
    pred2id = {"none": 0, "left": 1, "up": 2}
    predefined_objects1 = ['brown', 'purple', 'cyan', 'red', 'green', 'yellow', 'blue', 'gray']
    predefined_objects2 = ['brown', 'purple', 'cyan', 'red', 'green', 'yellow', 'blue', 'gray']
    pair2pred = {}
    for rel in relationships:
        if rel[1] != "__in_image__":
            pair2pred[(rel[0], rel[2])] = pred2id[rel[1]]

    idx = 0
    for ob1 in predefined_objects1:
        for ob2 in predefined_objects2:
            if (ob1, ob2) in pair2pred:
                a_key[idx] = pair2pred[(ob1, ob2)]
            idx += 1
    return tuple(a_key)


def inv_hash_sg(a_key):
    predefined_objects1 = ['brown', 'purple', 'cyan', 'red', 'green', 'yellow', 'blue', 'gray']
    predefined_objects2 = ['brown', 'purple', 'cyan', 'red', 'green', 'yellow', 'blue', 'gray']
    id2pred = {0: "none", 1: "left", 2: "up"}

    sg = []
    idx = 0
    for ob1 in predefined_objects1:
        for ob2 in predefined_objects2:
            if a_key[idx] > 0:
                sg.append([ob1, id2pred[a_key[idx]], ob2])
            idx += 1
    return sg


def plan(start_sg, end_sg):
    # bfs
    Q = Queue()
    start_v = Node(start_sg)
    goal_state = Node(end_sg)
    all_nodes = {start_v.get_key(): start_v, goal_state.get_key(): goal_state}
    start_v.visited = True
    Q.put(start_v)
    while not Q.empty():
        v = Q.get()
        if v == goal_state:
            print("search is completed")
            break
        for w, act in possible_next_states(v.get_sg()):
            w_key = hash_sg(w)
            if w_key not in all_nodes:
                all_nodes[w_key] = Node(w)
            if not all_nodes[w_key].visited:
                all_nodes[w_key].visited = True
                all_nodes[w_key].parent = v
                all_nodes[w_key].act = act
                Q.put(all_nodes[w_key])

    traces = []
    actions = []
    while True:
        traces.insert(0, goal_state.get_key())
        actions.insert(0, goal_state.act)
        if goal_state == start_v:
            break
        goal_state = goal_state.parent
    return traces, actions


if __name__ == '__main__':
    sg_from = [['brown', 'left', 'purple'], ['purple', 'left', 'cyan'], ['gray', 'up', 'red'],
               ['red', 'up', 'brown'], ['blue', 'up', 'purple'], ['yellow', 'up', 'cyan']]
    sg_to = [['brown', 'left', 'purple'], ['purple', 'left', 'cyan'], ['blue', 'up', 'gray'],
             ['gray', 'up', 'yellow'], ['yellow', 'up', 'red'], ['red', 'up', 'brown']]
    # print(action_model(sg_from, "12"))
    tr, ac = plan(sg_from, sg_to)
    print(ac)
    for t in tr:
        print([du for du in inv_hash_sg(t) if du[1] != "left"])