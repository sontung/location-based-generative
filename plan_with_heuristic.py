from data_loader import PBW_Planning_only
from PIL import Image
from planner import action_model, visualize_plan
import pickle
import random

def null_stack_cost(state_):
    return 0

def stack_cost(state_):
    score_ = 0
    weak_link = "blue"
    done = False
    while not done:
        done = True
        for rel in state_:
            if rel[1] == "up" and rel[2] == weak_link:
                score_ += 1
                weak_link = rel[0]
                done = False
    return score_

def rec_path(came_from, current):
    total_path = [current]
    cost = stack_cost(current)
    while hash_sg(current) in came_from:
        current = came_from[hash_sg(current)]
        total_path.insert(0, current)
        cost += stack_cost(current)

    return total_path, cost

def hash_sg(relationships, ob_names=('brown', 'purple', 'cyan', 'blue', 'red', 'green', 'gray')):
    """
    hash into unique ID
    :param relationships: [['brown', 'left', 'purple'] , ['yellow', 'up', 'yellow']]
    :param ob_names:
    :return:
    """
    a_key = [0]*len(ob_names)*len(ob_names)
    pred2id = {"none": 0, "left": 1, "up": 2}
    predefined_objects1 = ob_names[:]
    predefined_objects2 = ob_names[:]
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

def possible_next_states(current_state):
    predefined_actions = ["12", "13", "21", "23", "31", "32"]
    res = []
    for action in predefined_actions:
        next_state = action_model(current_state, action)
        if next_state is not None:
            res.append((next_state, action))
    return res

def a_star_search(start_, goal_, stack_cost_func):
    open_set = [start_]
    came_from = {}

    g_score = {hash_sg(start_): 0}
    f_score = {hash_sg(start_): stack_cost(start_)}

    while len(open_set) > 0:
        current = min(open_set, key=lambda x_: f_score[hash_sg(x_)])
        if hash_sg(current) == hash_sg(goal_):
            return rec_path(came_from, current)
        open_set.remove(current)

        for neighbor, act in possible_next_states(current):
            tentative_g_score = g_score[hash_sg(current)]+1
            if hash_sg(neighbor) not in g_score or tentative_g_score < g_score[hash_sg(neighbor)]:
                came_from[hash_sg(neighbor)] = current
                g_score[hash_sg(neighbor)] = tentative_g_score
                f_score[hash_sg(neighbor)] = g_score[hash_sg(neighbor)] + stack_cost_func(neighbor)
                if neighbor not in open_set:
                    open_set.append(neighbor)
    return None

# _ = PBW_Planning_only("/home/sontung/thesis/photorealistic-blocksworld/blocks-4-3", nb_samples=-1, base=True)
with open("data/blocks-4-3-360-planning", 'rb') as f:
    data = pickle.load(f)
sg2im = {}
for k in data:
    k2 = hash_sg(data[k][1])
    assert k2 not in sg2im
    sg2im[k2] = k.replace("json", "png").replace("scene", "image")

while True:
    start1 = random.choice(list(data.keys()))
    start2 = random.choice(list(data.keys()))

    # start1 = "/home/sontung/thesis/photorealistic-blocksworld/blocks-4-3/scene/CLEVR_new_000243.json"
    # start2 = "/home/sontung/thesis/photorealistic-blocksworld/blocks-4-3/scene/CLEVR_new_000233.json"

    # print("start:", data[start1][1], "stack cost:", stack_cost(data[start1][1]))
    # print("end:", data[start2][1], "stack cost:", stack_cost(data[start2][1]))

    path, stability_cost_null = a_star_search(data[start1][1], data[start2][1], null_stack_cost)
    visual_plan = [sg2im[hash_sg(p)] for p in path]
    visualize_plan(visual_plan, name="null")

    path, stability_cost = a_star_search(data[start1][1], data[start2][1], stack_cost)
    visual_plan = [sg2im[hash_sg(p)] for p in path]
    visualize_plan(visual_plan)


    if stability_cost == 0 and stability_cost_null > stability_cost:
        print(start1, start2)
        print("Cost when using null", stability_cost_null)
        print("Cost when using heuristic:", stability_cost)
        print()


# Image.open(start1.replace("json", "png").replace("scene", "image")).show()
# Image.open(start2.replace("json", "png").replace("scene", "image")).show()
# Image.open("solution.png").show()