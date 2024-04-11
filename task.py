import sys
import os

import gymnasium as gym
import numpy as np
import functools
import pickle
from multiprocessing import Pool
from network import NHNN
import json

from optimizer import ES2


def eval(data, render=False):
    args = data[1]
    cumulative_rewards = []
    task = gym.make("AntBulletEnv-v0")
    nodes = []
    nodes.append(28)
    nodes.append(128)
    nodes.append(64)
    nodes.append(8)

    agent = NHNN(nodes,0)
    x = data[0]
    agent.set_eta(x[:sum(nodes)])
    agent.set_hrules(x[sum(nodes):])
    obs = task.reset()
    start = time.time()
    cumulative_rewards.append(0)
    done = False
    # exit(1)

    neg_count = 0
    rew_ep = 0
    t = 0
    while not done:
        a =torch.tensor(obs)

        output = agent.forward(a)
        agent.update_weights()
        if render:
            task.render(mode="human")
        b =output.tolist()
        obs, _, done, info = task.step(b)

        rew = task.unwrapped.rewards[1]
        rew_ep += rew

        if t > 200:
            neg_count = neg_count + 1 if rew < 0.0 else 0
            if (neg_count > 30):
                done = True
        t+=1
    return rew_ep


def generator(random, args):
    return np.asarray([random.uniform(args["pop_init_range"][0],
                                      args["pop_init_range"][1])
                       for _ in range(args["num_vars"])])


def generator_wrapper(func):
    @functools.wraps(func)
    def _generator(random, args):
        return np.asarray(func(random, args))

    return _generator


def parallel_val(candidates, args):
    with Pool() as p:
        return p.map(eval, [[c, json.loads(json.dumps(args))] for c in candidates])
    # res = [eval([c, json.loads(json.dumps(args))]) for c in candidates]
    # return res


def experiment_launcher(config):
    seed = config["seed"]

    print(config)
    fka = NHNN(config["nodes"], 0, 0)
    args = config
    args["generations"] = 1
    args["num_vars"] = fka.nparams  # Number of dimensions of the search space
    print("this problem has " + str(args["num_vars"]) + " parameters")
    args["seed"] = seed

    es = ES1(seed, args["num_vars"], 4, 0.35)

    gen = 0
    logs = []
    while gen <= args["generations"]:
        candidates = es.ask()  # get list of new solutions
        fitnesses = parallel_val(candidates, args)
        log = "generation " + str(gen) + "  "+str(es.elite[0] if es.elite is not None else None)+"  " + str(max(fitnesses)) + "  " + str(np.mean(fitnesses))

        print(log)
        with open(os.path.join(args["dir"], "best_" + str(gen) + ".pkl"), "wb") as f:
            pickle.dump(candidates[np.argmax(fitnesses)], f)

        with open(os.path.join(args["dir"], "tlog.txt"), "a") as f:
            f.write(log + "\n")

        logs.append(log)

        es.tell(candidates, fitnesses)
        gen += 1

    best_guy = es.elite[1]
    best_fitness = es.elite[0]

    with open(os.path.join(args["dir"], str(best_fitness) + ".pkl"), "wb") as f:
        pickle.dump(best_guy, f)

    with open(os.path.join(args["dir"], "log.txt"), "w") as f:
        for l in logs:
            f.write(l + "\n")


def chs(dir):
    return os.path.exists(os.path.join(dir, "log.txt"))


if __name__ == "__main__":
    seed = int(sys.argv[1])

    args = {"seed": seed}
    args["dir"] = "res_NHNN_"+str(seed)
    os.makedirs(args["dir"])
    args["nodes"] = [27, 128,64, 8]
    experiment_launcher(args)
    print("ended experiment " + str(args))
