from FullGraph import AdjacencyGraph
import time
import math
import sumolib
import traci
import traci.constants
import os
import sys
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare env var 'SUMO_HOME'")

xml_begin = '<?xml version="1.0" encoding="UTF-8"?>\n'
edges_opening_tag = '<edges version="1.3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">\n'
edges_closing_tag = '</edges>'


SOURCE_NET_FILE = "KH_orig.net.xml"
TARGET_PREFIX = "KH"
TARGET_NET_FILE = TARGET_PREFIX + ".net.xml"
EDGES_FILE = TARGET_PREFIX + ".edg.xml"
NODES_FILE = TARGET_PREFIX + ".nod.xml"
ROUTES_FILE = TARGET_PREFIX + ".rou.xml"
CONFIG_FILE = "KH.sumocfg"
N_TRIPS = 500
BEST_EDGES_FILE = TARGET_PREFIX + "_best.edg.xml"


def decompose_net_file(source_file, target_prefix):
    command = ["netconvert", "--sumo-net",
               source_file, "--plain-output-prefix", target_prefix]
    os.system(" ".join(command))


def edge_xml(id, n_from, n_to, numlanes, speed, priority=-1):
    edg_x = "\t<edge"
    end = "/>\n"
    quote = "\""
    id_x = " id=\""
    from_x = " from=\""
    to_x = " to=\""
    priority_x = " priority=\""
    numlanes_x = " numLanes=\""
    speed_x = " speed=\""
    edg_x = edg_x + id_x + id + quote + from_x + \
        n_from + quote + to_x + n_to + quote
    edg_x = edg_x + priority_x + str(priority) + quote + numlanes_x + str(
        numlanes) + quote + speed_x + str(speed) + quote + end
    return edg_x


def simulate_config(config_file):
    sumocmd = ["sumo", "-c", config_file, "--start"]
    traci.start(sumocmd)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    traci.close()


class SumoGraph(AdjacencyGraph):
    def copy_graph(self, g):
        for i in range(len(g.edges)):
            self.edges[i] = g.edges[i]
        for i in range(len(g.vertices)):
            for j in range(len(g.vertices)):
                self.vertices[i][j] = g.vertices[i][j]

    def flip_edge(self, index):
        if self.edges[index]:
            self.edges[index] = False
            self.vertices[edges[index][0]][edges[index][1]] = 0
            if self.vertices[edges[index][1]][edges[index][0]] > 0:
                self.vertices[edges[index][1]
                              ][edges[index][0]] += edges[index][2]
        else:
            self.edges[index] = True
            self.vertices[edges[index][0]][edges[index][1]] = edges[index][2]
            if self.vertices[edges[index][1]][edges[index][0]] > 0:
                self.vertices[edges[index][1]
                              ][edges[index][0]] -= edges[index][2]
            else:
                self.vertices[edges[index][0]][edges[index][1]
                                               ] += orig.vertices[edges[index][1]][edges[index][0]]

    def save_file(self, loc):
        f = open(loc, "w")
        f.write(xml_begin)
        f.write(edges_opening_tag)
        for i, e in enumerate(edges_entities):
            if self.edges[i]:
                f.write(edge_xml(e.getID(), e.getFromNode().getID(), e.getToNode().getID(
                ), self.vertices[edges[i][0]][edges[i][1]], e.getSpeed(), e.getPriority()))
        f.write(edges_closing_tag)
        f.close()

    def gen_net_file(self, node_file, edg_file, loc):
        ns = "--node-files=" + node_file
        es = "--edge-files=" + edg_file
        out_file = "--output-file=" + loc
        netcmd = ["netconvert", ns, es, out_file]
        os.system(" ".join(netcmd))

    def gen_rand_trips(self, net_file, loc, end_time):
        randroute = ["python", "c:/\"Program Files (x86)\"/Eclipse/Sumo/tools/randomTrips.py",
                     "-n", net_file, "-r", loc, "-e", str(end_time)]
        os.system(" ".join(randroute))

    def simulate(self, config_file):
        sumocmd = ["sumo", "-c", config_file, "--start", "-W"]
        traci.start(sumocmd)
        total_speed = []
        total_halt = []
        number = []
        for e in edges_entities:
            total_halt.append(-1)
            total_speed.append(-1)
            number.append(-1)
        current_edges = []
        for edg_id in traci.edge.getIDList():
            if edg_id not in edges_dict.keys():
                continue
            current_edges.append(edg_id)
            total_halt[edges_dict[edg_id]] = 0
            total_speed[edges_dict[edg_id]] = 0
            number[edges_dict[edg_id]] = 0
        print("\nStarting simulation...")
        while traci.simulation.getMinExpectedNumber() > 0:
            for edg_id in current_edges:
                if traci.edge.getLastStepVehicleNumber(edg_id) > 0:
                    number[edges_dict[edg_id]] += 1
                    total_speed[edges_dict[edg_id]
                                ] += traci.edge.getLastStepMeanSpeed(edg_id)
                    total_halt[edges_dict[edg_id]
                               ] += traci.edge.getLastStepHaltingNumber(edg_id)
            traci.simulationStep()
        traci.close()
        reward = 0
        inactive_edgs = 0
        min_number = math.inf
        min_index = -1
        max_number = 0
        max_index = -1
        min_speed = math.inf
        min_sp_index = -1
        max_halt = 0
        max_halt_index = -1
        for i in range(len(number)):
            if number[i] != -1:
                if number[i] < min_number:
                    min_index = i
                    min_number = number[i]

                if number[i] > max_number:
                    max_index = i
                    max_number = number[i]

            if number[i] > 0:
                total_speed[i] = total_speed[i]/number[i]
                total_halt[i] = total_halt[i]/number[i]

                if total_speed[i] < min_speed:
                    min_sp_index = i
                    min_speed = total_speed[i]

                if total_halt[i] > max_halt:
                    max_halt = total_halt[i]
                    max_halt_index = i

                reward += total_speed[i] - total_halt[i]
        return reward, [min_index, max_index, min_sp_index, max_halt_index]

    def create_and_simulate(self, edgf, nodf, rouf, netf, cfgf, edt):
        self.save_file(edgf)
        self.gen_net_file(nodf, edgf, netf)
        self.gen_rand_trips(netf, rouf, edt)
        return self.simulate(cfgf)

    def make_action(self, action, edgs, orig_graph):
        if action < 4:
            self.flip_edge(edgs[action])
        elif action == 4:
            e = random.randint(0, len(self.edges)-1)
            self.flip_edge(e)
        elif action == 5:
            self.copy_graph(best)
        else:
            print("INVALID ACTION")


decompose_net_file(SOURCE_NET_FILE, TARGET_PREFIX)

net = sumolib.net.readNet(SOURCE_NET_FILE)
nodes = net.getNodes()
nodes = [n.getID() for n in nodes]
nodes_dict = {}
for i, n in enumerate(nodes):
    nodes_dict[n] = i

edges_entities = net.getEdges(False)
edges = []
edges_dict = {}
for i, e in enumerate(edges_entities):
    edges_dict[e.getID()] = i
    edges.append([nodes_dict[e.getFromNode().getID()],
                  nodes_dict[e.getToNode().getID()], e.getLaneNumber()])

# print(*edges)
orig = SumoGraph(edges, len(nodes), True)
best = SumoGraph(edges, len(nodes), True)
best_reward = 0
s = SumoGraph(edges, len(nodes), True)


class DQN(nn.Module):
    def __init__(self, n_vertices):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(in_features=n_vertices *
                             n_vertices, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=6)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


Experience = namedtuple(
    "Experience", ("state", "action", "next_state", "reward"))


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)


class Agent():
    def __init__(self, strategy, num_actions):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action])
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1)


class QValues():
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        values = target_net(next_states).max(dim=1)[0].detach()
        return values


def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(values)

    moving_avg = get_moving_avg(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\n", moving_avg_period,
          "episode moving avg:", moving_avg[-1])
    if is_ipython:
        display.clear_output(wait=True)


def get_moving_avg(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(
            dim=1).flatten(start_dim=0)
    else:
        moving_avg = torch.zeros_like(values)
    return moving_avg.numpy()


def extract_tensors(exps):
    batch = Experience(*zip(*exps))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.next_state)
    t4 = torch.cat(batch.reward)

    return (t1, t2, t3, t4)


batch_size = 256
gamma = 0.7
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100
lr = 0.001
num_episodes = 10_000

strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, 6)
memory = ReplayMemory(memory_size)

policy_net = DQN(len(nodes))
target_net = DQN(len(nodes))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_durations = []
for episode in range(num_episodes):
    timer = time.time()
    print("\n\nEPISODE:", episode)
    state = torch.tensor([s.vertices], dtype=torch.float)
    r1, action_edgs = s.create_and_simulate(EDGES_FILE, NODES_FILE, ROUTES_FILE,
                                            TARGET_NET_FILE, CONFIG_FILE, N_TRIPS)
    if r1 > best_reward and len(s.strongly_connected()) == 1:
        best_reward = r1
        best.copy_graph(s)

    action = agent.select_action(state, policy_net)
    s.make_action(action, action_edgs, orig)
    r2, action_edgs = s.create_and_simulate(EDGES_FILE, NODES_FILE, ROUTES_FILE,
                                            TARGET_NET_FILE, CONFIG_FILE, N_TRIPS)
    if r2 > best_reward and len(s.strongly_connected()) == 1:
        best_reward = r2
        best.copy_graph(s)

    reward = torch.tensor([r2 - r1], dtype=torch.float)
    next_state = torch.tensor([s.vertices], dtype=torch.float)
    memory.push(Experience(state, action, next_state, reward))
    state = next_state

    if memory.can_provide_sample(batch_size):
        exps = memory.sample(batch_size)
        states, actions, next_states, rewards = extract_tensors(exps)
        current_q_values = QValues.get_current(policy_net, states, actions)
        next_q_values = QValues.get_next(target_net, next_states)
        target_q_values = (next_q_values * gamma) + rewards

        optimizer.zero_grad()
        loss = F.smooth_l1_loss(
            current_q_values, target_q_values.unsqueeze(1))
        loss.backward()
        optimizer.step()

    best.save_file(BEST_EDGES_FILE)
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print("EPISODE DONE IN {} SECONDS \n\n".format(time.time() - timer))
