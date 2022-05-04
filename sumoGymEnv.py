import os
import sys
import numpy as np
import gym
from gym import spaces
import traci
import sumolib
import math
import random
from copy import deepcopy

xml_begin = '<?xml version="1.0" encoding="UTF-8"?>\n'
edges_opening_tag = '<edges version="1.3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">\n'
edges_closing_tag = '</edges>'

class SumoGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_actions:int, n_rand_trips:int, source_path:str, config_path:str, target_prefix:str, n_episodes:int):
        super(SumoGymEnv, self).__init__()

        if "SUMO_HOME" in os.environ:
            self.tools = os.path.join(os.environ["SUMO_HOME"], 'tools')
            sys.path.append(self.tools)
        else:
            sys.exit("please declare env var 'SUMO_HOME'")
            
        self.n_episodes = n_episodes
        self.counter = 0

        net = sumolib.net.readNet(source_path)
        self.nodes = net.getNodes()
        self.nodes = [n.getID() for n in self.nodes]
        self.nodes_dict = {}
        for i, n in enumerate(self.nodes):
            self.nodes_dict[n] = i

        self.edges_entities = net.getEdges(False)
        self.edges_data = []
        self.edges_dict = {}
        for i, e in enumerate(self.edges_entities):
            self.edges_dict[e.getID()] = i
            self.edges_data.append([self.nodes_dict[e.getFromNode().getID()],
                        self.nodes_dict[e.getToNode().getID()], e.getLaneNumber()])

        self.edges = [True]*len(self.edges_data)
        self.vertices = []
        self.directed = True
        self.visited = [False]*len(self.nodes)

        for i in range(len(self.nodes)):
            self.vertices.append([])
            for j in range(len(self.nodes)):
                self.vertices[i].append(0)
        for e in self.edges_data:
            self.vertices[e[0]][e[1]] = e[2]


        self.init_graph = deepcopy(self.vertices)
        max_lanes = np.max(np.array(self.init_graph))
        self.n_rand_trips = n_rand_trips
        self.source_path = source_path
        self.config_path = config_path
        self.target_prefix = target_prefix
        self.edges_path = target_prefix + ".edg.xml"
        self.nodes_path = target_prefix + ".nod.xml"
        self.routes_path = target_prefix + ".rou.xml"
        self.net_path = target_prefix + ".net.xml"

        self.actions_array = [0, 0, 0, 0]
        self.reward = -math.inf
        self.best_reward = -math.inf
        self.best_graph = []
        self.copy_graph_best()

        self.action_space = spaces.Discrete(n_actions) # action space
        self.observation_space = spaces.Box(low=0, high=2*max_lanes,
                                            shape=(len(self.nodes), len(self.nodes), 1), dtype=np.uint8) # state space


    def copy_graph_best(self):
        self.best_graph = []
        for i in range(len(self.nodes)):
            self.best_graph.append([])
            for j in range(len(self.nodes)):
                self.best_graph[i].append(self.vertices[i][j])

    def copy_graph_from_best(self):
        self.vertices = []
        for i in range(len(self.nodes)):
            self.vertices.append([])
            for j in range(len(self.nodes)):
                self.vertices[i].append(self.best_graph[i][j])

    def _topo_sort(self, node=0, reverse=False):
        nodes=[node]
        index = 1

        self.visited[node] = True
        if reverse:
            for i in range(len(self.vertices)):
                if self.vertices[i][node] > 0:
                    next_node = i
                    if self.visited[next_node]:
                        continue
                    temp = self._topo_sort(next_node, reverse)
                    for i in range(len(temp)):
                        nodes.insert(index+i, temp[i])
        else:
            for i, e in enumerate(self.vertices[node]):
                if e > 0:
                    next_node = i
                    if self.visited[next_node]:
                        continue
                    temp = self._topo_sort(next_node)
                    for i in range(len(temp)):
                        nodes.insert(index+i, temp[i])

        return nodes

    def topo_sort(self, reverse=False):
        topo = []
        for i in range(len(self.vertices)):
            if self.visited[i]==False:
                temp = self._topo_sort(i, reverse)
                temp.extend(topo)
                topo = temp
        self.visited = [False for i in self.visited]
        return topo

    def strongly_connected(self):
        sort_topo = self.topo_sort(True)
        scc = []
        for i in sort_topo:
            if self.visited[i]:
                continue
            scc.append(self._topo_sort(i))
        self.visited = [False for i in self.visited]
        return scc

    def decompose_net_file(self):
        command = ["netconvert", "--sumo-net",
                self.source_path, "--plain-output-prefix", self.target_prefix]
        os.system(" ".join(command))


    def edge_xml(self, id, n_from, n_to, numlanes, speed, priority=-1):
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


    def simulate_config(self):
        sumocmd = ["sumo", "-c", self.config_file, "--start"]
        traci.start(sumocmd)
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
        traci.close()

    def flip_edge(self, index):
        if self.edges[index]:
            self.edges[index] = False
            self.vertices[self.edges_data[index][0]][self.edges_data[index][1]] = 0
            if self.vertices[self.edges_data[index][1]][self.edges_data[index][0]] > 0:
                self.vertices[self.edges_data[index][1]
                              ][self.edges_data[index][0]] += self.edges_data[index][2]
        else:
            self.edges[index] = True
            self.vertices[self.edges_data[index][0]][self.edges_data[index][1]] = self.edges_data[index][2]
            if self.vertices[self.edges_data[index][1]][self.edges_data[index][0]] > 0:
                self.vertices[self.edges_data[index][1]
                              ][self.edges_data[index][0]] -= self.edges_data[index][2]
            else:
                self.vertices[self.edges_data[index][0]][self.edges_data[index][1]
                                               ] += self.init_graph[self.edges_data[index][1]][self.edges_data[index][0]]

    def save_file(self):
        f = open(self.edges_path, "w")
        f.write(xml_begin)
        f.write(edges_opening_tag)
        for i, e in enumerate(self.edges_entities):
            if self.edges[i]:
                f.write(self.edge_xml(e.getID(), e.getFromNode().getID(), e.getToNode().getID(
                ), self.vertices[self.edges_data[i][0]][self.edges_data[i][1]], e.getSpeed(), e.getPriority()))
        f.write(edges_closing_tag)
        f.close()

    def gen_net_file(self):
        ns = "--node-files=" + self.nodes_path
        es = "--edge-files=" + self.edges_path
        out_file = "--output-file=" + self.net_path
        netcmd = ["netconvert", ns, es, out_file]
        os.system(" ".join(netcmd))

    def gen_rand_trips(self):
        randroute = ["python", "$SUMO_HOME/tools/randomTrips.py",
                     "-n", self.net_path, "-r", self.routes_path, "-e", str(self.n_rand_trips)]
        os.system(" ".join(randroute))

    def simulate(self):
        sumocmd = ["sumo", "-c", self.config_path, "--start", "-W"]
        traci.start(sumocmd)
        total_speed = []
        total_halt = []
        number = []
        for e in self.edges_entities:
            total_halt.append(-1)
            total_speed.append(-1)
            number.append(-1)
        current_edges = []
        for edg_id in traci.edge.getIDList():
            if edg_id not in self.edges_dict.keys():
                continue
            current_edges.append(edg_id)
            total_halt[self.edges_dict[edg_id]] = 0
            total_speed[self.edges_dict[edg_id]] = 0
            number[self.edges_dict[edg_id]] = 0
        print("\nStarting simulation...")
        while traci.simulation.getMinExpectedNumber() > 0:
            for edg_id in current_edges:
                if traci.edge.getLastStepVehicleNumber(edg_id) > 0:
                    number[self.edges_dict[edg_id]] += 1
                    total_speed[self.edges_dict[edg_id]
                                ] += traci.edge.getLastStepMeanSpeed(edg_id)
                    total_halt[self.edges_dict[edg_id]
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

    def create_and_simulate(self):
        self.save_file()
        self.gen_net_file()
        self.gen_rand_trips()
        self.reward, self.actions_array = self.simulate()
        return self.reward

    def make_action(self, action):
        if action < 4:
            self.flip_edge(self.actions_array[action])
        elif action == 4:
            e = random.randint(0, len(self.edges)-1)
            self.flip_edge(e)
        elif action == 5:
            self.copy_graph_from_best()
        else:
            print("INVALID ACTION")


    def step(self, action):
        self.counter += 1
        self.make_action(action)
        reward = self.create_and_simulate()
        observation = deepcopy(self.vertices)
        done = False if self.counter < self.n_episodes else True
        info= {}
        return observation, reward, done, info


    def reset(self):
        self.counter = 0
        self.vertices = deepcopy(self.init_graph)
        self.best_graph = deepcopy(self.init_graph)
        observation = deepcopy(self.init_graph)
        self.edges = [True]*len(self.edges_data)
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass 

    def close (self):
        pass


env = SumoGymEnv(6, 1500, "KH_orig.net.xml", "KH.sumocfg", "KH", 100)
env.step(4)