import numpy as np

class Temporal_Walk(object):
    def __init__(self, learn_data, inv_relation_id, transition_distr):
        self.learn_data = learn_data
        self.inv_relation_id = inv_relation_id
        self.transition_distr = transition_distr
        self.neighbors = store_neighbors(learn_data) # 仅找训练集中sub的邻居 {0:[[0, 160, 711, 63],[0,127, 733, 68],……]
        self.edges = store_edges(learn_data) #在train(包含逆向)中找出每个关系连接的事件的历史 {0:[[1, 0, 711, 63],[63,0, 733, 68],……]

    def sample_start_edge(self, rel_idx, type = 'default'):
        rel_edges = self.edges[rel_idx]
        # if type == "Reflexive":
        #     rel_edges = rel_edges[rel_edges[:, 0] == rel_edges[:, 2]]
        #     if len(rel_edges) == 0:
        #         return -1
        start_edge = rel_edges[np.random.choice(len(rel_edges))]
        return start_edge

    def sample_next_edge(self, filtered_edges, cur_ts):
        if self.transition_distr == "unif":
            next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
        elif self.transition_distr == "exp":
            tss = filtered_edges[:, 3]
            prob = np.exp(tss - cur_ts)
            try:
                prob = prob / np.sum(prob)
                next_edge = filtered_edges[
                    np.random.choice(range(len(filtered_edges)), p=prob)
                ]
            except ValueError:
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
        return next_edge

    def transition_step(self, rel, cur_node, cur_ts, prev_edge, start_node, step, L, type = 'default'):
        next_edges = self.neighbors[cur_node]
        if step == 1: # TODO The next timestamp should be smaller than the current timestamp，第一条因为是小于，则不会逆向
            # filtered_edges = next_edges[next_edges[:, 3] < cur_ts]
            # --------
            filtered_edges = next_edges[next_edges[:, 3] != cur_ts]
            # --------
        else: # The next timestamp should be smaller than or equal to the current timestamp
            # filtered_edges = next_edges[next_edges[:, 3] <= cur_ts]
            # --------
            filtered_edges = next_edges[next_edges[:, 3] != cur_ts]
            # --------
            #   TODO Delete inverse edge  不能返回pre前面的一条边
            inv_edge = [
                cur_node,
                self.inv_relation_id[prev_edge[1]],
                prev_edge[0],
                cur_ts,
            ]
            row_idx = np.where(np.all(filtered_edges == inv_edge, axis=1))
            filtered_edges = np.delete(filtered_edges, row_idx, axis=0)

        if type == "Symmetry":
            filtered_edges = filtered_edges[filtered_edges[:, 1] == rel]
        if type == "Inverse" or type == "Equivalent":
            filtered_edges = filtered_edges[filtered_edges[:, 1] != rel]
        if step == L - 1:# Find an edge that connects to the source of the walk
            # TODO 这是反向去预测，规则最后一个三元组尾必须是start_node
            filtered_edges = filtered_edges[filtered_edges[:, 2] == start_node]

        if len(filtered_edges):
            next_edge = self.sample_next_edge(filtered_edges, cur_ts) #根据概率，选择一个事件作为下一跳
        else:
            next_edge = []
        return next_edge


    def sample_walk_symmetry(self, rel_idx, L = 2):
        walk_successful = True
        walk = dict()
        prev_edge = self.sample_start_edge(rel_idx)
        start_node = prev_edge[0]
        cur_node = prev_edge[2]
        cur_ts = prev_edge[3]
        walk["entities"] = [start_node, cur_node]
        walk["relations"] = [prev_edge[1]]
        walk["timestamps"] = [cur_ts]

        walk["type"] = "Symmetry"
        for step in range(1, L):
            next_edge = self.transition_step(
                rel_idx, cur_node, cur_ts, prev_edge, start_node, step, L, "Symmetry"
            )
            if len(next_edge):
                cur_node = next_edge[2]
                cur_ts = next_edge[3]
                walk["relations"].append(next_edge[1])
                walk["entities"].append(cur_node)
                walk["timestamps"].append(cur_ts)
                prev_edge = next_edge
            else:
                walk_successful = False
                break

        return walk_successful, walk

    def sample_walk_inverse(self, rel_idx, L = 2):
        walk_successful = True
        walk = dict()
        prev_edge = self.sample_start_edge(rel_idx)
        start_node = prev_edge[0]
        cur_node = prev_edge[2]
        cur_ts = prev_edge[3]
        walk["entities"] = [start_node, cur_node]
        walk["relations"] = [prev_edge[1]]
        walk["timestamps"] = [cur_ts]

        walk["type"] = "Inverse"
        for step in range(1, L):
            next_edge = self.transition_step(
                rel_idx, cur_node, cur_ts, prev_edge, start_node, step, L, 'Inverse'
            )
            if len(next_edge):
                cur_node = next_edge[2]
                cur_ts = next_edge[3]
                walk["relations"].append(next_edge[1])
                walk["entities"].append(cur_node)
                walk["timestamps"].append(cur_ts)
                prev_edge = next_edge
            else:
                walk_successful = False
                break
        return walk_successful, walk

    def sample_walk_equivalent(self, rel_idx, L = 2):
        walk_successful = True
        walk = dict()
        prev_edge = self.sample_start_edge(rel_idx)
        start_node = prev_edge[0]
        cur_node = prev_edge[2]
        cur_ts = prev_edge[3]
        walk["entities"] = [start_node, start_node] #----
        walk["relations"] = [prev_edge[1]]
        walk["timestamps"] = [cur_ts]

        walk["type"] = "Equivalent"
        for step in range(1, L):
            next_edge = self.transition_step(
                rel_idx, start_node, cur_ts, prev_edge, cur_node, step, L, 'Equivalent'
            ) #----
            if len(next_edge):
                cur_node = next_edge[2]
                cur_ts = next_edge[3]
                walk["relations"].append(next_edge[1])
                walk["entities"].append(cur_node)
                walk["timestamps"].append(cur_ts)
                prev_edge = next_edge
            else:
                walk_successful = False
                break

        return walk_successful, walk

    def sample_walk_transitive(self, L, rel_idx):
        walk_successful = True
        walk = dict()
        prev_edge = self.sample_start_edge(rel_idx)
        start_node = prev_edge[0]
        cur_node = prev_edge[2]
        cur_ts = prev_edge[3]
        walk["entities"] = [start_node, cur_node]
        walk["relations"] = [prev_edge[1]]
        walk["timestamps"] = [cur_ts]

        walk["type"] = "Transitive" + str(L-1)
        for step in range(1, L):
            next_edge = self.transition_step(
                rel_idx, cur_node, cur_ts, prev_edge, start_node, step, L, 'Transitive'
            )
            if len(next_edge):
                cur_node = next_edge[2]
                cur_ts = next_edge[3]
                walk["relations"].append(next_edge[1])
                walk["entities"].append(cur_node)
                walk["timestamps"].append(cur_ts)
                prev_edge = next_edge
            else:
                walk_successful = False
                break
        return walk_successful, walk


    def sample_walk(self, L, rel_idx, type, Chain_type = 'tail'):
        if type == 'Symmetry':
            return self.sample_walk_symmetry(rel_idx)
        elif type == "Inverse":
            return self.sample_walk_inverse(rel_idx)
        elif type == "Equivalent":
            return self.sample_walk_equivalent(rel_idx)
        elif type == "Transitive":
            return self.sample_walk_transitive(L, rel_idx)

def store_neighbors(quads):
    neighbors = dict()
    nodes = list(set(quads[:, 0]))
    for node in nodes:
        neighbors[node] = quads[quads[:, 0] == node]
    return neighbors

def store_edges(quads):
    edges = dict()
    relations = list(set(quads[:, 1]))
    for rel in relations:
        edges[rel] = quads[quads[:, 1] == rel]
    return edges

