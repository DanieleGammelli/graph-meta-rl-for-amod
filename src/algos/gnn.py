"""
A2C-GNN
-------
This file contains the A2C-GNN specifications. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy represented by a Temporal Graph Network (Section 4.3 in the paper)
(3) GNNCritic:
    Critic represented by a Temporal Graph Network (Section 4.3 in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""

import numpy as np
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.utils import grid
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render= True
args.gamma = 0.99
args.log_interval = 10

#########################################
############## A2C PARSER ###############
#########################################

class GNNParser():
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """
    def __init__(self, env, T=10, scale_factor=0.01, json_file=None):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.json_file = json_file
        if self.json_file is not None:
            with open(json_file,"r") as file:
                self.data = json.load(file)
        
    def parse_obs(self, obs, a_t_1, r1_t_1, r2_t_1, d_t_1):
        # Takes input from the environemnt and returns a graph (with node features and connectivity)
        # Here we aggregate environment observations into node-wise features
        # In order, x is a collection of the following information:
        # 1) current availability, 2) Estimated availability, 3) one-hot representation of time-of-day
        # 4) current demand, 5) RL2 info (i.e., a_{t-1}, r_{t-1} (both matching and rebalancing), d_{t-1}
        x = torch.cat((
            torch.tensor([obs[0][n][self.env.time+1] for n in self.env.region]).view(1, 1, self.env.nregion).float(),
            torch.tensor([[(obs[0][n][self.env.time+1] + self.env.dacc[n][t]) for n in self.env.region] \
                          for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.nregion).float(),
        torch.nn.functional.one_hot(torch.tensor([self.env.time]*self.env.nregion), num_classes=self.env.tf).view(1, self.env.tf, self.env.nregion).float(),
            torch.tensor([sum([(self.env.demand[i,j][self.env.time+1]) \
                          for j in self.env.region]) for i in self.env.region]).view(1, 1, self.env.nregion).float(),
            a_t_1.view(1, 1, self.env.nregion).float(),
            r1_t_1.view(1, 1, self.env.nregion).float(),
            r2_t_1.view(1, 1, self.env.nregion).float(),
            d_t_1.view(1, 1, self.env.nregion).float()),
              dim=1).squeeze(0).view(36, self.env.nregion).T
        if self.json_file is not None:
            edge_index = torch.vstack((torch.tensor([edge['i'] for edge in self.data["topology_graph"]]).view(1,-1), 
                                      torch.tensor([edge['j'] for edge in self.data["topology_graph"]]).view(1,-1))).long()
        else:
            edge_index = torch.cat((torch.arange(self.env.nregion).view(1, self.env.nregion), 
                                    torch.arange(self.env.nregion).view(1, self.env.nregion)), dim=0).long()
        data = Data(x, edge_index)
        return data

#########################################
############## A2C ACTOR ################
#########################################

class Actor(torch.nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """
    def __init__(self, input_size=4, hidden_dim=256, output_dim=1):
        super(Actor, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lin1 = nn.Linear(input_size, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.h_to_g = nn.Linear(hidden_dim, hidden_dim)
        self.g_to_a = nn.Linear(input_size + hidden_dim, output_dim)

    def forward(self, x, edge_index, h):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # h: GRU hiddent state
        row, col = edge_index
        x_temp = torch.cat([x[row]])
        h_temp = h

        x_temp = F.relu(self.lin1(x_temp))
        x_temp = scatter_sum(x_temp, col, dim=0, dim_size=x.size(0))

        h = self.gru(x_temp, h_temp)

        x_temp = F.relu(self.h_to_g(h))
        x_temp = torch.cat([x, x_temp], dim=1)

        a = F.softplus(self.g_to_a(x_temp))
        return a, h

#########################################
############## A2C CRITIC ###############
#########################################

class Critic(torch.nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """
    def __init__(self, input_size=4, hidden_dim=256, output_dim=1):
        super(Critic, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lin1 = nn.Linear(input_size, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.h_to_g = nn.Linear(hidden_dim, hidden_dim)
        self.g_to_a = nn.Linear(input_size + hidden_dim, output_dim)

    def forward(self, x, edge_index, h):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # h: GRU hiddent state
        row, col = edge_index
        x_temp = torch.cat([x[row]])
        h_temp = h

        x_temp = F.relu(self.lin1(x_temp))
        x_temp = scatter_sum(x_temp, col, dim=0, dim_size=x.size(0))

        h = self.gru(x_temp, h_temp)

        x_temp = F.relu(self.h_to_g(h))

        x_temp = torch.cat([x, x_temp], dim=1)
        x_temp = torch.sum(x_temp, dim=0)

        v = self.g_to_a(x_temp)
        return v, h

#########################################
############## A2C AGENT ################
#########################################

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    def __init__(self, env, input_size, hidden_size=128, eps=np.finfo(np.float32).eps.item(), device=torch.device("cpu"),
                clip=50, env_baseline=None):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.clip = clip if clip is not None else -1
        
        self.actor = Actor(self.input_size, self.hidden_size, 1)
        self.critic = Critic(self.input_size, self.hidden_size, 1)
        
        self.optimizers = self.configure_optimizers()
        
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.env_baseline = env_baseline
        self.to(self.device)
        
    def forward(self, obs, h_a, h_c, jitter=1e-20):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        graph = obs.to(self.device)
        
        # actor: computes concentration parameters of a X distribution
        a_probs, h_a = self.actor(graph.x, graph.edge_index, h_a)

        # critic: estimates V(s_t)
        value, h_c = self.critic(graph.x, graph.edge_index, h_c)
        return a_probs + jitter, value, h_a, h_c
    
    def select_action(self, obs, h_a, h_c):
        a_probs, value, h_a, h_c = self.forward(obs, h_a, h_c)
        
        m = Dirichlet(concentration=a_probs.view(-1,))
        
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), value))
        return action, h_a, h_c

    def training_step(self, city):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            r = float((r - np.mean(self.env_baseline[city])) / (np.std(self.env_baseline[city]) + self.eps)) # env-dependent baseline to standardinze rewards
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).float()
        
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        if self.clip >= 0:
            a_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip)
        self.optimizers['a_optimizer'].step()
        
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        if self.clip >= 0:
            v_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip)
        self.optimizers['c_optimizer'].step()
        
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        if self.clip >= 0:
            return {"a_grad_norm": a_grad_norm, "v_grad_norm": a_grad_norm}
    
    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers['a_optimizer'] = torch.optim.Adam(actor_params, lr=3e-4)
        optimizers['c_optimizer'] = torch.optim.Adam(critic_params, lr=3e-4)
        return optimizers
    
    def save_checkpoint(self, path='ckpt.pth'):
        checkpoint = dict()
        checkpoint['model'] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path='ckpt.pth'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])
    
    def log(self, log_dict, path='log.pth'):
        torch.save(log_dict, path)