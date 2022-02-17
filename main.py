from __future__ import print_function
import argparse
import tqdm
from tqdm import trange
import numpy as np
import torch
from src.envs.amod_env import Scenario, AMoD
from src.algos.gnn import A2C, GNNParser
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum

# Define calibrated simulation parameters
demand_ratio = {'san_francisco': 2, 'washington_dc': 4.2, 'chicago': 1.8, 'nyc_man_north': 1.8, 'nyc_man_middle': 1.8,
                'nyc_man_south': 1.8, 'nyc_brooklyn': 9, 'porto': 4, 'rome': 1.8, 'shenzhen_baoan': 2.5,
                'shenzhen_downtown_west': 2.5, 'shenzhen_downtown_east': 3, 'shenzhen_north': 3
               }
json_hr = {'san_francisco':19, 'washington_dc': 19, 'chicago': 19, 'nyc_man_north': 19, 'nyc_man_middle': 19,
           'nyc_man_south': 19, 'nyc_brooklyn': 19, 'porto': 8, 'rome': 8, 'shenzhen_baoan': 8,
           'shenzhen_downtown_west': 8, 'shenzhen_downtown_east': 8, 'shenzhen_north': 8
          }
beta = {'san_francisco': 0.2, 'washington_dc': 0.5, 'chicago': 0.5, 'nyc_man_north': 0.5, 'nyc_man_middle': 0.5,
                'nyc_man_south': 0.5, 'nyc_brooklyn':0.5, 'porto': 0.1, 'rome': 0.1, 'shenzhen_baoan': 0.5,
                'shenzhen_downtown_west': 0.5, 'shenzhen_downtown_east': 0.5, 'shenzhen_north': 0.5}

test_tstep = {'san_francisco': 3, 'nyc_brooklyn': 4, 'shenzhen_downtown_west': 3}
jitter = 0.000001 # used for numerical stability of Dirichlet mean computation

parser = argparse.ArgumentParser(description='A2C-GNN')

# Simulator parameters
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--json_tstep', type=int, default=3, metavar='S',
                    help='minutes per timestep (default: 3min)')

# Model parameters
parser.add_argument('--test', type=bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--cplexpath', type=str, default='/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/',
                    help='defines directory of the CPLEX installation')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')
parser.add_argument('--max_trials', type=int, default=3000, metavar='N',
                    help='number of trails to train agent (default: 3k)')
parser.add_argument('--max_episodes', type=int, default=10, metavar='N',
                    help='number of episodes within each trial (default: 10)')
parser.add_argument('--max_steps', type=int, default=20, metavar='N',
                    help='number of steps per episode (default: T=20)')
parser.add_argument('--max_test_iter', type=int, default=10, metavar='N',
                    help='number of repeated experiments at test time')
parser.add_argument('--no-cuda', type=bool, default=True,
                    help='disables CUDA training')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of hidden states in neural nets')
parser.add_argument('--clip', type=int, default=50, metavar='N',
                    help='vector magnitude used for gradient clipping')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.demand_ratio = demand_ratio
args.json_hr = json_hr
args.beta = beta
args.test_tstep = test_tstep
device = torch.device("cuda" if args.cuda else "cpu")

# Initialize reward buffer to compute environment-dependent baseline (i.e., normalize rewards across cities)
train_env = [
 'washington_dc',
 'chicago',
 'nyc_man_north',
 'nyc_man_south',
 'nyc_man_middle',
 'porto',
 'shenzhen_baoan',
 'rome',
 'shenzhen_downtown_east',
 'shenzhen_north']
test_env = ['san_francisco',
 'shenzhen_downtown_west',
 'nyc_brooklyn']
env_baseline = dict()
for city in train_env + test_env:
    env_baseline[city] = [0.]
    
# Define AMoD Simulator Environment
city = np.random.choice(train_env) # initially sample a random city from the meta-training set
scenario = Scenario(json_file=f"data/scenario_{city}.json", demand_ratio=args.demand_ratio[city], json_hr=args.json_hr[city], sd=args.seed, json_tstep=args.json_tstep, tf=args.max_steps)
env = AMoD(scenario, beta=args.beta[city])
# Initialize A2C-GNN
agent = A2C(env=env, input_size=36, hidden_size=args.hidden_size, clip=args.clip, env_baseline=env_baseline).to(device)
parser = GNNParser(env, json_file=f"data/scenario_{city}.json")

if not args.test:
    #######################################
    #############Training Loop#############
    #######################################
    num_iter = args.max_trials #set max number of trials
    iterations = trange(num_iter) #trial iterator
    num_episodes = args.max_episodes #set number of episodes within one trial
    # initialize lists for book-keeping
    episode_reward_list = [] 
    avg_episode_reward = []
    task_reward_list = []
    a_grad_norms_list = []
    v_grad_norms_list = []
    for task in iterations: #initialize trial (i.e., MDP)
        task_reward = 0
        episode_reward_list = [0]
        # randomly sample a city from the meta-training set
        np.random.seed()
        city = np.random.choice(train_env)
        # initialize AMoD Simulator Environment
        scenario = Scenario(json_file=f"data/scenario_{city}.json", demand_ratio=args.demand_ratio[city], json_hr=args.json_hr[city], sd=None, json_tstep=args.json_tstep, tf=args.max_steps)
        env = AMoD(scenario, beta=beta[city])
        parser = GNNParser(env, json_file=f"data/scenario_{city}.json")
        # initialize placeholder values for RL2 
        #
        # we define reward (both from customer dispatching and rebalancing) and "done" signal at node level
        #
        a_t = torch.zeros((env.nregion, 1)).float() # previous action
        ext1_r_t = torch.zeros((env.nregion, 1)).float() # previous dispacthing reward for all nodes
        ext2_r_t = torch.zeros((env.nregion, 1)).float() # previous rebalancing reward for all nodes
        ext_d_t = [False]*env.nregion # previous done-signal for all nodes
        ext_d_t = torch.tensor(ext_d_t).view(env.nregion, 1)
        h_t_a = torch.zeros((env.nregion, agent.hidden_size)).float() # actor graph-GRU hidden state
        h_t_c = torch.zeros((env.nregion, agent.hidden_size)).float() # critic graph-GRU hidden state
        seed = np.random.randint(low=0, high=100000) # select a different random seed to allow consistency across episodes
        for episode in range(num_episodes):
            # initialize AMoD Simulator Environment (fixed across trial)
            np.random.seed(seed)
            scenario = Scenario(json_file=f"data/scenario_{city}.json", demand_ratio=args.demand_ratio[city], json_hr=args.json_hr[city], sd=None, json_tstep=args.json_tstep, tf=args.max_steps)
            env = AMoD(scenario, beta=beta[city])
            parser = GNNParser(env, json_file=f"data/scenario_{city}.json")
            d_t = False
            obs = env.reset() # reset environment
            # initialize episode-level book-keeping variables
            episode_reward = 0
            episode_served_demand = 0
            episode_rebalancing_cost = 0
            episode_ext_reward = np.zeros(env.nregion)
            episode_ext_paxreward = np.zeros(env.nregion)
            episode_ext_rebreward = np.zeros(env.nregion)
            while not d_t:
                # take matching step (Step 1 in paper)
                o_t, paxreward, d_t, info, ext_paxreward, ext_done = env.pax_step(CPLEXPATH=args.cplexpath, PATH='metarl-amod')
                episode_reward += paxreward
                episode_ext_paxreward += ext_paxreward
                episode_ext_reward += ext_paxreward
                ext1_r_t = torch.from_numpy(ext_paxreward).view(env.nregion, 1)
                # parse raw environment observations
                data = parser.parse_obs(o_t, a_t, ext1_r_t, ext2_r_t, ext_d_t)
                # use GNN-RL policy (Step 2 in paper)
                a_t, h_t_a, h_t_c = agent.select_action(data, h_t_a, h_t_c)
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {env.region[i]: int(a_t[i] *dictsum(env.acc,env.time+1))for i in range(len(env.region))}
                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveRebFlow(env=env,res_path='metarl-amod',desiredAcc=desiredAcc,CPLEXPATH=args.cplexpath)
                # take action in environment
                o_t, rebreward, d_t, info, ext_rebreward, ext_done = env.reb_step(rebAction)
                # book-keeping
                episode_reward += rebreward
                episode_ext_paxreward += ext_rebreward
                episode_ext_reward += ext_rebreward
                ext2_r_t = torch.from_numpy(ext_rebreward).view(env.nregion, 1) 
                ext_d_t = torch.from_numpy(np.array(ext_done)).view(env.nregion, 1)
                r_t = paxreward + rebreward
                task_reward += r_t
                episode_served_demand += info['served_demand']
                episode_rebalancing_cost += info['rebalancing_cost']
                a_t = a_t.view(env.nregion, 1).float() #ensure shape consistency of action
                # store the transition in memory
                agent.rewards.append(r_t) 
                # update the buffer to compute the environment dependent baseline (i.e., standardize rewards across cities)
                if len(agent.env_baseline[city]) <= 1000: # if len buffer <= max, then: append
                    agent.env_baseline[city].append(r_t)
                else: # else, replace oldest entry
                    _ = agent.env_baseline[city].pop(0)
                    agent.env_baseline[city].append(r_t)
            episode_reward_list.append(episode_reward)
            iterations.set_description(f"Task {task+1} ({city}) | Episode Reward: {[int(v) for v in episode_reward_list]}")
        # perform on-policy backprop
        grad_norms = agent.training_step(city=city)
        a_grad_norms_list.append(grad_norms['a_grad_norm'])
        v_grad_norms_list.append(grad_norms['v_grad_norm'])
        # Send current statistics to screen
        iterations.set_description(f"Task {task+1} ({city}) | Task Reward: {task_reward:.2f} | Grad Norms: Actor={grad_norms['a_grad_norm']:.2f}, Critic={grad_norms['v_grad_norm']:.2f}")
        # Checkpoint best performing model
        agent.save_checkpoint(path=f"{args.directory}/rl_logs/a2c_gnn.pth")
else:
    #######################################
    ######Loop over Test Environments######
    #######################################
    for city in test_env:
        print("===========================")
        print(f"Testing AMoD Controller on {city} environment")
        print("===========================")
        # Define AMoD Simulator Environment
        scenario = Scenario(json_file=f"data/scenario_{city}.json", demand_ratio=args.demand_ratio[city], json_hr=args.json_hr[city], sd=args.seed, json_tstep=args.test_tstep[city], tf=args.max_steps)
        env = AMoD(scenario, beta=beta[city])
        agent = A2C(env=env, input_size=36, hidden_size=args.hidden_size, clip=args.clip, env_baseline=env_baseline).to(device)
        try:
            agent.load_checkpoint(path=f"{args.directory}/rl_logs/a2c_gnn.pth") # Load pre-trained agent
        except:
            raise ValueError('Impossible to find a pre-trained agent checkpoint. Check you are using the correct directory and/or if you pre-trained an agent before running this testing script!')
        parser = GNNParser(env, json_file=f"data/scenario_{city}.json")
        # initialize test settings
        num_test_iter = args.max_test_iter
        iterations = trange(num_test_iter)
        num_episodes = args.max_episodes
        # test book-keeping
        episode_reward_list = []
        task_reward_list = []
        task_demand_list = []
        task_cost_list = []
        for task in iterations:
            task_reward = 0
            task_episode_reward_list = []
            # initialize placeholder values for RL2             
            a_t = torch.zeros((env.nregion, 1)).float() # previous action
            ext1_r_t = torch.zeros((env.nregion, 1)).float() # previous dispacthing reward for all nodes
            ext2_r_t = torch.zeros((env.nregion, 1)).float() # previous rebalancing reward for all nodes
            ext_d_t = [False]*env.nregion # previous done-signal for all nodes
            ext_d_t = torch.tensor(ext_d_t).view(env.nregion, 1)
            h_t_a = torch.zeros((env.nregion, agent.hidden_size)).float() # actor graph-GRU hidden state
            h_t_c = torch.zeros((env.nregion, agent.hidden_size)).float() # critic graph-GRU hidden state
            for episode in range(num_episodes):
                d_t = False
                episode_reward_list = []
                a_t_list = [] # use to store actions over a single episode
                acc_t_list = [] # use to store OD rebalancing flows over a single episode
                obs = env.reset() # reset environment
                episode_reward = 0
                episode_served_demand = 0
                episode_rebalancing_cost = 0
                while not d_t:
                    # take matching step (Step 1 in paper)
                    obs, paxreward, done, info, ext_paxreward, ext_done = env.pax_step(CPLEXPATH=args.cplexpath, PATH='metarl_amod_test')
                    episode_reward += paxreward
                    ext1_r_t = torch.from_numpy(ext_paxreward).view(env.nregion, 1)
                    # parse raw environment observations
                    data = parser.parse_obs(obs, a_t, ext1_r_t, ext2_r_t, ext_d_t)
                    # use GNN-RL policy (Step 2 in paper)
                    # at test time, we do not sample, we take the mean of our Dirichlet policy
                    with torch.no_grad():
                        c, _, h_t_a, h_t_c = agent(data, h_t_a, h_t_c) # c: concentration, h_t_X: hidden states
                        action_rl = (c) / (c.sum() + jitter) # computes the Dirichlet mean (jitter for numerical stability)
                        a_t_list.append(list(action_rl.cpu().numpy()))
                        acc_t_list.append([obs[0][n][env.time+1] for n in env.region])
                    # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                    desiredAcc = {env.region[i]: int(action_rl[i] *dictsum(env.acc,env.time+1))for i in range(len(env.region))}
                    # solve minimum rebalancing distance problem (Step 3 in paper)
                    rebAction = solveRebFlow(env=env,res_path='metarl_amod_test',desiredAcc=desiredAcc,CPLEXPATH=args.cplexpath)
                    # Take action in environment
                    new_obs, rebreward, d_t, info, ext_rebreward, ext_done = env.reb_step(rebAction)
                    episode_reward += rebreward
                    ext2_r_t = torch.from_numpy(ext_rebreward).view(env.nregion, 1) 
                    ext_d_t = torch.from_numpy(np.array(ext_done)).view(env.nregion, 1)
                    r_t = paxreward + rebreward
                    task_reward += r_t
                    episode_reward_list.append(r_t)
                    episode_served_demand += info['served_demand']
                    episode_rebalancing_cost += info['rebalancing_cost']
                    a_t = a_t.view(env.nregion, 1).float()
                
                if episode == num_episodes - 1: # evaluate performance after k interactions with the environment
                    task_reward_list.append(episode_reward)
                    task_demand_list.append(episode_served_demand)
                    task_cost_list.append(episode_rebalancing_cost)
                task_episode_reward_list.append(episode_reward)
                # Send current statistics to screen
                if episode == (num_episodes-1):
                    iterations.set_description(f"Task {task+1} | Task Reward: {task_reward:.2f} | Episode Reward: {[int(v) for v in task_episode_reward_list]} | Served Demand: {episode_served_demand:.2f} | Reb Cost: {episode_rebalancing_cost:.2f}| Aggregated: {np.mean(task_reward_list):.0f} +- {np.std(task_reward_list):.0f}")
                else:
                    iterations.set_description(f"Task {task+1} | Task Reward: {task_reward:.2f} | Episode Reward: {[int(v) for v in task_episode_reward_list]} | Aggregated: {np.mean(task_reward_list):.0f} +- {np.std(task_reward_list):.0f}")
    




