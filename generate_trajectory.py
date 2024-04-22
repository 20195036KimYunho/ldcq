import numpy as np
import gymnasium as gym
import utils as ut
import time
import os
import sys
from tqdm  import tqdm
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default='ARCLE/O2ARCv2Env-v0')
parser.add_argument('--tasks', nargs='+', type=str, required=True) # 46442a0e
parser.add_argument('--arc_folder_path', type=str, default=os.path.dirname(os.path.abspath(__file__)))# directory to save data
parser.add_argument('--num_samples', type=int, default=10000)# number of samples to make in each tasks.
parser.add_argument('--max_grid_size', nargs=2, type=int, metavar=('h', 'w'), default=(30,30))
parser.add_argument('--horizon',type=int, default= 5)
parser.add_argument('--save_whole_trace',type=bool, default= True)
parser.add_argument('--save_seg_trace',type=bool, default= True)
parser.add_argument('--render_mode', type=str, default=None) #ansi if you want to see the real-time result, else do not set this argument
parser.add_argument('--max_trial', type=int, default=3) 
parser.add_argument('--data_loader',type=str,default='random')#random / if you make new type of dataloader, change this argument
args = parser.parse_args()

env = args.env
tasks=args.tasks
arc_folder_path=args.arc_folder_path
num_samples = args.num_samples
max_grid_size = args.max_grid_size
H = args.horizon
save_whole_trace=args.save_whole_trace
save_seg_trace=args.save_seg_trace
render_mode=args.render_mode
max_trial=args.max_trial
data_loader=args.data_loader

if data_loader=='random':
    ARC_maker=ut.RandomGridLoader(tasks=tasks,num_samples=num_samples)
else :
    raise NotImplementedError

possible_trace=[
    [33,29,30,24,29,30,26,27,34],
    [33,29,30,25,29,30,26,27,34],
    [33,29,30,24,29,30,24,29,30,24,34],
    [33,29,30,25,29,30,25,29,30,25,34],
    ]

actions = {
    '46442a0e' : possible_trace #list of possible trajectories are saved in dictionary.
}

           
env = gym.make(env,render_mode=render_mode, data_loader=ARC_maker, max_grid_size=max_grid_size, colors = 10, max_episode_steps=None, max_trial=max_trial)   

for i in tqdm(range(len(tasks)),desc="total",position=0):
    ex_in, ex_out, _, _, desc = ARC_maker.pick(data_index=i)
    for j in tqdm(range(len(ex_in)),desc=f"task:{tasks[i]}",position=1):
        
        data = {
            "desc" : desc,
            "step" : [],
            "selection" : [],
            "operation" : [],
            "operation_name":[],
            "reward" : [],
            "terminated" : [],
            "in_grid" : [],
            "out_grid" : [],
            "grid" : [],
            "selection_mask" : []
        }
    
        obs, info = env.reset(options={'prob_index':i, 'subprob_index':j})
        
        sel = [0,0,0,0] #x,y,h,w (row,col,height,width)
        
        sel_mask =np.zeros((30,30),dtype=np.int8)
        reward = 0
        term = False
        grid=np.full((30,30),10,dtype=np.uint8)
        h,w = obs['input_dim']
        
        # repeat start state.
        for _ in range(H//2):
            operation=35
            grid_pad=grid.copy()
            grid_pad[:h,:w]=obs['grid'][:h,:w]
            data['grid'].append(grid_pad.tolist())
            data['selection'].append(sel)
            data['selection_mask'].append(sel_mask.tolist())
            data['operation'].append(operation)
            data['operation_name'].append(ut.mapping_operation(operation))
            data['reward'].append(reward)
            data['terminated'].append(term)
            data['step'].append(0)
        
        #make trajectory using ARCLE
        idx=np.random.randint(0,len(actions[tasks[i]]))
        opr=actions[tasks[i]][idx]
        temp_sel_mask=ut.make_sel(idx,h,w)
        
        trace_length=0
        while True:
            operation=opr[k]
    
            sel = []
            if operation==35:
                # pass
                raise NotImplementedError
            else:
                if render_mode=='ansi':
                    time.sleep(1)
                else:
                    time.sleep(0)
                    
                sel_mask=ut.sel_bbox_to_mask(sel)
                
                action = {'selection': sel_mask.astype(bool), 'operation': operation}
                
                obs,reward,term,trunc,info = env.step(action)
                g_h,g_w=obs['grid_dim']
                
                grid_pad=grid.copy()
                grid_pad[:g_h,:g_w]=obs['grid'][:g_h,:g_w]
                data['grid'].append(grid_pad.tolist())
                
                data['selection'].append(sel)
                data['selection_mask'].append(sel_mask.tolist())
                data['operation'].append(operation)
                data['operation_name'].append(ut.mapping_operation(operation))
                data['reward'].append(reward)
                data['terminated'].append(term)
                data['step'].append(k)

        # repeat end state.
        for _ in range(H//2):
            operation=35
            grid_pad=grid.copy()
            grid_pad[:g_h,:g_w]=obs['grid'][:g_h,:g_w]
            data['grid'].append(grid_pad.tolist())
            data['selection'].append(sel)
            data['selection_mask'].append(sel_mask.tolist())
            data['operation'].append(operation)
            data['operation_name'].append(ut.mapping_operation(operation))
            data['reward'].append(reward)
            data['terminated'].append(term)
            data['step'].append(len(opr)-1)
        
        data['in_grid']=data['grid'][0]
        data['out_grid']=data['grid'][-1]
        
        ## check it is reasonable trajectory(compare output from ARCLE with hadcrafted answer)
        if not np.array_equal(np.array(data['grid'][-1])[:g_h, :g_w], ex_out[j]):
            print("not correct answer")
            continue
        
        #set data path correctly.
        parts = arc_folder_path.split('/')
        if parts[-1]=="ARC_data":
            pass
        elif os.path.exists(arc_folder_path+f'/ARC_data'):
            arc_folder_path=arc_folder_path+f'/ARC_data'
        else :
            os.makedirs(arc_folder_path+f'/ARC_data')
            arc_folder_path=arc_folder_path+f'/ARC_data'
          
        #if save option is true, save the sample traces.
        if save_whole_trace == True:
            ut.save_whole(data,j,arc_folder_path)
            
        if save_seg_trace == True:
            ut.save_seg(data,j,H,arc_folder_path)

env.close()