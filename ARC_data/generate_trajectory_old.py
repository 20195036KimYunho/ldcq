import json
import numpy as np
import gymnasium as gym
from arcle.loaders import ARCLoader, Loader
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import time
import os
import sys
import copy 
from tqdm  import tqdm

class TrajectoryMakerLoader(Loader):
    def get_path(self, **kwargs) -> List[str]:
        return ['']

    def parse(self, **kwargs) -> List[Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], Dict]]:
        dat = []

        #randomly generate inputs
        for task in tasks:
            ti: List[NDArray] = []
            to: List[NDArray] = []
            ei: List[NDArray] = []
            eo: List[NDArray] = []
            num=0
            while num < num_samples:
                num += 1
                # set grid size randomly and randomly generate grid.
                h= np.random.randint(1,15)
                w=h # w=np.random.randint(1,15) for rectangular grid
                rand_grid=np.random.randint(0,10, size=[h,w],dtype=np.uint8)
                ti.append(rand_grid)
                to.append(self.make_answer(task,rand_grid))
                
                #not used for making trajecotry
                ei.append(None)
                eo.append(None)

                desc = {'id': f'{task}'}
            dat.append((ti,to,ei,eo,desc))    
        return dat

    def make_answer(self,task,grid):
        if task == '46442a0e':
            h,w=grid.shape
            ans=np.zeros([2*h,2*w],dtype=np.int8)
            ans[:h,:w]=grid.copy()
            ans[h:2*h,:w]=np.rot90(grid).copy()
            ans[h:2*h,w:2*w]=np.rot90(grid,2).copy()
            ans[:h,w:2*w]=np.rot90(grid,3).copy()
            return ans
        else :
            raise NotImplementedError
            
def make_sel(i,h,w):
    ret=[]
    mask=np.zeros((30,30),dtype=np.int8)
    
    #33 - crop_grid
    mask[:,:] = 0
    mask[:2*h,:2*w] = 1
    temp=mask.copy()
    ret.append(temp)
    
    #29 - output_copy
    mask[:,:] = 0
    mask[:h,:w]=1
    temp=mask.copy()    
    ret.append(temp)
        
    if i==0:
        #30 - paste
        mask[:,:] = 0
        mask[h:2*h,:w]=1
        temp=mask.copy()
        ret.append(temp)

        #24 - counterclock_rotate
        mask[:,:] = 0
        mask[h:2*h,:w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #29 - output_copy
        mask[:,:] = 0
        mask[:2*h,:w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #30 - paste
        mask[:,:] = 0
        mask[:2*h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #26 - horizon_flip
        mask[:,:] = 0
        mask[:2*h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #27 - vertical_flip
        mask[:,:] = 0
        mask[:2*h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        # 34 - submit
        mask[:,:] = 0
        mask[:2*h,:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
    elif i==1:
        #30 - paste
        mask[:,:] = 0
        mask[:h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)

        #25 - clock_rotate
        mask[:,:] = 0
        mask[:h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #29 - output_copy
        mask[:,:] = 0
        mask[:h,:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #30 - paste
        mask[:,:] = 0
        mask[h:2*h,:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #26 - horizon_flip
        mask[:,:] = 0
        mask[h:2*h,:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #27 - vertical_flip
        mask[:,:] = 0
        mask[h:2*h,:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        # 34 - submit
        mask[:,:] = 0
        mask[:2*h,:2*w]=1
        temp=mask.copy()
        ret.append(temp)
    
    elif i==2:
        #30 - paste
        mask[:,:] = 0
        mask[h:2*h,:w]=1
        temp=mask.copy()
        ret.append(temp)

        #24 - counterclock_rotate
        mask[:,:] = 0
        mask[h:2*h,:w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #29 - output_copy
        mask[:,:] = 0
        mask[h:2*h,:w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #30 - paste
        mask[:,:] = 0
        mask[h:2*h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #24 - counterclock_rotate
        mask[:,:] = 0
        mask[h:2*h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #29 - output_copy
        mask[:,:] = 0
        mask[h:2*h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #30 - paste
        mask[:,:] = 0
        mask[:h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #24 - counterclock_rotate
        mask[:,:] = 0
        mask[:h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        # 34 - submit
        mask[:,:] = 0
        mask[:2*h,:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
    elif i==3:
        #30 - paste
        mask[:,:] = 0
        mask[:h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)

        #25 - clock_rotate
        mask[:,:] = 0
        mask[:h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #29 - output_copy
        mask[:,:] = 0
        mask[:h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #30 - paste
        mask[:,:] = 0
        mask[h:2*h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #25 - clock_rotate
        mask[:,:] = 0
        mask[h:2*h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #29 - output_copy
        mask[:,:] = 0
        mask[h:2*h,w:2*w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #30 - paste
        mask[:,:] = 0
        mask[h:2*h,:w]=1
        temp=mask.copy()
        ret.append(temp)
        
        #25 - clock_rotate
        mask[:,:] = 0
        mask[h:2*h,:w]=1
        temp=mask.copy()
        ret.append(temp)
        
        # 34 - submit
        mask[:,:] = 0
        mask[:2*h,:2*w]=1
        temp=mask.copy()
        ret.append(temp)
    else:
        NotImplementedError
    return ret

root_folder='/home/yunho' # chage it to your folder
sys.path.append(root_folder) 

num_samples = 100
max_grid_size = (30,30)
H = 5

tasks = ['46442a0e'] #task_id to augument

path=[
    [33,29,30,24,29,30,26,27,34],
    [33,29,30,25,29,30,26,27,34],
    [33,29,30,24,29,30,24,29,30,24,34],
    [33,29,30,25,29,30,25,29,30,25,34],
    ]

actions = {
    '46442a0e' : path #list of possible trajectories are saved in dictionary.
}

ACTION = [
    "Color0",#0
    "Color1",#1
    "Color2",#2
    "Color3",#3
    "Color4",#4
    "Color5",#5
    "Color6",#6
    "Color7",#7
    "Color8",#8
    "Color9",#9
    "FloodFill0",#10
    "FloodFill1",#11
    "FloodFill2",#12
    "FloodFill3",#13
    "FloodFill4",#14
    "FloodFill5",#15
    "FloodFill6",#16
    "FloodFill7",#17
    "FloodFill8",#18
    "FloodFill9",#19
    "MoveU",#20
    "MoveD",#21
    "MoveR",#22
    "MoveL",#23
    "Rotate90",#24
    "Rotate270",#25
    "FlipH",#26
    "FlipV",#27 
    "CopyI",#28
    "CopyO",#29 
    "Paste",#30
    "CopyInput",#31
    "ResetGrid",#32
    "CropGrid"#33 
    "Submit",#34
    "None" #35
]

ARC_maker=TrajectoryMakerLoader()
            
env = gym.make('ARCLE/O2ARCv2Env-v0',render_mode=None, data_loader=ARC_maker, max_grid_size=max_grid_size, colors = 10, max_episode_steps=None, max_trial=3)   

for i in tqdm(range(len(tasks)),desc="total",position=0):
    ex_in, ex_out, _, _, desc = ARC_maker.pick(data_index=i)
    for j in tqdm(range(len(ex_in)),desc=f"task:{tasks[i]}",position=1):
        data = {
            "desc" : desc,
            "in_grid" : [],
            "out_grid" : [],
            "step" : [],
            "selection" : [],
            "operation" : [],
            "reward" : [],
            "terminated" : [],
            "grid" : [],
            "selection_mask" : []
        }
    
        obs, info = env.reset(options={'prob_index':i, 'subprob_index':j})
        
        sel = [0,0,0,0] #x,y,h,w
        # sel = {
        #         'x':0,
        #         'y':0,
        #         'h':0,
        #         'w':0
        #     }
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
            data['reward'].append(reward)
            data['terminated'].append(term)
            data['step'].append(0)
        
        #make trajectory using ARCLE
        idx=np.random.randint(0,len(actions[tasks[i]]))
        opr=actions[tasks[i]][idx]
        temp_sel_mask=make_sel(idx,h,w)
        
        for k in range(len(opr)):
            operation=opr[k]
    
            sel = []
            if operation==35:
                # pass
                raise NotImplementedError
            else:
                time.sleep(0)
                if tasks[i]=='46442a0e':
                    sel_mask=temp_sel_mask[k]
                    rows = np.any(sel_mask, axis=1)
                    cols = np.any(sel_mask, axis=0)
                    
                    if not np.any(rows) or not np.any(cols): # if not select anythin
                        x = int(0)
                        y = int(0)
                        h = int(0)
                        w = int(0)
                        sel = [x,y,h,w]
                        # sel['x']=int(cmin)
                        # sel['y']=int(rmin)
                        # sel['h']=int(cmax-cmin+1)
                        # sel['w']=int(rmax-rmin+1)
                    
                    else: # if select. we consider x,y as a point and h,w as a length of edge.
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]

                        x = int(cmin)
                        y = int(rmin)
                        h = int(cmax-cmin+1)
                        w = int(rmax-rmin+1)
                        sel = [x,y,h,w]
                        # sel['x']=int(cmin)
                        # sel['y']=int(rmin)
                        # sel['h']=int(cmax-cmin+1)
                        # sel['w']=int(rmax-rmin+1)
                    
                else :
                    raise NotImplementedError
                
                # op = env.action_space['operation'].sample(mask=operation[:35])
                # print(opr[k])
                action = {'selection': sel_mask.astype(bool), 'operation': operation}
                
                obs,reward,term,trunc,info = env.step(action)
                g_h,g_w=obs['grid_dim']
                
                grid_pad=grid.copy()
                grid_pad[:g_h,:g_w]=obs['grid'][:g_h,:g_w]
                data['grid'].append(grid_pad.tolist())
                
                data['selection'].append(sel)
                data['selection_mask'].append(sel_mask.tolist())
                data['operation'].append(operation)
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
            data['reward'].append(reward)
            data['terminated'].append(term)
            data['step'].append(len(opr)-1)
        
        data['in_grid']=data['grid'][0]
        data['out_grid']=data['grid'][-1]
        ## check it is reasonable trajectory
        if not np.array_equal(np.array(data['grid'][-1])[:g_h, :g_w], ex_out[j]):
            print("not correct answer")
            continue
        
        #save whole trajectory
        folder_path=root_folder+f'/ldcq/ARC_data/whole/{tasks[i]}'  
        
        whole_data={
            "desc" : desc,
            "in_grid" : data['in_grid'],
            "out_grid" : data['out_grid'],
            "step" : data['step'][2:-2],
            "selection" : data['selection'][2:-2],
            "operation" : data['operation'][2:-2],
            "reward" : data['reward'][2:-2],
            "terminated" : data['terminated'][2:-2],
            "grid" : data['grid'][2:-2],
            "selection_mask" : data['selection_mask'][2:-2]
        }
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        with open(f"{folder_path}/{whole_data['desc']['id']}_{j}.json", 'w') as f:
            json.dump(whole_data,f)    
            f.close()
        
        #save segment trajectory
        sub_folder_path=root_folder+f'/ldcq/ARC_data/segment/{tasks[i]}/{tasks[i]}_{j}'  
        if not os.path.exists(sub_folder_path):
            os.makedirs(sub_folder_path)
            
        #when previous trajectory is exist, delete and save trajectory.(because of difference of path length)
        else:
            for filename in os.listdir(sub_folder_path):
                file_path = os.path.join(sub_folder_path, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        for l in range(len(data['grid'])-H+1):
            seg_data={
                "desc" : {"id":data['desc']['id']+f"_{j}_{l}"},
                "in_grid" : data['in_grid'],
                "out_grid" : data['out_grid'],
                "step" : data['step'][l:l+H],
                "selection" : data['selection'][l:l+H],
                "operation" : data['operation'][l:l+H],
                "reward" : data['reward'][l:l+H],
                "terminated" : data['terminated'][l:l+H],
                "grid" : data['grid'][l:l+H],
                "selection_mask" : data['selection_mask'][l:l+H]
                
            }
            
            with open(f"{sub_folder_path}/{seg_data['desc']['id']}.json", 'w') as f:
                json.dump(seg_data,f)    
                f.close()
env.close()