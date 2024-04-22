import json
from arcle.loaders import Loader
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import os
import numpy as np

class RandomGridLoader(Loader):
    def get_path(self, **kwargs) -> List[str]:
        return ['']

    def parse(self, **kwargs) -> List[Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], Dict]]:
        dat = []

        tasks=kwargs['tasks']
        num_samples=kwargs['num_samples']
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

action_names = [
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
    "CropGrid",#33 
    "Submit",#34
    "None" #35
]


def sel_bbox_to_mask(selection_bbox):
    x,y,h,w=selection_bbox
    sel_mask =np.zeros((30,30),dtype=np.int8)
    sel_mask[x:x+h,y:y+w]=1
    return sel_mask

def mapping_operation(n):
    return action_names[n]

def save_whole(data,sample_number,arc_folder_path): #save whole trajectory
    
        whole_folder_path=arc_folder_path+'/whole'  
        if not os.path.exists(whole_folder_path):
            os.makedirs(whole_folder_path)
            
        task_folder_path=whole_folder_path+f"/{data['desc']['id']}"
        if not os.path.exists(task_folder_path):
            os.makedirs(task_folder_path)
            
        whole_data={
            "desc" : {"id":data['desc']['id']+f"_{sample_number}"},
            "step" : data['step'][2:-2],
            "selection" : data['selection'][2:-2],
            "operation" : data['operation'][2:-2],
            "operation_name": data['operation_name'][2:-2],
            "reward" : data['reward'][2:-2],
            "terminated" : data['terminated'][2:-2],
            "in_grid" : data['in_grid'],
            "out_grid" : data['out_grid'],
            "grid" : data['grid'][2:-2],
            "selection_mask" : data['selection_mask'][2:-2]
        }
        with open(f"{task_folder_path}/{whole_data['desc']['id']}_{sample_number}.json", 'w') as f:
            json.dump(whole_data,f)    
            f.close()

def save_seg(data,sample_number,H,arc_folder_path):#save segment trajectory
        
        seg_folder_path=arc_folder_path+'/segment'  
        if not os.path.exists(seg_folder_path):
            os.makedirs(seg_folder_path)
            
        task_folder_path=seg_folder_path+f"/{data['desc']['id']}"
        if not os.path.exists(task_folder_path):
            os.makedirs(task_folder_path)
            
        sub_folder_path=task_folder_path+f"/{data['desc']['id']}_{sample_number}"
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
                "desc" : {"id":data['desc']['id']+f"_{sample_number}_{l}"},
                "step" : data['step'][l:l+H],
                "selection" : data['selection'][l:l+H],
                "operation" : data['operation'][l:l+H],
                "operation_name": data['operation_name'][l:l+H],
                "reward" : data['reward'][l:l+H],
                "terminated" : data['terminated'][l:l+H],
                "in_grid" : data['in_grid'],
                "out_grid" : data['out_grid'],
                "grid" : data['grid'][l:l+H],
                "selection_mask" : data['selection_mask'][l:l+H]
                
            }
            
            with open(f"{sub_folder_path}/{seg_data['desc']['id']}.json", 'w') as f:
                json.dump(seg_data,f)    
                f.close()
