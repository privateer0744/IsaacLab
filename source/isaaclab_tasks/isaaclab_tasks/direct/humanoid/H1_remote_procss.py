# modify H1Env. functions here. add data processing here
# utilizes broadcasting, PyTorch will automatically expand the shape of tensor b to align with tensor a, 
# so you can compute only one group of data from recvdata and cmddata  

import numpy as np
import torch
#import isaaclab_tasks.direct.humanoid.exoskltn_intef
from isaaclab_tasks.direct.humanoid.exoskltn_intef import Communicating
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

Scale_x = 1.0
Scale_z = 1.0

Fixed_biped_width = 0.5    #width of the robot, attention not the width of exoskeleton


def euler_to_quaternion(roll:torch.tensor, pitch:torch.tensor, yaw:torch.tensor):
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)

    q = torch.tensor([
        cr * cp * cy + sr * sp * sy,  # w
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy   # z
    ])
    return q


def quaternion_to_rotation_matrix(quat:torch.tensor):
    quat = quat / quat.norm(dim=-1, keepdim=True)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    R = torch.stack([
        1 - 2 * (y**2 + z**2),  2 * (x*y - w*z),      2 * (x*z + w*y),
        2 * (x*y + w*z),        1 - 2 * (x**2 + z**2),2 * (y*z - w*x),
        2 * (x*z - w*y),        2 * (y*z + w*x),      1 - 2 * (x**2 + y**2)
    ], dim=-1).reshape(*quat.shape[:-1], 3, 3)
    return R  # (batch, 3, 3)

def quaternion_conjugate(quat:torch.tensor):
    return torch.cat([quat[..., 0:1], -quat[..., 1:4]], dim=-1)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def Judge_contact(accz_root: torch.Tensor, h_lf_w: torch.Tensor, h_rf_w: torch.Tensor):
    GRAV = -6

    fall_mask = accz_root < GRAV  

    lf_lower = h_lf_w - h_rf_w <- 0.01 
    rf_lower = h_rf_w - h_lf_w >- 0.01

    ct_lf = torch.where(fall_mask, torch.zeros_like(h_lf_w), torch.where(lf_lower, torch.zeros_like(h_lf_w), torch.ones_like(h_lf_w)))
    ct_rf = torch.where(fall_mask, torch.zeros_like(h_rf_w), torch.where(rf_lower, torch.ones_like(h_rf_w), torch.zeros_like(h_rf_w)))

    return ct_lf, ct_rf



class Intermediate_Motion:
    def __init__(self, initialdata:list, device = "cuda:0"):

        self.device = device

        [root_flx, root_fly,  root_flz,  root_eula_ly,  root_flxd,  root_flzd,  root_rotlyd,  
         root_frx, root_fry,  root_frz,  root_eula_ry,  root_frxd,  root_frzd,  root_rotryd,  
         posture_T,  angular_vel_T,  contact_flgl, contact_flgr] = initialdata
        
       
        self.master_root_pos_lf = torch.tensor([Scale_x * root_flx, root_fly, Scale_z * root_flz], device=self.device)
        self.master_root_vel_lf = torch.tensor([Scale_x * root_flxd, 0.0, Scale_z * root_flzd], device=self.device)
        tensor_eula = torch.tensor([0, root_eula_ly, 0], device=self.device)
        self.master_root_quat_lf = euler_to_quaternion(tensor_eula[0], tensor_eula[1], tensor_eula[2]) # all the rotation expressed by quaternion
        self.master_root_quat_lf = self.master_root_quat_lf.to(self.device)

        self.master_root_pos_rf = torch.tensor([Scale_x * root_frx, root_fry, Scale_z * root_frz], device=self.device)
        self.master_root_vel_rf = torch.tensor([Scale_x * root_frxd, 0.0, Scale_z * root_frzd], device=self.device)
        tensor_eula = torch.tensor([0, root_eula_ry, 0], device=self.device)
        self.master_root_quat_rf = euler_to_quaternion(tensor_eula[0], tensor_eula[1], tensor_eula[2])
        self.master_root_quat_rf = self.master_root_quat_rf.to(self.device)
        
        self. master_root_omeg_lf = torch.zeros(3, device=self.device)
        self. master_root_omeg_lf[1] = root_rotlyd

        self. master_root_omeg_rf = torch.zeros(3, device=self.device)
        self. master_root_omeg_rf[1] = root_rotryd

        self.master_posture_T = torch.tensor(posture_T, device=self.device)
        self.master_angular_vel_T = torch.tensor(angular_vel_T, device=self.device)
        self.master_contact_flgl = torch.tensor(contact_flgl, device=self.device).unsqueeze(-1)
        self.master_contact_flgr = torch.tensor(contact_flgr, device=self.device).unsqueeze(-1)

        pass

    def MasterData_Process(self, recvdata:list): #deal with received data

        #print("******************* recvdata: ", recvdata)
        # modify if the message from exosklton is expanded
        [root_flx, root_fly,  root_flz,  root_eula_ly,  root_flxd,  root_flzd,  root_rotlyd,  
         root_frx, root_fry,  root_frz,  root_eula_ry,  root_frxd,  root_frzd,  root_rotryd,  
         posture_T,  angular_vel_T,  contact_flgl, contact_flgr] = recvdata
        
       
        self.master_root_pos_lf = torch.tensor([Scale_x * root_flx, root_fly, Scale_z * root_flz], device=self.device)
        self.master_root_vel_lf = torch.tensor([Scale_x * root_flxd, 0.0, Scale_z * root_flzd], device=self.device)
        tensor_eula = torch.tensor([0, root_eula_ly, 0], device=self.device)
        self.master_root_quat_lf = euler_to_quaternion(tensor_eula[0], tensor_eula[1], tensor_eula[2]) # all the rotation expressed by quaternion
        self.master_root_quat_lf = self.master_root_quat_lf.to(self.device)

        self.master_root_pos_rf = torch.tensor([Scale_x * root_frx, root_fry, Scale_z * root_frz], device=self.device)
        self.master_root_vel_rf = torch.tensor([Scale_x * root_frxd, 0.0, Scale_z * root_frzd], device=self.device)
        tensor_eula = torch.tensor([0, root_eula_ry, 0], device=self.device)
        self.master_root_quat_rf = euler_to_quaternion(tensor_eula[0], tensor_eula[1], tensor_eula[2])
        self.master_root_quat_rf = self.master_root_quat_rf.to(self.device)

        self. master_root_omeg_lf = torch.zeros(3, device=self.device)
        self. master_root_omeg_lf[1] = root_rotlyd

        self. master_root_omeg_rf = torch.zeros(3, device=self.device)
        self. master_root_omeg_rf[1] = root_rotryd

        self.master_posture_T = torch.tensor(posture_T, device=self.device)
        self.master_angular_vel_T = torch.tensor(angular_vel_T, device=self.device)
        self.master_contact_flgl = torch.tensor(contact_flgl, device=self.device).unsqueeze(-1)
        self.master_contact_flgr = torch.tensor(contact_flgr, device=self.device).unsqueeze(-1)


        #print("******************master_root_pos_lf is on:", self.master_root_pos_lf.shape)
        #print("******************master_root_vel_lf is on:", self.master_root_vel_lf.shape)
        #print("******************master_root_pos_rf is on:", self.master_root_pos_rf.shape)
        #print("******************master_root_vel_rf is on:", self.master_root_vel_rf.shape)
        #print("******************master_root_quat_lf is on:", self.master_root_quat_lf.shape)
        #print("******************master_root_omeg_lf is on:", self.master_root_omeg_lf.shape)
        #print("******************master_root_quat_rf is on:", self.master_root_quat_rf.shape)
        #print("******************master_root_omeg_rf is on:", self.master_root_omeg_rf.shape)
        #print("******************master_contact_flgl is on:", self.master_contact_flgl.shape)
        #print("******************master_contact_flgr is on:", self.master_contact_flgr.shape)



       

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    
    

    



