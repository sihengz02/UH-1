import os, sys
from statistics import mode
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_mimic import Actor, StateHistoryEncoder, get_activation
from rsl_rl.modules.estimator import Estimator
import argparse
import code
import shutil

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path, checkpoint

class HardwareRefNN(nn.Module):
    def __init__(self,  num_prop,
                        num_demo,
                        text_feat_input_dim,
                        text_feat_output_dim,
                        feat_hist_len,
                        num_scan,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        tanh,
                        actor_hidden_dims=[512, 256, 128],
                        scan_encoder_dims=[128, 64, 32],
                        depth_encoder_hidden_dim=512,
                        activation='elu',
                        priv_encoder_dims=[64, 20]
                        ):
        super().__init__()

        # assert text_feat_input_dim == 0 and text_feat_output_dim == 0, "Not implemented"
        self.text_feat_input_dim = text_feat_input_dim
        self.text_feat_output_dim = text_feat_output_dim
        self.feat_hist_len = feat_hist_len

        self.num_demo = num_demo

        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        num_obs = num_prop + num_scan + num_hist*num_prop + num_priv_latent + num_priv_explicit
        self.num_obs = num_obs
        activation = get_activation(activation)
        
        self.actor = Actor(num_prop, 
                           num_demo,
                           text_feat_input_dim,
                           text_feat_output_dim,
                           feat_hist_len,
                           num_scan, num_actions, 
                           scan_encoder_dims, actor_hidden_dims, 
                           priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, 
                           activation, tanh_encoder_output=tanh)

        self.estimator = Estimator(input_dim=num_prop, output_dim=num_priv_explicit, hidden_dims=[128, 64])
        
    def forward(self, obs):
        # obs[:, self.num_prop + self.num_scan : self.num_prop+  self.num_scan+self.num_priv_explicit] = self.estimator(obs[:, :self.num_prop])
        # obs[:, self.num_prop + self.num_demo +self.num_scan : self.num_prop+ self.num_demo + self.num_scan+self.num_priv_explicit] = self.estimator(obs[:, :self.num_prop])
        obs = torch.concat([obs[:, -self.text_feat_input_dim:], obs], dim=1)
        return self.actor(obs, hist_encoding=True, eval=False)
        # return obs, depth_latent

def play(args):    
    load_run = "../../logs/h1/" + args.exptid
    checkpoint = args.checkpoint

    n_priv_explicit = 3
    n_priv_latent = 4 + 1 + 19*2
    num_scan = 0
    num_actions = 19
    
    n_proprio = 3 + 2 + 2 + 19*3 + 2
    history_len = 10
    
    num_demo = 9 + 3 + 3 + 3 + 6*3
    feat_hist_len = 4
    text_feat_input_dim = feat_hist_len * n_proprio
    text_feat_output_dim = 16

    device = torch.device('cpu')
    policy = HardwareRefNN(n_proprio, 
                           num_demo,
                           text_feat_input_dim,
                           text_feat_output_dim,
                           feat_hist_len,
                           num_scan, 
                           n_priv_latent, n_priv_explicit, history_len, 
                           num_actions, args.tanh).to(device)
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=checkpoint)
    load_run = os.path.dirname(load_path)
    print(f"Loading model from: {load_path}")
    ac_state_dict = torch.load(load_path, map_location=device)
    policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)

    policy.estimator.load_state_dict(ac_state_dict['estimator_state_dict'], strict=True)
    
    policy = policy.to(device)#.cpu()
    if not os.path.exists(os.path.join(load_run, "traced")):
        os.mkdir(os.path.join(load_run, "traced"))

    # Save the traced actor
    policy.eval()
    with torch.no_grad(): 
        num_envs = 2
        
        obs_input = torch.ones(num_envs, n_proprio + num_demo + num_scan + n_priv_explicit + history_len*n_proprio, device=device)
        print("obs_input shape: ", obs_input.shape)
        
        traced_policy = torch.jit.trace(policy, obs_input)

        save_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-base_jit.pt")
        traced_policy.save(save_path)
        print("Saved traced_actor at ", os.path.abspath(save_path))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exptid', type=str)
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--tanh', action='store_true')
    args = parser.parse_args()
    play(args)
