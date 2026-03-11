# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch, unscale, quat_apply, tensor_clamp, torch_rand_float, quat_conjugate, quat_mul
from glob import glob
from hora.utils.misc import tprint
from .base.vec_task import VecTask


class AllegroHandHora(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.config = config
        # ==========================================
        # ��️ 补丁 A：强制覆盖 Config 字典，掐断 YAML 污染
        # ==========================================
        config['env']['numActions'] = 19
        config['env']['object']['type'] = 'simple_tennis_ball'
        self.num_actions = 19
        # ==========================================
        # before calling init in VecTask, need to do
        # 1. setup randomization
        self._setup_domain_rand_config(config['env']['randomization'])
        # 2. setup privileged information
        self._setup_priv_option_config(config['env']['privInfo'])
        # 3. setup object assets
        self._setup_object_info(config['env']['object'])
        # 4. setup reward
        self._setup_reward_config(config['env']['reward'])
        self.base_obj_scale = config['env']['baseObjScale']
        self.save_init_pose = config['env']['genGrasps']
        self.aggregate_mode = self.config['env']['aggregateMode']
        self.up_axis = 'z'
        self.reset_z_threshold = self.config['env']['reset_height_threshold']
        self.grasp_cache_name = self.config['env']['grasp_cache_name']
        self.evaluate = self.config['on_evaluation']
        self.priv_info_dict = {
            'obj_position': (0, 3),
            'obj_scale': (3, 4),
            'obj_mass': (4, 5),
            'obj_friction': (5, 6),
            'obj_com': (6, 9),
        }

        super().__init__(config, sim_device, graphics_device_id, headless)

        self.debug_viz = self.config['env']['enableDebugVis']
        self.max_episode_length = self.config['env']['episodeLength']
        self.dt = self.sim_params.dt

        if self.viewer:
            cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        # --- 【核心修改 1：设置笼式预抓取姿态】 ---
        # 16个关节：食指、中指、无名指、拇指。每组4个：[旋转, 弯曲1, 弯曲2, 弯曲3]
        cage_pose = [
            0.0, 0.8, 0.5, 0.4,  # 食指弯曲
            0.0, 0.8, 0.5, 0.4,  # 中指弯曲
            0.0, 0.8, 0.5, 0.4,  # 无名指弯曲
            1.1, 0.6, 0.2, 0.5   # 拇指外展并微弯
        ]
        self.allegro_hand_default_dof_pos = torch.tensor(cage_pose, dtype=torch.float, device=self.device)
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self._refresh_gym()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # object apply random forces parameters
        self.force_scale = self.config['env'].get('forceScale', 0.0)
        self.random_force_prob_scalar = self.config['env'].get('randomForceProbScalar', 0.0)
        self.force_decay = self.config['env'].get('forceDecay', 0.99)
        self.force_decay_interval = self.config['env'].get('forceDecayInterval', 0.08)
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        # --- 【核心修改 2：彻底绕过缓存文件加载逻辑】 ---
        # 无论 randomization 开启与否，我们都初始化空字典，防止 reset_idx 报错
        self.saved_grasping_states = {} 
        # 原有的 np.load 循环已移除，以解决 FileNotFoundError

        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        # useful buffers
        self.object_rot_prev = self.object_rot.clone()
        self.object_pos_prev = self.object_pos.clone()
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.torques = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [int, float], 'assume p_gain and d_gain are only scalars'
        self.p_gain = torch.ones((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) * self.p_gain
        self.d_gain = torch.ones((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) * self.d_gain

        # debug and understanding statistics
        self.env_timeout_counter = to_torch(np.zeros((len(self.envs)))).long().to(self.device)  # max 10 (10000 envs)
        self.stat_sum_rewards = 0
        self.stat_sum_rotate_rewards = 0
        self.stat_sum_episode_length = 0
        self.stat_sum_obj_linvel = 0
        self.stat_sum_rotate_rewards = 0 # 重置统计项
        self.stat_sum_torques = 0
        self.env_evaluated = 0
        self.max_evaluate_envs = 500000
        # (在 __init__ 函数的最后一行加入)
        self.env_evaluated = 0
        self.max_evaluate_envs = 500000

        # ==========================================
        # ��️ 补丁 B：底层空间劫持，欺骗 PPO 算法
        # ==========================================
        import gym
        self.num_actions = 19
        self.act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,)) # <--- 换成 act_space！
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device, requires_grad=False)
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._create_object_asset()

        # set allegro_hand dof properties
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []

        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            allegro_hand_dof_props['effort'][i] = 0.5
            if self.torque_control:
                allegro_hand_dof_props['stiffness'][i] = 0.
                allegro_hand_dof_props['damping'][i] = 0.
                allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            else:
                allegro_hand_dof_props['stiffness'][i] = self.config['env']['controller']['pgain']
                allegro_hand_dof_props['damping'][i] = self.config['env']['controller']['dgain']
            allegro_hand_dof_props['friction'][i] = 0.01
            allegro_hand_dof_props['armature'][i] = 0.001

        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)

        hand_pose, obj_pose = self._init_object_pose()

        # compute aggregate size
        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        max_agg_bodies = self.num_allegro_hand_bodies + 2
        max_agg_shapes = self.num_allegro_hand_shapes + 2

        self.envs = []

        self.object_init_state = []

        self.hand_indices = []
        self.object_indices = []

        allegro_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        object_rb_count = 1
        self.object_rb_handles = list(range(allegro_hand_rb_count, allegro_hand_rb_count + object_rb_count))

        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, allegro_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # add object
            object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            object_asset = self.object_asset_list[object_type_id]

            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, 'object', i, 0, 0)
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            obj_scale = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                obj_scale = np.random.uniform(self.randomize_scale_list[i % num_scales] - 0.025, self.randomize_scale_list[i % num_scales] + 0.025)
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
            self._update_priv_buf(env_id=i, name='obj_scale', value=obj_scale)

            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                assert len(prop) == 1
                obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            self._update_priv_buf(env_id=i, name='obj_com', value=obj_com)

            obj_friction = 1.0
            if self.randomize_friction:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
                for p in hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                obj_friction = rand_friction
            self._update_priv_buf(env_id=i, name='obj_friction', value=obj_friction)

            if self.randomize_mass:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)
            else:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

        # =========================================================================
        # �� 修复 1：保存初始手掌状态 (解决 AttributeError: hand_start_states)
        # =========================================================================
        hand_pose, _ = self._init_object_pose()
        initial_hand_quat = torch.tensor(
            [hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w],
            device=self.device, dtype=torch.float
        )
        self.hand_start_states = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)
        self.hand_start_states[:, 3:7] = initial_hand_quat.unsqueeze(0).repeat(self.num_envs, 1)

        # =========================================================================
        # �� 修复 2：获取手掌和指尖的刚体索引 (Rigid Body Indices)
        # 只需要在 envs[0] 中查一次，因为所有环境的内存拓扑布局是完全一致的
        # =========================================================================
        env_ptr0 = self.envs[0]
        hand_actor0 = self.gym.find_actor_handle(env_ptr0, 'hand')

        # 获取手掌基座索引
        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_handle(env_ptr0, hand_actor0, 'base_link')

        # 获取四个指尖的索引
        self.ff_idx = self.gym.find_actor_rigid_body_handle(env_ptr0, hand_actor0, 'link_3.0_tip')  # 食指
        self.mf_idx = self.gym.find_actor_rigid_body_handle(env_ptr0, hand_actor0, 'link_7.0_tip')  # 中指
        self.rf_idx = self.gym.find_actor_rigid_body_handle(env_ptr0, hand_actor0, 'link_11.0_tip')  # 无名指
        self.th_idx = self.gym.find_actor_rigid_body_handle(env_ptr0, hand_actor0, 'link_15.0_tip')  # 拇指

    def reset_idx(self, env_ids):
        # 1. 随机化 PD 增益（保持原样，这对强化学习的稳定性有帮助）
        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_dofs), # <--- 改这里！
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_dofs), # <--- 改这里！
                device=self.device).squeeze(1)
        # 2. 重置外力缓存
        self.rb_forces[env_ids, :, :] = 0.0

        # --- 【核心修改点：改为固定位置生成】 ---
        
        # 3. 重置物体（球）的状态：直接读取初始化时存下的固定位置（球在 0.04m 桌面）
        # 这一行彻底取代了之前报错的 num_scales 循环和 np.load 逻辑
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()

        # 4. 重置机械手关节位置：使用默认姿态（self.allegro_hand_default_dof_pos）
        # 由于手在 _init_object_pose 中设为 0.5m 高空，手会回到那个高度
        pos = self.allegro_hand_default_dof_pos.repeat(len(env_ids), 1)
        self.allegro_hand_dof_pos[env_ids, :] = pos
        self.allegro_hand_dof_vel[env_ids, :] = 0
        
        # 5. 同步更新控制目标和重置参考缓存
        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = pos
        self.init_pose_buf[env_ids, :] = pos.clone()

        # --- 【物理引擎同步】 ---

        # 6. 将更新后的物体位置写入仿真器
        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(object_indices), len(object_indices))
        
        # 7. 将更新后的机械手姿态写入仿真器
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # 8. 清理训练相关的各种 Buffer
        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def compute_observations(self):
        self._refresh_gym()
        # deal with normal observation, do sliding window
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        joint_noise_matrix = (torch.rand(self.allegro_hand_dof_pos.shape) * 2.0 - 1.0) * self.joint_noise_scale
        cur_obs_buf = unscale(
            joint_noise_matrix.to(self.device) + self.allegro_hand_dof_pos, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits
        ).clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:16] = unscale(
            self.allegro_hand_dof_pos[at_reset_env_ids], self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits
        ).clone().unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 16:32] = self.allegro_hand_dof_pos[at_reset_env_ids].unsqueeze(1)
        t_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone()

        self.obs_buf[:, :t_buf.shape[1]] = t_buf
        self.at_reset_buf[at_reset_env_ids] = 0

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_position', value=self.object_pos.clone())

    def compute_reward(self, actions):
        # ==========================================
        # 1. 提取物理位置和旋转
        # ==========================================
        palm_pos = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3]
        palm_rot = self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7]  # 获取手掌旋转
        ff_pos = self.rigid_body_states[:, self.ff_idx, 0:3]
        mf_pos = self.rigid_body_states[:, self.mf_idx, 0:3]
        rf_pos = self.rigid_body_states[:, self.rf_idx, 0:3]
        th_pos = self.rigid_body_states[:, self.th_idx, 0:3]

        target_pos = torch.zeros_like(self.object_pos)
        target_pos[:, 0] = self.object_init_state[:, 0]
        target_pos[:, 1] = self.object_init_state[:, 1]
        target_pos[:, 2] = 0.24

        # ==========================================
        # 2. 构造你的“手工专家先验数据” (代替 Uni 的数据集)
        # ==========================================
        # (A) 手指关节误差 (16维专家姿态差)
        expert_qpos = self.allegro_hand_default_dof_pos.repeat(self.num_envs, 1)
        delta_qpos = self.allegro_hand_dof_pos - expert_qpos

        # (B) 手掌相对位置误差 (要求手掌在球的正上方 6 厘米处)
        ideal_rel_pos = torch.zeros_like(palm_pos)
        ideal_rel_pos[:, 2] = 0.06  # Z轴上方 0.06m
        actual_rel_pos = palm_pos - self.object_pos
        delta_target_hand_pos = actual_rel_pos - ideal_rel_pos

        # (C) 手掌旋转误差 (要求手掌保持初始向下的旋转状态)
        ideal_hand_rot = self.hand_start_states[:, 3:7]  # 读取初始手掌朝向
        delta_target_hand_rot = quat_mul(palm_rot, quat_conjugate(ideal_hand_rot))

        # ==========================================
        # 3. 调用底层的算分引擎 (传入三个 delta)
        # ==========================================
        self.rew_buf[:], self.reset_buf[:] = compute_hand_reward(
            self.object_init_state[:, 2], self.reset_buf, self.progress_buf, self.max_episode_length,
            self.object_pos, palm_pos, ff_pos, mf_pos, rf_pos, th_pos, target_pos, actions,
            self.reset_z_threshold,
            delta_qpos, delta_target_hand_pos, delta_target_hand_rot  # <--- 将算好的先验误差传进去
        )

        # ==========================================
        # 4. 记录核心数据供 TensorBoard 观察
        # ==========================================
        self.extras['goal_dist'] = torch.norm(target_pos - self.object_pos, p=2, dim=-1).mean()
        self.extras['hand_dist'] = torch.norm(palm_pos - self.object_pos, p=2, dim=-1).mean()
        self.extras['obj_height'] = self.object_pos[:, 2].mean()

        if self.evaluate:
            finished_episode_mask = self.reset_buf == 1
            self.stat_sum_rewards += self.rew_buf.sum()
            self.stat_sum_episode_length += (self.reset_buf == 0).sum()
            self.env_evaluated += (self.reset_buf == 1).sum()
            self.env_timeout_counter[finished_episode_mask] += 1
            info = f'progress {self.env_evaluated} / {self.max_evaluate_envs} | ' \
                   f'reward: {self.stat_sum_rewards / self.env_evaluated:.2f} | ' \
                   f'eps length: {self.stat_sum_episode_length / self.env_evaluated:.2f}'
            tprint(info)
            if self.env_evaluated >= self.max_evaluate_envs:
                exit()

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_gym()
        self.compute_reward(self.actions)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        # =========================================================
        # ��️ 补丁 C：终极拦截器 (化解 Dummy Step 的 16 维空动作)
        # =========================================================
        if actions.shape[1] == 16:
            padded_actions = torch.zeros((actions.shape[0], 19), device=actions.device, dtype=actions.dtype)
            padded_actions[:, 3:19] = actions  # 把 16 维动作放到手指的位置
            actions = padded_actions  # 替换成完美的 19 维
        # =========================================================

        self.actions = actions.clone().to(self.device)



        # ==========================================
        # 1. 拦截与分流 (Action Splitting)
        # ==========================================
        # 把 19 维动作拆开：前 3 维控制 Base 移动，后 16 维控制手指关节
        base_actions = self.actions[:, 0:3]
        finger_actions = self.actions[:, 3:19]

        # ==========================================
        # 2. 控制手指关节 (保留 HORA 原有逻辑)
        # ==========================================
        # 注意：这里把 self.actions 替换成了剥离出来的 finger_actions
        targets = self.prev_targets + 1 / 24 * finger_actions
        self.cur_targets[:] = tensor_clamp(targets, self.allegro_hand_dof_lower_limits,
                                           self.allegro_hand_dof_upper_limits)
        self.prev_targets[:] = self.cur_targets.clone()

        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos

        # ==========================================
        # 3. 控制手掌底座移动 (复刻 UniDexGrasp 推进器)
        # ==========================================
        # 为了不每一帧都申请显存，我们做个安全检查，初始化一个力向量表
        if not hasattr(self, 'apply_forces_buf'):
            # 形状：[环境数量, 刚体数量, 3维力(x,y,z)]
            self.apply_forces_buf = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device,
                                                dtype=torch.float)

        # 每一帧开始前，清空上一帧残留的力
        self.apply_forces_buf[:] = 0.0

        # 【核心施力】：将 AI 输出的前 3 维动作转化为巨大的推力
        # 10000.0 是力的放大系数。如果手飞得太慢，可以调成 20000；如果乱飞，可以调成 5000。
        self.apply_forces_buf[:, self.hand_base_rigid_body_index, :] = base_actions * 10000.0

        # 将力表注入物理引擎
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces_buf), None,
                                                gymapi.ENV_SPACE)

        # ==========================================
        # 4. 保留原有的随机扰动力逻辑 (Domain Randomization)
        # ==========================================
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # apply new forces
            obj_mass = to_torch(
                [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for
                 env in self.envs], device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale

            # 对球施加扰动力
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None,
                                                    gymapi.ENV_SPACE)

    def reset(self):
        super().reset()
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        return self.obs_dict

    def step(self, actions):
        super().step(actions)
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def update_low_level_control(self):
        previous_dof_pos = self.allegro_hand_dof_pos.clone()
        self._refresh_gym()
        if self.torque_control:
            dof_pos = self.allegro_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff = dof_vel.clone()
            torques = self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
            self.torques = torch.clip(torques, -0.5, 0.5).clone()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def check_termination(self, object_pos):
        resets = torch.logical_or(
            torch.less(object_pos[:, -1], self.reset_z_threshold),
            torch.greater_equal(self.progress_buf, self.max_episode_length),
        )
        return resets

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

    def _setup_domain_rand_config(self, rand_config):
        self.randomize_mass = rand_config['randomizeMass']
        self.randomize_mass_lower = rand_config['randomizeMassLower']
        self.randomize_mass_upper = rand_config['randomizeMassUpper']
        self.randomize_com = rand_config['randomizeCOM']
        self.randomize_com_lower = rand_config['randomizeCOMLower']
        self.randomize_com_upper = rand_config['randomizeCOMUpper']
        self.randomize_friction = rand_config['randomizeFriction']
        self.randomize_friction_lower = rand_config['randomizeFrictionLower']
        self.randomize_friction_upper = rand_config['randomizeFrictionUpper']
        self.randomize_scale = rand_config['randomizeScale']
        self.scale_list_init = rand_config['scaleListInit']
        self.randomize_scale_list = rand_config['randomizeScaleList']
        self.randomize_scale_lower = rand_config['randomizeScaleLower']
        self.randomize_scale_upper = rand_config['randomizeScaleUpper']
        self.randomize_pd_gains = rand_config['randomizePDGains']
        self.randomize_p_gain_lower = rand_config['randomizePGainLower']
        self.randomize_p_gain_upper = rand_config['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_config['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_config['randomizeDGainUpper']
        self.joint_noise_scale = rand_config['jointNoiseScale']

    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config['enableObjPos']
        self.enable_priv_obj_mass = p_config['enableObjMass']
        self.enable_priv_obj_scale = p_config['enableObjScale']
        self.enable_priv_obj_com = p_config['enableObjCOM']
        self.enable_priv_obj_friction = p_config['enableObjFriction']

    def _update_priv_buf(self, env_id, name, value, lower=None, upper=None):
        # normalize to -1, 1
        s, e = self.priv_info_dict[name]
        if eval(f'self.enable_priv_{name}'):
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            if type(lower) is list or upper is list:
                lower = to_torch(lower, dtype=torch.float, device=self.device)
                upper = to_torch(upper, dtype=torch.float, device=self.device)
            if lower is not None and upper is not None:
                value = (2.0 * value - upper - lower) / (upper - lower)
            self.priv_info_buf[env_id, s:e] = value
        else:
            self.priv_info_buf[env_id, s:e] = 0

    def _setup_object_info(self, o_config):
        self.object_type = o_config['type']
        raw_prob = o_config['sampleProb']
        assert (sum(raw_prob) == 1)

        primitive_list = self.object_type.split('+')
        print('---- Primitive List ----')
        print(primitive_list)
        self.object_type_prob = []
        self.object_type_list = []
        self.asset_files_dict = {
            'simple_tennis_ball': 'assets/ball.urdf',
        }
        for p_id, prim in enumerate(primitive_list):
            if 'cuboid' in prim:
                subset_name = self.object_type.split('_')[-1]
                cuboids = sorted(glob(f'../assets/cuboid/{subset_name}/*.urdf'))
                cuboid_list = [f'cuboid_{i}' for i in range(len(cuboids))]
                self.object_type_list += cuboid_list
                for i, name in enumerate(cuboids):
                    self.asset_files_dict[f'cuboid_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cuboid_list) for _ in cuboid_list]
            elif 'cylinder' in prim:
                subset_name = self.object_type.split('_')[-1]
                cylinders = sorted(glob(f'assets/cylinder/{subset_name}/*.urdf'))
                cylinder_list = [f'cylinder_{i}' for i in range(len(cylinders))]
                self.object_type_list += cylinder_list
                for i, name in enumerate(cylinders):
                    self.asset_files_dict[f'cylinder_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cylinder_list) for _ in cylinder_list]
            else:
                self.object_type_list += [prim]
                self.object_type_prob += [raw_prob[p_id]]
        print('---- Object List ----')
        print(self.object_type_list)
        assert (len(self.object_type_list) == len(self.object_type_prob))

    def _allocate_task_buffer(self, num_envs):
        # extra buffers for observe randomized params
        self.prop_hist_len = self.config['env']['hora']['propHistoryLen']
        self.num_env_factors = self.config['env']['hora']['privInfoDim']
        self.priv_info_buf = torch.zeros((num_envs, self.num_env_factors), device=self.device, dtype=torch.float)
        self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, 32), device=self.device, dtype=torch.float)

    def _setup_reward_config(self, r_config):
        self.angvel_clip_min = r_config['angvelClipMin']
        self.angvel_clip_max = r_config['angvelClipMax']
        self.rotate_reward_scale = r_config['rotateRewardScale']
        self.object_linvel_penalty_scale = r_config['objLinvelPenaltyScale']
        self.pose_diff_penalty_scale = r_config.get('poseDiffPenaltyScale', r_config.get('pose_diff_penalty_scale', -0.3))
        self.torque_penalty_scale = r_config['torquePenaltyScale']
        self.work_penalty_scale = r_config['workPenaltyScale']

    def _create_object_asset(self):
        # object file to asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
        hand_asset_file = self.config['env']['asset']['handAsset']
        # load hand asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = True
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01

        if self.torque_control:
            hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        else:
            hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, hand_asset_options)

        # load object asset
        self.object_asset_list = []
        for object_type in self.object_type_list:
            object_asset_file = self.asset_files_dict[object_type]
            object_asset_options = gymapi.AssetOptions()
            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            self.object_asset_list.append(object_asset)

    def _init_object_pose(self):
        allegro_hand_start_pose = gymapi.Transform()
        # 【修改】将机械手掌心初始化在 0.5m 高空
        allegro_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.5) 
        
        # --- 【关键修正：将掌心旋转至垂直向下】 ---
        # 原本的组合 (-np.pi/2 Y * np.pi/2 X) 导致了掌心朝上
        # 修正为绕 Y 轴旋转 pi/2，这通常能让 Allegro 手的掌心正对地面 (-Z 方向)
        allegro_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.pi / 2)

        object_start_pose = gymapi.Transform()
        # 【修正】Vec3 只能接收 3 个参数 (x, y, z)，设置球在桌面的高度为 0.04m
        # 对应你之前截图 image_ea4805 中的 TypeError 报错
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.04) 

        # 【保留结构】将偏移量设为 0，确保手和球在水平面中心对齐
        pose_dx, pose_dy, pose_dz = 0.0, 0.0, 0.0 

        # 以下计算逻辑保留，由于偏移量为 0，它们不再干扰位置
        object_start_pose.p.x = allegro_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = allegro_hand_start_pose.p.y + pose_dy
        # 此处 object_start_pose.p.z 的计算被下方的固定高度覆盖

        # 进一步微调 Y 轴（保留原结构）
        object_start_pose.p.y = allegro_hand_start_pose.p.y

        # 【核心修正】强制将球的高度锁定在桌面 (0.04m)
        # 解决了之前 object_z 会被覆盖回 0.65m 的逻辑问题
        object_z = 0.04
        object_start_pose.p.z = object_z
        
        return allegro_hand_start_pose, object_start_pose


@torch.jit.script
def compute_hand_reward(
        object_init_z: torch.Tensor, reset_buf: torch.Tensor, progress_buf: torch.Tensor, max_episode_length: float,
        object_pos: torch.Tensor, palm_pos: torch.Tensor,
        ff_pos: torch.Tensor, mf_pos: torch.Tensor, rf_pos: torch.Tensor, th_pos: torch.Tensor,
        target_pos: torch.Tensor, actions: torch.Tensor, reset_z_threshold: float,
        delta_qpos: torch.Tensor, delta_target_hand_pos: torch.Tensor, delta_target_hand_rot: torch.Tensor
):
    # ==========================================
    # 1. 基础物理距离计算
    # ==========================================
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

    hand_dist = torch.norm(object_pos - palm_pos, p=2, dim=-1)
    hand_dist = torch.where(hand_dist >= 0.5, 0.5 + 0 * hand_dist, hand_dist)

    finger_dist = (
            torch.norm(object_pos - ff_pos, p=2, dim=-1) +
            torch.norm(object_pos - mf_pos, p=2, dim=-1) +
            torch.norm(object_pos - rf_pos, p=2, dim=-1) +
            torch.norm(object_pos - th_pos, p=2, dim=-1)
    )
    finger_dist = torch.where(finger_dist >= 3.0, 3.0 + 0 * finger_dist, finger_dist)

    # ==========================================
    # 2. 【核心】：100% 还原 UniDexGrasp 的模仿学习惩罚公式
    # ==========================================
    # 计算曼哈顿距离(p=1)和旋转误差角度
    delta_hand_pos_value = torch.norm(delta_target_hand_pos, p=1, dim=-1)
    delta_hand_rot_value = 2.0 * torch.asin(
        torch.clamp(torch.norm(delta_target_hand_rot[:, 0:3], p=2, dim=-1), max=1.0))
    delta_qpos_value = torch.norm(delta_qpos, p=1, dim=-1)

    # Uni 官方给定的三种误差的绝对权重比例
    delta_value = 0.3 * delta_hand_pos_value + 0.04 * delta_hand_rot_value + 0.02 * delta_qpos_value

    # ==========================================
    # 3. 状态机门控逻辑与奖励 (0.6 与 0.12 阈值)
    # ==========================================
    flag = (finger_dist <= 0.6).int() + (hand_dist <= 0.12).int()

    lowest = object_pos[:, 2]
    lift_z = object_init_z + 0.005

    goal_hand_rew = torch.zeros_like(finger_dist)
    goal_hand_rew = torch.where(flag == 2, 1.0 * (0.9 - 2.0 * goal_dist), goal_hand_rew)

    hand_up = torch.zeros_like(finger_dist)
    hand_up = torch.where(lowest >= lift_z, torch.where(flag == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)

    bonus = torch.zeros_like(goal_dist)
    bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1.0 + 10.0 * goal_dist), bonus), bonus)

    # ==========================================
    # 4. 汇总总分 (原版扣分系数 -0.5 * delta_value)
    # ==========================================
    reward = -0.5 * finger_dist - 1.0 * hand_dist + goal_hand_rew + hand_up + bonus - 0.5 * delta_value

    resets = torch.where(lowest < reset_z_threshold, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    return reward, resets


def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.
    Adapted from PyTorch3D:
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_axis_angle
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., :3], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., 3:])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., :3] / sin_half_angles_over_angles
