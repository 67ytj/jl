@torch.jit.script
def compute_hand_reward(
        object_init_z: torch.Tensor, reset_buf: torch.Tensor, progress_buf: torch.Tensor, max_episode_length: float,
        object_pos: torch.Tensor, palm_pos: torch.Tensor,
        ff_pos: torch.Tensor, mf_pos: torch.Tensor, rf_pos: torch.Tensor, th_pos: torch.Tensor,
        target_pos: torch.Tensor, actions: torch.Tensor, reset_z_threshold: float,
        delta_qpos: torch.Tensor, delta_target_hand_pos: torch.Tensor, delta_target_hand_rot: torch.Tensor
):
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    goal_hand_dist = torch.norm(target_pos - palm_pos, p=2, dim=-1)

    hand_dist = torch.norm(object_pos - palm_pos, p=2, dim=-1)
    hand_dist = torch.where(hand_dist >= 0.5, 0.5 + 0 * hand_dist, hand_dist)

    finger_dist = (
            torch.norm(object_pos - ff_pos, p=2, dim=-1) +
            torch.norm(object_pos - mf_pos, p=2, dim=-1) +
            torch.norm(object_pos - rf_pos, p=2, dim=-1) +
            torch.norm(object_pos - th_pos, p=2, dim=-1)
    )
    finger_dist = torch.where(finger_dist >= 3.0, 3.0 + 0 * finger_dist, finger_dist)

    delta_hand_pos_value = torch.norm(delta_target_hand_pos, p=1, dim=-1)
    delta_hand_rot_value = 2.0 * torch.asin(
        torch.clamp(torch.norm(delta_target_hand_rot[:, 0:3], p=2, dim=-1), max=1.0))
    delta_qpos_value = torch.norm(delta_qpos, p=1, dim=-1)

    # 权重改为与 UniDexGrasp2 一致
    delta_value = 0.6 * delta_hand_pos_value + 0.04 * delta_hand_rot_value + 0.1 * delta_qpos_value

    flag = (finger_dist <= 0.6).int() + (hand_dist <= 0.12).int()

    lowest = object_pos[:, 2]
    # lift_z 改为与 UniDexGrasp2 一致
    lift_z = object_init_z + 0.6 + 0.003

    goal_hand_rew = torch.zeros_like(finger_dist)
    goal_hand_rew = torch.where(flag == 2, 1.0 * (0.9 - 2.0 * goal_dist), goal_hand_rew)

    # hand_up 改为两档，与 UniDexGrasp2 的 else 分支一致
    hand_up = torch.zeros_like(finger_dist)
    hand_up = torch.where(lowest >= lift_z, torch.where(flag == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
    hand_up = torch.where(lowest >= 0.80, torch.where(flag == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

    bonus = torch.zeros_like(goal_dist)
    bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1.0 + 10.0 * goal_dist), bonus), bonus)

    reward = -0.5 * finger_dist - 1.0 * hand_dist + goal_hand_rew + hand_up + bonus - 0.5 * delta_value

    resets = torch.where(lowest < reset_z_threshold, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    return reward, resets