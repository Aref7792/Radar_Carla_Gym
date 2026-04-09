import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gym
import gym_carla
import carla
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import einops
import tyro
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch

msgpack_numpy_patch()


# =========================================================
# Args
# =========================================================
@dataclass
class Args:
    exp_name: str = "carla_multimodal_dqn_sac_template"
    seed: int = 20
    torch_deterministic: bool = True
    cuda: bool = True

    total_timesteps: int = int(10e5)

    # DQN learning params
    encoder_lr: float = 5e-5
    q_lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 32
    buffer_size: int = int(5e4)
    learning_starts: int = 10000
    train_frequency: int = 4
    target_network_frequency: int = 2500
    max_grad_norm: float = 10.0

    # epsilon-greedy
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = int(10e5)

    # logging / saving
    log_interval: int = 5000
    save_interval: int = 10000
    save_dir: str = "DQN_models"
    log_dir: str = "logs/MM_DQN"

    # encoder / network
    latent_size: int = 64
    final_layer: int = 512
    num_heads: int = 8

    # CARLA / env params
    number_of_vehicles: int = 40
    number_of_walkers: int = 0
    display_size: int = 256
    max_past_step: int = 1
    dt: float = 0.1
    port: int = 2000
    town: str = "Town03"
    task_mode: str = "random"
    max_time_episode: int = 500
    max_waypt: int = 12
    obs_range: int = 32
    lidar_bin: float = 0.5
    d_behind: int = 12
    out_lane_thres: float = 2.0
    desired_speed: int = 8
    max_ego_spawn_times: int = 200
    pixor_size: int = 64
    pixor: bool = False
    use_radar: bool = True
    radar_height: float = 1.0
    radar_x: float = 2.0
    radar_hfov: int = 60
    radar_vfov: int = 20
    radar_range: int = 32
    radar_pps: int = 3000
    radar_vmax: float = 30.0
    render_panels: int = 3
    render: bool = False
    enable_pygame: bool = False
    frame_stack: int = 1


# =========================================================
# Utils
# =========================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if hasattr(layer, "weight") and layer.weight is not None:
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / float(duration)
    return max(slope * t + start_e, end_e)


def reset_env(env, seed=None):
    if seed is None:
        out = env.reset()
    else:
        try:
            out = env.reset(seed=seed)
        except TypeError:
            out = env.reset()

    if isinstance(out, tuple):
        obs, _info = out
        return obs
    return out


def step_env(env, action):
    if isinstance(action, torch.Tensor):
        action = int(action.item())
    elif isinstance(action, np.ndarray):
        action = int(action.item())

    out = env.step(action)
    if len(out) == 4:
        obs, rew, done, info = out
        return obs, rew, done, info
    elif len(out) == 5:
        obs, rew, terminated, truncated, info = out
        return obs, rew, (terminated or truncated), info
    raise ValueError("Unexpected env.step output format")


def obs_to_tensors(obs, device):
    birdeye = torch.as_tensor(obs["birdeye"], dtype=torch.float32, device=device)
    radar = torch.as_tensor(obs["radar/lidar"], dtype=torch.float32, device=device)
    return birdeye, radar


# =========================================================
# Replay Buffer
# =========================================================
class ReplayBuffer:
    def __init__(self, buffer_size, obs_b_shape, obs_r_shape, device):
        self.buffer_size = int(buffer_size)
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs_b = np.zeros((self.buffer_size,) + obs_b_shape, dtype=np.uint8)
        self.obs_r = np.zeros((self.buffer_size,) + obs_r_shape, dtype=np.uint8)
        self.next_obs_b = np.zeros((self.buffer_size,) + obs_b_shape, dtype=np.uint8)
        self.next_obs_r = np.zeros((self.buffer_size,) + obs_r_shape, dtype=np.uint8)

        self.actions = np.zeros((self.buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.ks = np.ones((self.buffer_size, 1), dtype=np.float32)

    def add(self, obs_b, obs_r, next_obs_b, next_obs_r, action, reward, done, k=1):
        self.obs_b[self.ptr] = obs_b
        self.obs_r[self.ptr] = obs_r
        self.next_obs_b[self.ptr] = next_obs_b
        self.next_obs_r[self.ptr] = next_obs_r
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = float(done)
        self.ks[self.ptr] = float(k)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs_b = torch.tensor(self.obs_b[idxs], dtype=torch.float32, device=self.device)
        obs_r = torch.tensor(self.obs_r[idxs], dtype=torch.float32, device=self.device)
        next_obs_b = torch.tensor(self.next_obs_b[idxs], dtype=torch.float32, device=self.device)
        next_obs_r = torch.tensor(self.next_obs_r[idxs], dtype=torch.float32, device=self.device)

        actions = torch.tensor(self.actions[idxs], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(self.rewards[idxs], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[idxs], dtype=torch.float32, device=self.device)
        ks = torch.tensor(self.ks[idxs], dtype=torch.float32, device=self.device)

        return {
            "obs_b": obs_b,
            "obs_r": obs_r,
            "next_obs_b": next_obs_b,
            "next_obs_r": next_obs_r,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "ks": ks,
        }


# =========================================================
# PPO-style Encoder kept from SAC template
# =========================================================
class EncoderBlock(nn.Module):
    def __init__(self, latent_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_size)
        self.norm2 = nn.LayerNorm(latent_size)
        self.norm3 = nn.LayerNorm(latent_size)

        self.multihead = nn.MultiheadAttention(
            latent_size, num_heads, batch_first=True
        )

        self.enc_mlp = nn.Sequential(
            layer_init(nn.Linear(latent_size, latent_size * 4)),
            nn.GELU(),
            layer_init(nn.Linear(latent_size * 4, latent_size)),
        )

    def forward(self, embedded_patches1, embedded_patches2):
        q = self.norm1(embedded_patches1)
        v = self.norm2(embedded_patches2)
        attention_out = self.multihead(q, v, v)[0]
        x = embedded_patches1 + attention_out
        y = self.norm3(x)
        y = self.enc_mlp(y)
        return x + y


class InputEmbedding(nn.Module):
    def __init__(self, n_channels, latent_size, dim1, dim2):
        super().__init__()
        self.linear_projection = layer_init(nn.Linear(n_channels, latent_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, dim1, dim2))

    def forward(self, input_data):
        patches = einops.rearrange(input_data, "b c h w -> b h w c")
        patches = patches + self.pos_embedding.unsqueeze(-1)
        patches = einops.rearrange(patches, "b h w c -> b (h w) c")
        return self.linear_projection(patches)


class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        birdeye_shape,
        radar_shape,
        latent_size=64,
        num_heads=8,
        depths1=(16, 32, 64),
        depths2=(16, 32, 64),
        radar_div_255=True,
    ):
        super().__init__()

        n_input_channels1 = birdeye_shape[-1]
        n_input_channels2 = radar_shape[-1]

        h1, w1 = birdeye_shape[0], birdeye_shape[1]
        h2, w2 = radar_shape[0], radar_shape[1]

        dim1_h, dim1_w = self._conv_output_hw(h1, w1)
        dim2_h, dim2_w = self._conv_output_hw(h2, w2)

        if (dim1_h != dim2_h) or (dim1_w != dim2_w):
            raise ValueError(
                f"Birdeye and radar conv outputs must match spatially. "
                f"Got birdeye=({dim1_h}, {dim1_w}) and radar=({dim2_h}, {dim2_w})."
            )

        self.dim_h = dim1_h
        self.dim_w = dim1_w
        self.latent_size = latent_size
        self.radar_div_255 = radar_div_255

        self.bev_conv = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels1, depths1[0], kernel_size=5, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(depths1[0], depths1[1], kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(depths1[1], depths1[2], kernel_size=3, stride=1)),
            nn.ReLU(),
        )
        self.bev_emb = InputEmbedding(
            n_channels=depths1[2],
            latent_size=latent_size,
            dim1=self.dim_h,
            dim2=self.dim_w,
        )

        self.radar_conv = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels2, depths2[0], kernel_size=5, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(depths2[0], depths2[1], kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(depths2[1], depths2[2], kernel_size=3, stride=1)),
            nn.ReLU(),
        )
        self.radar_emb = InputEmbedding(
            n_channels=depths2[2],
            latent_size=latent_size,
            dim1=self.dim_h,
            dim2=self.dim_w,
        )

        self.cross = EncoderBlock(latent_size, num_heads)
        self.flatten = nn.Flatten()
        self.output_dim = latent_size * self.dim_h * self.dim_w

    @staticmethod
    def _conv2d_out(size, kernel, stride, padding=0, dilation=1):
        return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

    def _conv_output_hw(self, h, w):
        h = self._conv2d_out(h, 5, 2)
        w = self._conv2d_out(w, 5, 2)

        h = self._conv2d_out(h, 3, 2)
        w = self._conv2d_out(w, 3, 2)

        h = self._conv2d_out(h, 3, 1)
        w = self._conv2d_out(w, 3, 1)
        return h, w

    def _normalize_birdeye(self, x):
        return x.float() / 255.0

    def _normalize_radar(self, x):
        if self.radar_div_255:
            return x.float() / 255.0
        return x.float()

    def forward(self, x1, x2):
        x1 = self._normalize_birdeye(x1.permute(0, 3, 1, 2))
        x2 = self._normalize_radar(x2.permute(0, 3, 1, 2))

        bev_feat = self.bev_conv(x1)
        radar_feat = self.radar_conv(x2)

        bev_tokens = self.bev_emb(bev_feat)
        radar_tokens = self.radar_emb(radar_feat)

        fused = self.cross(radar_tokens, bev_tokens)
        fused = einops.rearrange(
            fused, "b (h w) c -> b c h w", h=self.dim_h, w=self.dim_w
        )
        flat = self.flatten(fused)
        return flat


# =========================================================
# DQN Network
# =========================================================
class QNetwork(nn.Module):
    def __init__(
        self,
        num_actions,
        birdeye_shape,
        radar_shape,
        final_layer=512,
        latent_size=64,
        num_heads=8,
        depths1=(16, 32, 64),
        depths2=(16, 32, 64),
    ):
        super().__init__()

        self.num_actions = num_actions

        self.encoder = MultiModalEncoder(
            birdeye_shape=birdeye_shape,
            radar_shape=radar_shape,
            latent_size=latent_size,
            num_heads=num_heads,
            depths1=depths1,
            depths2=depths2,
            radar_div_255=True,
        )

        enc_dim = self.encoder.output_dim
        self.q = nn.Sequential(
            layer_init(nn.Linear(enc_dim, final_layer)),
            nn.ReLU(),
            layer_init(nn.Linear(final_layer, num_actions), std=1.0),
        )

    def forward(self, x1, x2):
        z = self.encoder(x1, x2)
        return self.q(z)

    def act(self, obs, epsilon, device):
        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            return None

        with torch.no_grad():
            obses_b = torch.as_tensor(obs["birdeye"], dtype=torch.float32, device=device).unsqueeze(0)
            obses_r = torch.as_tensor(obs["radar/lidar"], dtype=torch.float32, device=device).unsqueeze(0)
            q_values = self(obses_b, obses_r)
            action = torch.argmax(q_values, dim=1).item()
        return int(action)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))


# =========================================================
# Environment factory
# =========================================================
def make_env(args):
    params = {
        "number_of_vehicles": args.number_of_vehicles,
        "number_of_walkers": args.number_of_walkers,
        "display_size": args.display_size,
        "max_past_step": args.max_past_step,
        "dt": args.dt,
        "discrete": True,
        "discrete_acc": [
            (0.00, 0.00),
            (0.25, 0.00),
            (0.40, 0.00),
            (0.60, 0.00),
            (0.80, 0.00),
            (1.00, 0.00),
            (0.00, 0.20),
            (0.00, 0.50),
        ],
        "discrete_steer": [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20],
        "continuous_accel_range": [-3.0, 3.0],
        "continuous_steer_range": [-0.3, 0.3],
        "ego_vehicle_filter": "vehicle.lincoln*",
        "port": args.port,
        "town": args.town,
        "task_mode": args.task_mode,
        "max_time_episode": args.max_time_episode,
        "max_waypt": args.max_waypt,
        "obs_range": args.obs_range,
        "lidar_bin": args.lidar_bin,
        "d_behind": args.d_behind,
        "out_lane_thres": args.out_lane_thres,
        "desired_speed": args.desired_speed,
        "max_ego_spawn_times": args.max_ego_spawn_times,
        "display_route": True,
        "pixor_size": args.pixor_size,
        "pixor": args.pixor,
        "use_radar": args.use_radar,
        "radar_height": args.radar_height,
        "radar_x": args.radar_x,
        "radar_hfov": args.radar_hfov,
        "radar_vfov": args.radar_vfov,
        "radar_range": args.radar_range,
        "radar_pps": args.radar_pps,
        "radar_vmax": args.radar_vmax,
        "render_panels": args.render_panels,
        "render": args.render,
        "enable_pygame": args.enable_pygame,
        "frame_stack": args.frame_stack,
    }
    return gym.make("carla-v0", params=params), params


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    args = tyro.cli(Args)

    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(args.log_dir, run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])
        ),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = not args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env, env_params = make_env(args)
    try:
        env.action_space.seed(args.seed)
    except Exception:
        pass

    assert isinstance(env.action_space, gym.spaces.Discrete), "DQN requires discrete action space"

    initial_obs = reset_env(env, args.seed)
    obs_b_shape = initial_obs["birdeye"].shape
    obs_r_shape = initial_obs["radar/lidar"].shape
    num_actions = env.action_space.n

    print("birdeye shape:", obs_b_shape)
    print("radar/lidar shape:", obs_r_shape)
    print("num actions:", num_actions)

    q_network = QNetwork(
        num_actions=num_actions,
        birdeye_shape=obs_b_shape,
        radar_shape=obs_r_shape,
        final_layer=args.final_layer,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
    ).to(device)

    target_network = QNetwork(
        num_actions=num_actions,
        birdeye_shape=obs_b_shape,
        radar_shape=obs_r_shape,
        final_layer=args.final_layer,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
    ).to(device)

    target_network.load_state_dict(q_network.state_dict())

    print("Q encoder spatial size:", q_network.encoder.dim_h, q_network.encoder.dim_w)

    encoder_params = list(q_network.encoder.parameters())
    q_head_params = list(q_network.q.parameters())

    optimizer = optim.Adam(
        [
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": q_head_params, "lr": args.q_lr},
        ]
    )

    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_b_shape=obs_b_shape,
        obs_r_shape=obs_r_shape,
        device=device,
    )

    epinfos_buffer = deque(maxlen=100)
    start_time = time.time()

    obs = initial_obs
    episode_return = 0.0
    episode_length = 0
    episode_count = 0

    # -----------------------------------------------------
    # Replay warmup
    # -----------------------------------------------------
    for _ in trange(args.learning_starts, desc="Filling replay buffer"):
        action = env.action_space.sample()
        next_obs, reward, done, info = step_env(env, action)
        reward = .1 *reward

        k = info["action_repeat"] if isinstance(info, dict) and "action_repeat" in info else 1

        rb.add(
            obs["birdeye"],
            obs["radar/lidar"],
            next_obs["birdeye"],
            next_obs["radar/lidar"],
            action,
            reward,
            float(done),
            k=k,
        )

        obs = next_obs
        if done:
            obs = reset_env(env)

    obs = reset_env(env)

    # -----------------------------------------------------
    # Main training loop
    # -----------------------------------------------------
    for global_step in trange(args.total_timesteps):
        epsilon = linear_schedule(
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay_steps,
            global_step,
        )

        action = q_network.act(obs, epsilon, device)
        if action is None:
            action = env.action_space.sample()

        next_obs, reward, done, info = step_env(env, action)
        reward = .1 * reward

        k = info["action_repeat"] if isinstance(info, dict) and "action_repeat" in info else 1

        episode_return += reward
        episode_length += 1

        terminal = done or (episode_length >= args.max_time_episode)

        rb.add(
            obs["birdeye"],
            obs["radar/lidar"],
            next_obs["birdeye"],
            next_obs["radar/lidar"],
            action,
            reward,
            float(terminal),
            k=k,
        )

        obs = next_obs

        if terminal:
            epinfos_buffer.append(
                {"r": round(episode_return, 6), "l": episode_length}
            )

            writer.add_scalar("charts/episodic_return", episode_return, global_step)
            writer.add_scalar("charts/episodic_length", episode_length, global_step)

            print(
                f"global_step={global_step}, "
                f"episodic_return={episode_return:.4f}, "
                f"episodic_length={episode_length}"
            )

            obs = reset_env(env)
            episode_return = 0.0
            episode_length = 0
            episode_count += 1

        # training
        if rb.size >= args.batch_size and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                target_q_values = target_network(data["next_obs_b"], data["next_obs_r"])
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
                td_target = (
                    data["rewards"]
                    + (args.gamma) * (1.0 - data["dones"]) * max_target_q_values
                )

            current_q_values = q_network(data["obs_b"], data["obs_r"])
            chosen_q_values = current_q_values.gather(1, data["actions"])

            loss = F.smooth_l1_loss(chosen_q_values, td_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
            optimizer.step()

        # target network update
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

        # logging
        if global_step % args.log_interval == 0:
            rew_mean = np.mean([e["r"] for e in epinfos_buffer]) if len(epinfos_buffer) > 0 else 0.0
            len_mean = np.mean([e["l"] for e in epinfos_buffer]) if len(epinfos_buffer) > 0 else 0.0
            sps = int((global_step + 1) / (time.time() - start_time + 1e-8))

            print()
            print(f"Step: {global_step}")
            print(f"Epsilon: {epsilon:.4f}")
            print(f"Avg Rew: {rew_mean:.6f}")
            print(f"Avg Ep Len: {len_mean:.6f}")
            print(f"Episode: {episode_count}")
            print(f"SPS: {sps}")

            writer.add_scalar("charts/epsilon", epsilon, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/avg_100ep_return", rew_mean, global_step)
            writer.add_scalar("charts/avg_100ep_length", len_mean, global_step)
            writer.add_scalar("charts/encoder_lr", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/q_lr", optimizer.param_groups[1]["lr"], global_step)

            if rb.size >= args.batch_size and "loss" in locals():
                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar("losses/q_mean", chosen_q_values.mean().item(), global_step)
                writer.add_scalar("losses/target_q_mean", td_target.mean().item(), global_step)

        # save
        if global_step % args.save_interval == 0 and global_step > 0:
            save_path = os.path.join(args.save_dir, f"{run_name}_{global_step}.pth")
            print(f"saving q_network to {save_path}")
            q_network.save(save_path)

    final_path = os.path.join(args.save_dir, f"{run_name}_final.pth")
    print(f"saving final q_network to {final_path}")
    q_network.save(final_path)

    env.close()
    writer.close()