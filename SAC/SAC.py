import os
import random
import time
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
    exp_name: str = "carla_multimodal_sac_ppo_encoder"
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True

    total_timesteps: int = int(1e5)

    # kept from PPO
    encoder_lr: float = 5e-5
    policy_lr: float = 1e-4
    q_lr: float = 1e-4
    alpha_lr: float = 1e-4

    gamma: float = 0.99
    reward_scale: float = 0.1

    log_interval: int = 1000
    save_interval: int = 5000

    save_dir: str = "models_SAC"
    log_dir: str = "runs"

    latent_size: int = 64
    final_layer: int = 512
    num_heads: int = 8

    # SAC-specific
    tau: float = 0.005
    buffer_size: int = 100000
    batch_size: int = 16
    learning_starts: int = 5000
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True

    # CARLA / env params from PPO
    number_of_vehicles: int = 100
    number_of_walkers: int = 0
    display_size: int = 256
    max_past_step: int = 1
    dt: float = 0.1
    port: int = 2000
    town: str = "Town03"
    task_mode: str = "random"
    max_time_episode: int = 1000
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
# Replay Buffer for dict observations
# =========================================================
class ReplayBuffer:
    def __init__(self, buffer_size, obs_b_shape, obs_r_shape, action_dim, device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs_b = np.zeros((buffer_size,) + obs_b_shape, dtype=np.uint8)
        self.obs_r = np.zeros((buffer_size,) + obs_r_shape, dtype=np.uint8)
        self.next_obs_b = np.zeros((buffer_size,) + obs_b_shape, dtype=np.uint8)
        self.next_obs_r = np.zeros((buffer_size,) + obs_r_shape, dtype=np.uint8)

        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(self, obs_b, obs_r, next_obs_b, next_obs_r, action, reward, done):
        self.obs_b[self.ptr] = obs_b
        self.obs_r[self.ptr] = obs_r
        self.next_obs_b[self.ptr] = next_obs_b
        self.next_obs_r[self.ptr] = next_obs_r
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs_b = torch.tensor(self.obs_b[idxs], dtype=torch.float32, device=self.device)
        obs_r = torch.tensor(self.obs_r[idxs], dtype=torch.float32, device=self.device)
        next_obs_b = torch.tensor(self.next_obs_b[idxs], dtype=torch.float32, device=self.device)
        next_obs_r = torch.tensor(self.next_obs_r[idxs], dtype=torch.float32, device=self.device)

        actions = torch.tensor(self.actions[idxs], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.rewards[idxs], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[idxs], dtype=torch.float32, device=self.device)

        return {
            "obs_b": obs_b,
            "obs_r": obs_r,
            "next_obs_b": next_obs_b,
            "next_obs_r": next_obs_r,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }


# =========================================================
# PPO Encoder (kept)
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
# SAC Networks
# =========================================================
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(
        self,
        action_dim,
        birdeye_shape,
        radar_shape,
        final_layer=512,
        latent_size=64,
        num_heads=8,
        depths1=(16, 32, 64),
        depths2=(16, 32, 64),
    ):
        super().__init__()

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
            layer_init(nn.Linear(enc_dim + action_dim, final_layer)),
            nn.ReLU(),
            layer_init(nn.Linear(final_layer, final_layer)),
            nn.ReLU(),
            layer_init(nn.Linear(final_layer, 1), std=1.0),
        )

    def forward(self, x1, x2, action):
        z = self.encoder(x1, x2)
        x = torch.cat([z, action], dim=1)
        return self.q(x)


class Actor(nn.Module):
    def __init__(
        self,
        action_low,
        action_high,
        birdeye_shape,
        radar_shape,
        final_layer=512,
        latent_size=64,
        num_heads=8,
        depths1=(16, 32, 64),
        depths2=(16, 32, 64),
    ):
        super().__init__()

        action_dim = action_low.shape[0]

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
        self.fc = nn.Sequential(
            layer_init(nn.Linear(enc_dim, final_layer)),
            nn.ReLU(),
            layer_init(nn.Linear(final_layer, final_layer)),
            nn.ReLU(),
        )
        self.fc_mean = layer_init(nn.Linear(final_layer, action_dim), std=0.01)
        self.fc_logstd = layer_init(nn.Linear(final_layer, action_dim), std=0.01)

        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x1, x2):
        z = self.encoder(x1, x2)
        z = self.fc(z)
        mean = self.fc_mean(z)
        log_std = self.fc_logstd(z)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, x1, x2):
        mean, log_std = self(x1, x2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def save(self, path):
        torch.save(self.state_dict(), path)


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
        "discrete": False,
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
        "continuous_accel_range": [-.5, 2],
        "continuous_steer_range": [-.3, .3],
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

    assert isinstance(env.action_space, gym.spaces.Box), "SAC requires continuous action space"

    initial_obs = reset_env(env, args.seed)
    obs_b_shape = initial_obs["birdeye"].shape
    obs_r_shape = initial_obs["radar/lidar"].shape
    action_dim = int(np.prod(env.action_space.shape))
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    print("birdeye shape:", obs_b_shape)
    print("radar/lidar shape:", obs_r_shape)
    print("action shape:", env.action_space.shape)

    actor = Actor(
        action_low=action_low,
        action_high=action_high,
        birdeye_shape=obs_b_shape,
        radar_shape=obs_r_shape,
        final_layer=args.final_layer,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
    ).to(device)

    qf1 = SoftQNetwork(
        action_dim=action_dim,
        birdeye_shape=obs_b_shape,
        radar_shape=obs_r_shape,
        final_layer=args.final_layer,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
    ).to(device)

    qf2 = SoftQNetwork(
        action_dim=action_dim,
        birdeye_shape=obs_b_shape,
        radar_shape=obs_r_shape,
        final_layer=args.final_layer,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
    ).to(device)

    qf1_target = SoftQNetwork(
        action_dim=action_dim,
        birdeye_shape=obs_b_shape,
        radar_shape=obs_r_shape,
        final_layer=args.final_layer,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
    ).to(device)

    qf2_target = SoftQNetwork(
        action_dim=action_dim,
        birdeye_shape=obs_b_shape,
        radar_shape=obs_r_shape,
        final_layer=args.final_layer,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
    ).to(device)

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    print("Actor encoder spatial size:", actor.encoder.dim_h, actor.encoder.dim_w)
    print("Q1 encoder spatial size:", qf1.encoder.dim_h, qf1.encoder.dim_w)

    actor_encoder_params = list(actor.encoder.parameters())
    actor_head_params = (
        list(actor.fc.parameters()) +
        list(actor.fc_mean.parameters()) +
        list(actor.fc_logstd.parameters())
    )

    q_encoder_params = list(qf1.encoder.parameters()) + list(qf2.encoder.parameters())
    q_head_params = (
        list(qf1.q.parameters()) +
        list(qf2.q.parameters())
    )

    actor_optimizer = optim.Adam(
        [
            {"params": actor_encoder_params, "lr": args.encoder_lr},
            {"params": actor_head_params, "lr": args.policy_lr},
        ]
    )

    q_optimizer = optim.Adam(
        [
            {"params": q_encoder_params, "lr": args.encoder_lr},
            {"params": q_head_params, "lr": args.q_lr},
        ]
    )

    if args.autotune:
        target_entropy = -float(action_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_b_shape=obs_b_shape,
        obs_r_shape=obs_r_shape,
        action_dim=action_dim,
        device=device,
    )

    global_step = 0
    start_time = time.time()

    obs = initial_obs
    episode_return = 0.0
    episode_length = 0

    for global_step in trange(args.total_timesteps):
        # action selection
        if global_step < args.learning_starts:
            action = env.action_space.sample().astype(np.float32)
        else:
            with torch.no_grad():
                obs_b_t, obs_r_t = obs_to_tensors(obs, device)
                action_t, _, _ = actor.get_action(
                    obs_b_t.unsqueeze(0),
                    obs_r_t.unsqueeze(0),
                )
                action = action_t.squeeze(0).cpu().numpy().astype(np.float32)

        # env step
        next_obs, reward, done, info = step_env(env, action)

        scaled_reward = args.reward_scale * reward
        episode_return += reward
        episode_length += 1

        terminal = done or (episode_length >= args.max_time_episode)

        rb.add(
            obs["birdeye"],
            obs["radar/lidar"],
            next_obs["birdeye"],
            next_obs["radar/lidar"],
            action,
            scaled_reward,
            float(terminal),
        )

        obs = next_obs


        if terminal:
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

        

        # training
        if global_step > args.learning_starts and rb.size >= args.batch_size:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                next_actions, next_log_pi, _ = actor.get_action(
                    data["next_obs_b"], data["next_obs_r"]
                )
                qf1_next_target = qf1_target(
                    data["next_obs_b"], data["next_obs_r"], next_actions
                )
                qf2_next_target = qf2_target(
                    data["next_obs_b"], data["next_obs_r"], next_actions
                )
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_pi
                next_q_value = data["rewards"] + (1.0 - data["dones"]) * args.gamma * min_qf_next_target

            qf1_a_values = qf1(data["obs_b"], data["obs_r"], data["actions"])
            qf2_a_values = qf2(data["obs_b"], data["obs_r"], data["actions"])

            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                pi, log_pi, _ = actor.get_action(data["obs_b"], data["obs_r"])
                qf1_pi = qf1(data["obs_b"], data["obs_r"], pi)
                qf2_pi = qf2(data["obs_b"], data["obs_r"], pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = (alpha * log_pi - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi_alpha, _ = actor.get_action(data["obs_b"], data["obs_r"])
                    alpha_loss = (-log_alpha.exp() * (log_pi_alpha + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("charts/actor_encoder_lr", actor_optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("charts/policy_lr", actor_optimizer.param_groups[1]["lr"], global_step)
                writer.add_scalar("charts/q_encoder_lr", q_optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("charts/q_lr", q_optimizer.param_groups[1]["lr"], global_step)

                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)

                if global_step % args.policy_frequency == 0:
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

                sps = int(global_step / (time.time() - start_time + 1e-8))
                writer.add_scalar("charts/SPS", sps, global_step)

                print(
                    f"step={global_step}, "
                    f"qf1_loss={qf1_loss.item():.6f}, "
                    f"qf2_loss={qf2_loss.item():.6f}, "
                    f"alpha={alpha:.6f}, SPS={sps}"
                )

        if global_step % args.save_interval == 0 and global_step > 0:
            save_path = os.path.join(args.save_dir, f"{run_name}_{global_step}.pth")
            print(f"saving actor to {save_path}")
            actor.save(save_path)

    final_path = os.path.join(args.save_dir, f"{run_name}_final.pth")
    print(f"saving final actor to {final_path}")
    actor.save(final_path)

    env.close()
    writer.close()