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
import torch.optim as optim
import einops
import tyro
from torch.distributions.categorical import Categorical
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
    exp_name: str = "carla_multimodal_ppo_separate_encoders"
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True

    total_timesteps: int = int(524288)

    encoder_lr: float = 5e-5
    policy_lr: float = 1e-4
    value_lr: float = 1e-4
    anneal_lr: bool = False

    gamma: float = 0.99
    gae_lambda: float = 0.95

    num_steps: int = 256
    update_epochs: int = 8
    num_minibatches: int = 4

    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.01

    reward_scale: float = 0.1

    log_interval: int = 1000
    save_interval: int = 5000

    save_dir: str = "models_PPO"
    log_dir: str = "runs"

    latent_size: int = 64
    final_layer: int = 512
    num_heads: int = 8

    number_of_vehicles: int = 40
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

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


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
    radar = torch.as_tensor(obs["radar"], dtype=torch.float32, device=device)
    return birdeye, radar


# =========================================================
# Model blocks
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
# Separate Actor/Critic Encoders Agent
# =========================================================
class Agent(nn.Module):
    def __init__(
        self,
        action_dim,
        birdeye_shape,
        radar_shape,
        depths1=(16, 32, 64),
        depths2=(16, 32, 64),
        final_layer=512,
        latent_size=64,
        num_heads=8,
    ):
        super().__init__()
        self.action_dim = action_dim

        self.actor_encoder = MultiModalEncoder(
            birdeye_shape=birdeye_shape,
            radar_shape=radar_shape,
            latent_size=latent_size,
            num_heads=num_heads,
            depths1=depths1,
            depths2=depths2,
            radar_div_255=True,
        )

        self.critic_encoder = MultiModalEncoder(
            birdeye_shape=birdeye_shape,
            radar_shape=radar_shape,
            latent_size=latent_size,
            num_heads=num_heads,
            depths1=depths1,
            depths2=depths2,
            radar_div_255=True,
        )

        self.dim_h = self.actor_encoder.dim_h
        self.dim_w = self.actor_encoder.dim_w

        actor_dim = self.actor_encoder.output_dim
        critic_dim = self.critic_encoder.output_dim

        self.actor_fc = nn.Sequential(
            layer_init(nn.Linear(actor_dim, final_layer)),
            nn.ReLU(),
        )
        self.critic_fc = nn.Sequential(
            layer_init(nn.Linear(critic_dim, final_layer)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(final_layer, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(final_layer, 1), std=1.0)

    def get_value(self, x1, x2):
        critic_common = self.critic_encoder(x1, x2)
        critic_feat = self.critic_fc(critic_common)
        return self.critic(critic_feat).squeeze(-1)

    def get_action_and_value(self, x1, x2, action=None):
        actor_common = self.actor_encoder(x1, x2)
        critic_common = self.critic_encoder(x1, x2)

        actor_feat = self.actor_fc(actor_common)
        critic_feat = self.critic_fc(critic_common)

        logits = self.actor(actor_feat)
        value = self.critic(critic_feat).squeeze(-1)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value

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
    args.batch_size = args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

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

    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    initial_obs = reset_env(env, args.seed)
    obs_b_shape = initial_obs["birdeye"].shape
    obs_r_shape = initial_obs["radar"].shape

    print("birdeye shape:", obs_b_shape)
    print("radar shape:", obs_r_shape)

    agent = Agent(
        action_dim=env.action_space.n,
        birdeye_shape=obs_b_shape,
        radar_shape=obs_r_shape,
        depths1=(16, 32, 64),
        depths2=(16, 32, 64),
        final_layer=args.final_layer,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
    ).to(device)

    print("Actor encoder spatial size:", agent.actor_encoder.dim_h, agent.actor_encoder.dim_w)
    print("Critic encoder spatial size:", agent.critic_encoder.dim_h, agent.critic_encoder.dim_w)

    actor_encoder_params = list(agent.actor_encoder.parameters())
    actor_head_params = list(agent.actor_fc.parameters()) + list(agent.actor.parameters())
    critic_params = (
        list(agent.critic_encoder.parameters()) +
        list(agent.critic_fc.parameters()) +
        list(agent.critic.parameters())
    )

    optimizer = optim.Adam(
        [
            {"params": actor_encoder_params, "lr": args.encoder_lr},
            {"params": actor_head_params, "lr": args.policy_lr},
            {"params": critic_params, "lr": args.value_lr},
        ],
        eps=1e-5,
    )

    obs_b = torch.zeros((args.num_steps,) + obs_b_shape, dtype=torch.float32, device=device)
    obs_r = torch.zeros((args.num_steps,) + obs_r_shape, dtype=torch.float32, device=device)
    actions = torch.zeros((args.num_steps,), dtype=torch.long, device=device)
    logprobs = torch.zeros((args.num_steps,), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.num_steps,), dtype=torch.float32, device=device)
    dones = torch.zeros((args.num_steps,), dtype=torch.float32, device=device)
    values = torch.zeros((args.num_steps,), dtype=torch.float32, device=device)

    global_step = 0
    start_time = time.time()
    next_obs = initial_obs
    next_obs_b, next_obs_r = obs_to_tensors(next_obs, device)
    #next_done = torch.tensor(0.0, device=device)

    episode_return = 0.0
    episode_length = 0

    for iteration in trange(1, args.num_iterations + 1, desc="PPO Iterations"):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.encoder_lr
            optimizer.param_groups[1]["lr"] = frac * args.policy_lr
            optimizer.param_groups[2]["lr"] = frac * args.value_lr

        for step in trange(args.num_steps, desc="Rollout", leave=False):
            global_step += 1

            obs_b[step] = next_obs_b
            obs_r[step] = next_obs_r
            #dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs_b.unsqueeze(0),
                    next_obs_r.unsqueeze(0),
                )
                values[step] = value.squeeze(0)

            actions[step] = action.squeeze(0)
            logprobs[step] = logprob.squeeze(0)

            next_obs, reward, done, info = step_env(env, action.item())
            

            scaled_reward = args.reward_scale * reward
            rewards[step] = scaled_reward

            episode_return += scaled_reward
            episode_length += 1

            terminal = done or (episode_length >= args.max_time_episode)
            next_done = torch.tensor(float(terminal), dtype=torch.float32, device=device)
            dones[step] = next_done

            if terminal:
                writer.add_scalar("charts/episodic_return", episode_return, global_step)
                writer.add_scalar("charts/episodic_length", episode_length, global_step)
                print(
                    f"global_step={global_step}, "
                    f"episodic_return={episode_return:.4f}, "
                    f"episodic_length={episode_length}"
                )
                next_obs = reset_env(env)
                episode_return = 0.0
                episode_length = 0
                #next_done = torch.tensor(0.0, dtype=torch.float32, device=device)

            next_obs_b, next_obs_r = obs_to_tensors(next_obs, device)

        with torch.no_grad():
            next_value = agent.get_value(
                next_obs_b.unsqueeze(0),
                next_obs_r.unsqueeze(0)
            ).squeeze(0)

        advantages = torch.zeros_like(rewards, device=device)
        lastgaelam = 0.0

        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                next_nonterminal = 1.0 - next_done
                next_values = next_value
            else:
                next_nonterminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]

            delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values

        b_obs_b = obs_b.reshape((-1,) + obs_b_shape)
        b_obs_r = obs_r.reshape((-1,) + obs_r_shape)
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []

        approx_kl = torch.tensor(0.0, device=device)
        old_approx_kl = torch.tensor(0.0, device=device)
        pg_loss = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs_b[mb_inds],
                    b_obs_r[mb_inds],
                    b_actions[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (
                        (mb_advantages - mb_advantages.mean()) /
                        (mb_advantages.std(unbiased=False) + 1e-8)
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl.item() > args.target_kl:
                break

        with torch.no_grad():
            pred_values = agent.get_value(b_obs_b, b_obs_r).detach().cpu().numpy()

        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - pred_values) / var_y

        writer.add_scalar("charts/actor_encoder_lr", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/policy_lr", optimizer.param_groups[1]["lr"], global_step)
        writer.add_scalar("charts/value_lr", optimizer.param_groups[2]["lr"], global_step)

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs) if len(clipfracs) > 0 else 0.0, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        if global_step % args.log_interval < args.num_steps:
            print(f"iteration={iteration}, global_step={global_step}, SPS={sps}")
            print(
                f"value_loss={v_loss.item():.6f}, "
                f"policy_loss={pg_loss.item():.6f}, "
                f"entropy={entropy_loss.item():.6f}"
            )
            print(
                f"approx_kl={approx_kl.item():.6f}, "
                f"clipfrac={np.mean(clipfracs) if len(clipfracs) > 0 else 0.0:.6f}, "
                f"explained_var={explained_var:.6f}"
            )

        if global_step % args.save_interval < args.num_steps and global_step > 0:
            save_path = os.path.join(args.save_dir, f"{run_name}_{global_step}.pth")
            print(f"saving model to {save_path}")
            agent.save(save_path)

    final_path = os.path.join(args.save_dir, f"{run_name}_final.pth")
    print(f"saving final model to {final_path}")
    agent.save(final_path)

    env.close()
    writer.close()