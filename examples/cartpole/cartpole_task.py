import bdb
from dataclasses import dataclass
import signal
import sys
import textwrap
from threading import Thread
import traceback
from typing import Dict, Literal
import gymnasium as gym

import flax.linen as nn
import jax
import optax
import numpy as np
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOBatch
from ksim.task.rl import RLTask
import xax
from jaxtyping import PRNGKeyArray, Array, PyTree
import jax.numpy as jnp
from dpshdl.dataset import Dataset
from ksim.task.ppo import PPOConfig, PPOTask
from brax.base import System
from brax.envs.base import State as BraxState
import equinox as eqx


# Helper function: compute discounted cumulative sum for GAE.
def discount_cumsum_gae(delta: Array, mask: Array, gamma: float, lam: float) -> Array:
    """Computes the discounted cumulative sums of deltas for Generalized Advantage Estimation.

    Args:
        delta: The deltas to compute the discounted cumulative sum of (time, envs).
        mask: The mask to compute the discounted cumulative sum of (time, envs).
        gamma: The discount factor.
        lam: The GAE lambda parameter.
    Returns:
        The discounted cumulative sum of the deltas.
    """

    def scan_fn(carry: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        d, m = x
        new_carry = d + gamma * lam * m * carry
        return new_carry, new_carry

    # Reverse time axis, scan, then reverse back.
    _, out = jax.lax.scan(scan_fn, jnp.zeros_like(delta[-1]), (delta[::-1], mask[::-1]))
    return out[::-1]


class ActorCriticModel(nn.Module):
    """Actor-Critic model."""

    actor_module: nn.Module
    critic_module: nn.Module

    def __call__(self, obs: Array) -> tuple[Array, Array]:
        return self.actor(obs), self.critic(obs)

    def actor(self, x: Array) -> Array:
        return self.actor_module(x)

    def critic(self, x: Array) -> Array:
        return self.critic_module(x)


class CartPoleEnv(gym.Env):
    """CartPole environment wrapper to match the BraxState interface."""

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, rng: PRNGKeyArray) -> BraxState:
        obs, info = self.env.reset()
        return BraxState(
            pipeline_state=None,  # CartPole doesn't use pipeline state
            obs={"observations": jnp.array(obs)[None, :]},
            reward=jnp.array(1.0)[None],
            done=jnp.array(False)[None],
            info={"rng": rng, **info},
        )

    def step(self, state: BraxState, action: Array) -> BraxState:
        obs, reward, terminated, truncated, info = self.env.step(action.item())
        done = np.logical_or(terminated, truncated)

        return BraxState(
            pipeline_state=None,
            obs={"observations": jnp.array(obs)[None, :]},
            reward=jnp.array(reward)[None],
            done=jnp.array(done)[None],
            info={"rng": state.info["rng"], **info},
        )


@dataclass
class CartPoleConfig(PPOConfig):
    """Configuration for CartPole training."""

    # Env parameters.
    sutton_barto_reward: bool = xax.field(value=False, help="Use Sutton and Barto reward function.")
    batch_size: int = xax.field(value=16, help="Batch size.")

    # ML model parameters.
    actor_hidden_dims: int = xax.field(value=128, help="Hidden dimensions for the actor.")
    actor_num_layers: int = xax.field(value=2, help="Number of layers for the actor.")
    critic_hidden_dims: int = xax.field(value=128, help="Hidden dimensions for the critic.")
    critic_num_layers: int = xax.field(value=2, help="Number of layers for the critic.")


class CartPoleTask(RLTask[CartPoleConfig]):
    """Task for CartPole training."""

    def get_environment(self) -> CartPoleEnv:
        """Get the environment.
        Returns:
            The environment.
        """
        return CartPoleEnv()

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(1e-3),
        )

    def get_model(self, key: PRNGKeyArray) -> ActorCriticModel:
        """Get the model.
        Args:
            key: The random key.
        Returns:
            The model.
        """
        return ActorCriticModel(
            actor_module=MLP(
                num_hidden_layers=self.config.actor_num_layers,
                hidden_features=self.config.actor_hidden_dims,
                out_features=2,  # two discrete actions for CartPole
            ),
            critic_module=MLP(
                num_hidden_layers=self.config.critic_num_layers,
                hidden_features=self.config.critic_hidden_dims,
                out_features=1,
            ),
        )

    def get_init_params(self, key: PRNGKeyArray) -> PyTree:
        """Get the initial parameters.
        Args:
            key: The random key.
        Returns:
            The initial parameters.
        """
        # Provide a dummy observation (of appropriate shape) for initialization.
        dummy_obs = jnp.zeros((self.max_trajectory_steps, self.config.num_envs, 4))
        return self.get_model(key).init(key, obs=dummy_obs)

    def get_init_actor_carry(self) -> Array:
        raise NotImplementedError("Not a recurrent model.")

    @staticmethod
    @eqx.filter_jit
    def _apply_actor(model: ActorCriticModel, params: PyTree, x: Array) -> Array:
        """Apply the actor model to inputs."""
        return model.apply(params, method="actor", x=x)

    @staticmethod
    @eqx.filter_jit
    def _apply_critic(model: ActorCriticModel, params: PyTree, x: Array) -> Array:
        """Apply the critic model to inputs."""
        return model.apply(params, method="critic", x=x)

    @eqx.filter_jit
    def get_actor_output(
        self,
        model: ActorCriticModel,
        params: PyTree,
        sys: System,
        state: BraxState,
        rng: PRNGKeyArray,
        carry: Array | None,
    ) -> tuple[Array, None]:
        """Get the actor output.
        Args:
            model: The model definition.
            params: The parameters.
            sys: The system.
            state: The state.
            rng: The random key.
            carry: The carry state.
        Returns:
            The actor output and no carry state (keeping for consistency)
        """
        model_out = self._apply_actor(model, params, state.obs["observations"])
        assert isinstance(model_out, Array)
        return model_out, None

    def get_init_critic_carry(self) -> Array:
        """Get the critic carry state."""
        raise NotImplementedError("Not a recurrent model.")

    @eqx.filter_jit
    def get_critic_output(
        self,
        model: ActorCriticModel,
        params: PyTree,
        sys: System,
        state: BraxState,
        rng: PRNGKeyArray,
        carry: Array | None,
    ) -> tuple[Array, None]:
        """Get the critic output.
        Args:
            model: The model.
            params: The parameters.
            sys: The system.
            state: The state.
            rng: The random key.
            carry: The carry state.
        Returns:
            The critic output and no carry state (keeping for consistency)
        """
        model_out = self._apply_critic(model, params, state.obs["observations"])
        assert isinstance(model_out, Array)
        return model_out, None

    def unroll_trajectories(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env: CartPoleEnv,
        rng: PRNGKeyArray,
    ) -> PPOBatch:
        """Rollout the model for a given number of steps.
        Args:
            model: The model.
            params: The parameters (really a variable dictionary).
            env: The environment.
            rng: The random key.
        Returns:
            A PPOBatch containing trajectories with shape (num_steps, ...).
        """
        observations = []
        actions = []
        rewards = []
        done = []
        action_log_probs = []

        state = env.reset(rng)
        rng, _ = jax.random.split(rng)
        for _ in range(self.max_trajectory_steps):

            logits = self._apply_actor(model, params, state.obs["observations"])
            assert isinstance(logits, Array)
            log_probs = jax.nn.log_softmax(logits)
            sampled_actions = jax.random.categorical(rng, logits)
            log_probs = log_probs[jnp.arange(logits.shape[0]), sampled_actions]

            observations.append(state.obs)
            done.append(state.done)
            actions.append(sampled_actions)
            action_log_probs.append(log_probs)
            rewards.append(state.reward)
            # Defining reward at current state... my thought process is that otherwise the
            # policy loss term will be 0 during the last step of the trajectory (logp * r)

            if state.done:
                state = env.reset(rng)
                rng, _ = jax.random.split(rng)
            else:
                state = env.step(state, sampled_actions)

        observations = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *observations)
        next_observations = jax.tree_util.tree_map(
            lambda x: jnp.roll(x, shift=-1, axis=0), observations
        )
        actions = jnp.stack(actions)
        rewards = jnp.stack(rewards)
        done = jnp.stack(done)
        action_log_probs = jnp.stack(action_log_probs)

        return PPOBatch(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            done=done,
            action_log_probs=action_log_probs,
        )

    def compute_advantages(
        self,
        values: Array,
        batch: PPOBatch,
    ) -> Array:
        """Computes the advantages using Generalized Advantage Estimation (GAE).
        Args:
            values: The value estimates for observations, (num_envs, num_steps, ...).
            batch: The batch containing rewards and termination signals, (num_envs, num_steps, ...).
        Returns:
            Advantages with the same shape as values.
        """
        next_values = jnp.roll(values, shift=-1, axis=0)
        mask = jnp.where(batch.done, 0.0, 1.0)

        # getting td residuals
        deltas = batch.rewards + self.config.gamma * next_values * mask - values
        advantages = discount_cumsum_gae(deltas, mask, self.config.gamma, self.config.lam)

        return advantages

    @eqx.filter_jit
    def compute_loss(
        self,
        model: ActorCriticModel,
        params: PyTree,
        batch: PPOBatch,
    ) -> tuple[Array, Dict[str, Array]]:
        """Compute the PPO loss.
        Args:
            model: The model.
            params: The parameters.
            mini_batch: The mini-batch containing trajectories.
        Returns:
            A tuple of (loss, metrics).
        """
        # get the log probs of the current model
        actions = self._apply_actor(model, params, batch.observations["observations"])
        assert isinstance(actions, Array)
        log_probs = jax.nn.log_softmax(actions)
        log_prob = log_probs[
            jnp.arange(log_probs.shape[0])[:, None], jnp.arange(log_probs.shape[1]), batch.actions
        ]
        ratio = jnp.exp(log_prob - batch.action_log_probs)

        # get the state-value estimates
        values = self._apply_critic(model, params, batch.observations["observations"])
        assert isinstance(values, Array)
        values = values.squeeze(axis=-1)  # values is (time, env)
        advantages = self.compute_advantages(values, batch)  # (time, env)

        returns = advantages + values
        # normalizing advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # policy loss with clipping
        policy_loss = -jnp.mean(
            jnp.minimum(
                ratio * advantages,
                jnp.clip(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param)
                * advantages,
            )
        )

        # value loss term
        # TODO: add clipping
        value_pred = self._apply_critic(model, params, batch.observations["observations"])
        value_pred = value_pred.squeeze(axis=-1)  # (time, env)
        value_loss = 0.5 * jnp.mean((returns - value_pred) ** 2)

        # entropy bonus term
        probs = jax.nn.softmax(actions)
        entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
        entropy_loss = -self.config.entropy_coef * entropy

        total_loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss

        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
        }
        return total_loss, metrics

    def model_update(
        self,
        model: ActorCriticModel,
        params: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        batch: PPOBatch,
    ) -> tuple[PyTree, optax.OptState]:
        """Performs a single optimization update."""
        # Move jit outside the function to prevent recompilation
        loss_val, grads = self._jitted_value_and_grad(model, params, batch)
        print(f"Loss: {loss_val}")
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    @eqx.filter_jit
    def _jitted_value_and_grad(
        self,
        model: ActorCriticModel,
        params: PyTree,
        batch: PPOBatch,
    ) -> tuple[Array, PyTree]:
        """Jitted version of value_and_grad computation."""
        loss_fn = lambda p: self.compute_loss(model, p, batch)[0]
        return jax.value_and_grad(loss_fn)(params)

    def train_loop(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env: CartPoleEnv,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        state: xax.State,
    ) -> None:
        rng = self.prng_key()
        rng, train_rng, val_rng = jax.random.split(rng, 3)

        while not self.is_training_over(state):
            if self.valid_step_timer.is_valid_step(state):
                val_rng, step_val_rng = jax.random.split(val_rng)
                trajectory = self.unroll_trajectories(model, params, env, step_val_rng)
                # Calculate episode length by counting steps until done
                episode_lengths = jnp.sum(~trajectory.done) / jnp.sum(trajectory.done)
                print(f"Average episode length: {episode_lengths}")

            #     # Perform logging.
            #     with self.step_context("write_logs"):
            #         state.raw_phase = "valid"
            #         # self.log_state_timers(state)
            #         # self.log_trajectory(env, trajectory)
            #         # self.log_trajectory_stats(env, trajectory)
            #         self.logger.write(state)
            #         state.num_valid_samples += 1

            with self.step_context("on_step_start"):
                state = self.on_step_start(state)

            # Unrolls a trajectory.
            train_rng, step_rng = jax.random.split(train_rng)
            trajectories = self.unroll_trajectories(model, params, env, step_rng)

            # Updates the model on the collected trajectories.
            with self.step_context("update_state"):
                params, opt_state = self.model_update(
                    model, params, optimizer, opt_state, trajectories
                )

            # # Logs the trajectory statistics.
            with self.step_context("write_logs"):
                #     state.phase = "train"
                #     self.log_state_timers(state)
                #     self.log_trajectory_stats(env, trajectories)
                #     self.logger.write(state)
                state.num_steps += 1

            with self.step_context("on_step_end"):
                state = self.on_step_end(state)

            if self.should_checkpoint(state):
                self.save_checkpoint(
                    model=params, optimizer=optimizer, opt_state=opt_state, state=state
                )  # Update XAX to be flax compatible

    def run_training(self) -> None:
        """Runs the main PPO training loop."""
        with self:
            key = self.prng_key()

            self.set_loggers()

            env = self.get_environment()

            if xax.is_master():
                Thread(target=self.log_state, daemon=True, args=(env,)).start()

            key, model_key = jax.random.split(key)
            model, optimizer, opt_state, state = self.load_initial_state(model_key)

            state = self.on_training_start(state)

            def on_exit() -> None:
                self.save_checkpoint(model, optimizer, opt_state, state)

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

            params = self.get_init_params(key)
            opt_state = optimizer.init(params)

            try:
                self.train_loop(
                    model=model,
                    params=params,
                    env=env,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    state=state,
                )

            except xax.TrainingFinishedError:
                if xax.is_master():
                    xax.show_info(
                        f"Finished training after {state.num_steps} steps, {state.num_samples} samples",
                        important=True,
                    )
                self.save_checkpoint(model, optimizer, opt_state, state)

            except (KeyboardInterrupt, bdb.BdbQuit):
                if xax.is_master():
                    xax.show_info("Interrupted training", important=True)

            except BaseException:
                exception_tb = textwrap.indent(
                    xax.highlight_exception_message(traceback.format_exc()), "  "
                )
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()
                self.save_checkpoint(model, optimizer, opt_state, state)

            finally:
                state = self.on_training_end(state)


if __name__ == "__main__":
    # python -m examples.cartpole.cartpole_task train
    CartPoleTask.launch(
        CartPoleConfig(
            num_envs=1,
            batch_size=16,
            max_trajectory_seconds=10.0,
            valid_every_n_steps=5,
        ),
    )
