"""
Policy gradient to solve lunar lander

# Links
https://towardsdatascience.com/breaking-down-richard-suttons-policy-gradient-9768602cb63b

# Running a remote jupyter notebook
ssh -i ~/.ssh/aws/redeemsmart-dev.pem -NfL 8891:localhost:8891 ubuntu@10.5.32.177

"""
from collections import deque
import gym
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


class Params:
    # num_epochs = 5000
    alpha = 5e-3  # learning rate
    batch_size = 64  # how many episodes we want to pack into an epoch
    gamma = 0.99  # discount rate
    hidden_size = 64  # number of hidden nodes we have in our dnn
    beta = 0.1  # the entropy bonus multiplier


# Q-table is replaced by a neural network
class Agent(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Agent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_space_size, bias=True),
        )

    def forward(self, x):
        """
        Args:
            x: its shape is [1, 8]
        """
        x = normalize(x, dim=1)
        x = self.net(x)
        return x


class PolicyGradient(object):
    def __init__(self, problem: str = "CartPole", use_cuda: bool = False, params: dict = Params):
        self.alpha = params.alpha
        self.batch_size = params.batch_size
        self.gamma = params.gamma
        self.hidden_size = params.hidden_size
        self.beta = params.beta
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

        # instantiate the tensorboard writer
        self.writer = SummaryWriter(
            comment=f"_PG_CP_Gamma={self.gamma},"
            f"LR={self.alpha},"
            f"BS={self.batch_size},"
            f"NH={self.hidden_size},"
            f"beta={self.beta}"
        )
        self.env = gym.make("CartPole-v1") if problem == "CartPole" else gym.make("LunarLander-v2")
        self.agent = Agent(
            observation_space_size=self.env.observation_space.shape[0],
            action_space_size=self.env.action_space.n,
            hidden_size=self.hidden_size,
        ).to(self.DEVICE)

        self.adam = optim.Adam(params=self.agent.parameters(), lr=self.alpha)

        self.total_rewards = deque([], maxlen=100)

        # flag to figure out if we have rendered a single episode current epoch
        self.finished_rendering_this_epoch = False

    def solve_environment(self):
        """The main interface for the Policy Gradient solver
        """
        episode, epoch = 0, 0  # init the episode and the epoch

        # Init the epoch arrays, used for entropy calculation
        epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

        while True:
            # play an episode of the environment
            (episode_weighted_log_prob_trajectory, episode_logits, sum_of_episode_rewards, episode) = self.play_episode(
                episode=episode
            )
            # after each episode append the sum of total rewards to the deque
            self.total_rewards.append(sum_of_episode_rewards)

            # append the weighted log-probabilities of actions
            epoch_weighted_log_probs = torch.cat(
                (epoch_weighted_log_probs, episode_weighted_log_prob_trajectory), dim=0
            )

            # append the logits - needed for the entropy bonus calculation
            epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)

            # If the epoch is over - we have enough trajectories to perform the policy gradient
            if episode >= self.batch_size:
                # Calculate the loss
                loss, entropy = self.calculate_loss(
                    epoch_logits=epoch_logits, weighted_log_probs=epoch_weighted_log_probs
                )

                self.adam.zero_grad()
                loss.backward()
                self.adam.step()  # update the parameters
                print(
                    "\r", f"Epoch: {epoch}, Avg return per epoch: {np.mean(self.total_rewards):.3f}", end="", flush=True
                )
                self.writer.add_scalar(
                    tag="Average Return over 100 episodes", scalar_value=np.mean(self.total_rewards), global_step=epoch
                )
                self.writer.add_scalar(tag="Entropy", scalar_value=entropy, global_step=epoch)

                ###########################
                # Reset a few parameters
                ###########################
                self.finished_rendering_this_epoch = False  # Reset the rendering flag
                episode = 0
                epoch += 1
                # Reset the epoch arrays, used for entropy calculation
                epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
                epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

                # check if solved
                if np.mean(self.total_rewards) > 200:
                    print("\nSolved!")
                    break

        self.env.close()  # Close the gym environment
        self.writer.close()  # Close the writer

    def play_episode(self, episode: int):
        """Plays one episode

        Returns:
            sum_weighted_log_probs: the sum of the log-prob of an action multiplied by the reward-to-go from that state
            episode_logits: the logits of every step of the episode - needed to compute entropy for entropy bonus
            finished_rendering_this_epoch: pass-through rendering flag
            sum_of_rewards: sum of the rewards for the episode - needed for the average over 200 episode statistic
        """
        state = self.env.reset()  # A random init state

        # initialize the episode arrays
        episode_actions = torch.empty(size=(0,), dtype=torch.long, device=self.DEVICE)
        episode_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        average_rewards, episode_rewards = np.empty(shape=(0,), dtype=np.float), np.empty(shape=(0,), dtype=np.float)

        # Episode loop
        while True:
            # render the environment for the first episode in the epoch
            if not self.finished_rendering_this_epoch:
                self.env.render()

            # Get the action logits from the agent - (preferences)
            action_logits = self.agent(torch.tensor(state).float().unsqueeze(dim=0).to(self.DEVICE))
            episode_logits = torch.cat((episode_logits, action_logits), dim=0)  # append to episode_logits

            # Sample an action based on our policy probability distribution
            action = Categorical(logits=action_logits).sample()

            # append the action to the episode action list to obtain the trajectory
            # we need to store the actions and logits so we could calculate the gradient of the performance
            episode_actions = torch.cat((episode_actions, action), dim=0)

            state, reward, done, _ = self.env.step(action=action.cpu().item())  # Take the action

            # append the reward to the rewards pool
            # so we can calculate the weights for the policy gradient and the baseline of average
            episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)

            # here the average reward is state specific
            average_rewards = np.concatenate(
                (average_rewards, np.expand_dims(np.mean(episode_rewards), axis=0)), axis=0
            )

            # the episode is done
            if done:
                episode += 1

                # turn the rewards we accumulated during the episode into the rewards-to-go:
                # earlier actions are responsible for more rewards than the later taken actions
                discounted_rewards_to_go = PolicyGradient.get_discounted_rewards(
                    rewards=episode_rewards, gamma=self.gamma
                )
                discounted_rewards_to_go -= average_rewards  # baseline - state specific average

                # calculate the sum of the rewards for the running average metric
                sum_of_rewards = np.sum(episode_rewards)

                # set the mask for the actions taken in the episode
                mask = one_hot(episode_actions, num_classes=self.env.action_space.n)

                # calculate the log-probabilities of the taken actions
                # mask is needed to filter out log-probabilities of not related logits
                episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)

                # weight the episode log-probabilities by the rewards-to-go
                episode_weighted_log_probs = episode_log_probs * torch.tensor(discounted_rewards_to_go).float().to(
                    self.DEVICE
                )

                # calculate the sum over trajectory of the weighted log-probabilities
                sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

                # won't render again this epoch
                self.finished_rendering_this_epoch = True
                return sum_weighted_log_probs, episode_logits, sum_of_rewards, episode

    def calculate_loss(
        self, epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """Calculates the policy "loss" and the entropy bonus

        Args:
            epoch_logits: logits of the policy network we have collected over the epoch
            weighted_log_probs: loP * W of the actions taken
        Returns:
            policy loss + the entropy bonus
            entropy: needed for logging
        """
        policy_loss = -1 * torch.mean(weighted_log_probs)

        # Add the entropy bonus
        p = softmax(epoch_logits, dim=1)
        log_p = log_softmax(epoch_logits, dim=1)
        entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
        entropy_bonus = -1 * self.beta * entropy
        return policy_loss + entropy_bonus, entropy

    @staticmethod
    def get_discounted_rewards(rewards: np.array, gamma: float) -> np.array:
        """Calculates the sequence of discounted rewards-to-go.

        Args:
            rewards: the sequence of observed rewards
            gamma: the discount factor
        Returns:
            discounted_rewards: the sequence of the rewards-to-go
        """
        discounted_rewards = np.empty_like(rewards, dtype=np.float)
        for i in range(rewards.shape[0]):
            gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
            discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
            discounted_reward = np.sum(rewards[i:] * discounted_gammas)
            discounted_rewards[i] = discounted_reward
        return discounted_rewards


def main(env, use_cuda: bool = False):

    assert env in ["CartPole", "LunarLander"]
    policy_gradient = PolicyGradient(problem=env, use_cuda=use_cuda)
    policy_gradient.solve_environment()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="CartPole or LunarLander environment", default="LunarLander")
    parser.add_argument("--use_cuda", help="Use if you want to use CUDA", action="store_true")

    args = vars(parser.parse_args())
    print("Command line args:\n%s" % json.dumps(args, indent=4))
    main(**args)
