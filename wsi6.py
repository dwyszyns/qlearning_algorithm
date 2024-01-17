import numpy as np
import gym
from gym import Env
from dataclasses import dataclass
from matplotlib import pyplot as plt

random = np.random.default_rng(0)


@dataclass
class results_t:
    train_iterations: list
    train_rewards: list


@dataclass
class hyperparams_t:
    learning_rate: float
    discount_rate: float
    exploration: float


@dataclass
class params_t:
    episodes_number: int
    interval: int
    
    
def make_plot(results_list:list, strategy_names:list, params:params_t):
    for i, results in enumerate(results_list):
        strategy_name = strategy_names[i]
        yaxis = list(range(params.episodes_number))
        plt.plot(yaxis, results.train_rewards, label=strategy_name, alpha=0.7)

    plt.xlabel("number of iterations")
    plt.ylabel("reward value")
    plt.title("Reward Value Over Iterations") 
    plt.legend()
    plt.show()
    

@dataclass
class ExplorationStrategy:
    exploration: float


class EpsilonStrategy(ExplorationStrategy):
    def __init__(self, exploration:float=0.2):
        super().__init__(exploration)

    def choose_action(self, current_state:int, q_table:np.ndarray, actions_number:int) -> np.ndarray:
        if self.exploration < random.uniform():
            return np.argmax(q_table[current_state])
        return random.choice(np.array(range(actions_number)))


class BoltzmannStrategy(ExplorationStrategy):
    def __init__(self, exploration:float=0.2, temperature:float=0.1):
        super().__init__(exploration)
        self.temperature = temperature

    def choose_action(self, current_state:int, q_table:np.ndarray, actions_number:int) -> np.ndarray:
        q_values = q_table[current_state]
        probabilities = self.boltzmann_distribution(q_values)
        chosen_action = np.random.choice(actions_number, p=probabilities)
        return chosen_action

    def boltzmann_distribution(self, q_values:np.ndarray) -> np.ndarray:
        exp_q_values = np.exp(q_values / self.temperature)
        probabilities = exp_q_values / np.sum(exp_q_values)
        return probabilities


class QLearn:
    def __init__(
        self, observations_number:int, actions_number:int, hyperparams:hyperparams_t, exploration_strategy:ExplorationStrategy
    ):
        self.learning_rate = hyperparams.learning_rate
        self.discount_rate = hyperparams.discount_rate
        self.actions_number = actions_number
        self.exp_strategy = exploration_strategy
        self.exp_strategy.exploration = hyperparams.exploration
        self.q_table = np.zeros((observations_number, actions_number))

    def update_qtable(self, prev_state:int, curr_state:int, prev_action:int, reward:float) -> None:
        prev_q = self.q_table[prev_state, prev_action]
        temporal_difference = reward + self.discount_rate * np.max(self.q_table[curr_state]) - prev_q
        self.q_table[prev_state, prev_action] = prev_q + self.learning_rate * temporal_difference

    def run_episode(self, env:Env, max_iterations:int) -> tuple[int, float]:
        iteration = 0
        episode_reward = 0
        terminated = False
        truncated = False
        current_state, _ = env.reset()

        while not (iteration > max_iterations or terminated or truncated):
            action = self.exp_strategy.choose_action(
                current_state, self.q_table, self.actions_number
            )
            previous_state = current_state
            current_state, reward, terminated, truncated, _ = env.step(action)
            self.update_qtable(previous_state, current_state, action, reward)
            episode_reward += reward
            iteration += 1

        return iteration, episode_reward


def run_qlearning(agent: QLearn, env:Env, params:params_t) -> results_t:
    train_iteration_per_episode = []
    train_reward_per_episode = []

    for _ in range(params.episodes_number):
        iteration, episode_reward = agent.run_episode(env, params.interval)
        train_iteration_per_episode.append(iteration)
        train_reward_per_episode.append(episode_reward)

    return results_t(train_iteration_per_episode, train_reward_per_episode)


def main():
    env = gym.make("Taxi-v3", render_mode="rgb_array").env
    hyperparams = hyperparams_t(learning_rate=0.5, discount_rate=0.8, exploration=0.2)
    params = params_t(episodes_number=10000, interval=1000)
    
    strategy1 = EpsilonStrategy()
    strategy2 = BoltzmannStrategy(temperature=0.1)

    epsilon_agent = QLearn(
        env.observation_space.n, env.action_space.n, hyperparams, strategy1
    )
    boltzmann_agent = QLearn(
        env.observation_space.n, env.action_space.n, hyperparams, strategy2
    )

    results1 = run_qlearning(epsilon_agent, env, params)
    results2 = run_qlearning(boltzmann_agent, env, params)

    env.close()
    make_plot([results2, results1], ["Boltzmann, temperature=0.1", "Epsilon"], params)


if __name__ == "__main__":
    main()
