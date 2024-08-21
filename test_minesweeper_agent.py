from environment import Environment  # Import Environment class
from cspagent import ProbCSPAgent    # Import your agent class

def main():
    # Initialize the environment with the desired parameters
    env = Environment(n=8, mine_density=0.1, visual=True, end_game_on_mine_hit=True)

    # Generate the environment grid and mines
    env.generate_environment()

    # Initialize the agent with the environment
    agent = ProbCSPAgent(env=env, end_game_on_mine_hit=True, prob=0.5, use_probability_agent=True)

    # Run the agent to play the game
    agent.play()

    # Get and print game metrics
    metrics = agent.get_gameplay_metrics()
    print("Game Won:", metrics["game_won"])
    print("Number of Mines Hit:", metrics["number_of_mines_hit"])
    print("Correct Flags:", metrics["number_of_mines_flagged_correctly"])
    print("Incorrect Flags:", metrics["number_of_cells_flagged_incorrectly"])

if __name__ == "__main__":
    main()
