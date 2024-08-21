import numpy as np
from concurrent.futures import ThreadPoolExecutor
from environment import Environment
from variable import Variable

class ProbCSPAgent:

    def __init__(self, env=None, end_game_on_mine_hit=True, prob=0, use_probability_agent=False, threshold=0.9, max_backtrack=10):
        self.env = env
        self.end_game_on_mine_hit = end_game_on_mine_hit
        self.all_constraint_equations = list()
        self.non_mine_variables = list()
        self.mine_variables = list()
        self.prob = prob
        self.game_stuck = False
        self.use_probability_agent = use_probability_agent
        self.game_won = False
        self.move_history = []
        self.probability_threshold = threshold
        self.max_backtrack = max_backtrack  # Maximum depth for backtracking
        self.backtrack_count = 0  # Track the number of backtracks

    def _create_constraint_equation_for_variable(self, variable):
        row = variable.row
        column = variable.column

        p = np.random.binomial(1, self.prob)
        if p == 0:
            self.env.mine_ground_copy[row, column] = None
            self.env.clicked_and_not_revealed[row, column] = True
            return

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue

                if 0 <= row + i < self.env.n and 0 <= column + j < self.env.n:

                    if self.env.opened[row + i, column + j]:
                        continue

                    if self.env.flags[row + i, column + j]:
                        variable.constraint_value -= 1
                        continue

                    neighbour = self.env.variable_mine_ground_copy[row + i, column + j]
                    variable.add_constraint_variable(variable=neighbour)

        self.all_constraint_equations.append([variable.constraint_equation, variable.constraint_value])

    def _calculate_probability(self, cell):
        """Calculate the probability that a given cell is a mine."""
        surrounding_cells = cell.get_unopened_neighbours(self.env)
        if not surrounding_cells:
            return 0

        total_mine_prob = 0
        total_constraints = 0

        for equation in self.all_constraint_equations:
            if cell in equation[0]:
                probability = equation[1] / len(equation[0])
                total_mine_prob += probability
                total_constraints += 1

        if total_constraints == 0:
            return 0

        return total_mine_prob / total_constraints

    def _visualise_equations(self):
        for equation in self.all_constraint_equations:
            print(repr([(variable.row, variable.column) for variable in equation[0]]), " = ", equation[1])

    def _remove_duplicates(self, array):
        uniqueList = []
        for element in array:
            if element not in uniqueList:
                uniqueList.append(element)
        return uniqueList

    def _resolve_subsets(self):
        self.all_constraint_equations = sorted(self.all_constraint_equations, key=lambda x: len(x[0]))
        for equation in self.all_constraint_equations:
            for equation_ in self.all_constraint_equations:
                if equation == equation_ or not equation[0] or not equation_[0] or not equation[1] or not equation_[1]:
                    continue

                if set(equation[0]).issubset(set(equation_[0])):
                    equation_[0] = list(set(equation_[0]) - set(equation[0]))
                    equation_[1] -= equation[1]
                    continue

                if set(equation_[0]).issubset(set(equation[0])):
                    equation[0] = list(set(equation[0]) - set(equation_[0]))
                    equation[1] -= equation_[1]

        self._check_equations_for_mine_and_non_mine_variables()

    def _backtrack(self):
        if self.backtrack_count >= self.max_backtrack:
            return False  # Stop backtracking after reaching the maximum allowed depth

        if not self.move_history:
            return False

        last_move = self.move_history.pop()
        self.env.undo_move(last_move['row'], last_move['column'])

        if last_move['type'] == 'click':
            self.non_mine_variables.append(last_move['variable'])
        elif last_move['type'] == 'flag':
            self.mine_variables.append(last_move['variable'])

        self.backtrack_count += 1  # Increment the backtrack counter
        return True

    def _check_equations_for_mine_and_non_mine_variables(self):
        for equation in self.all_constraint_equations.copy():
            if len(equation) == 0 or len(equation[0]) == 0:
                self.all_constraint_equations.remove(equation)
                continue

            if equation[1] == 0:
                self.all_constraint_equations.remove(equation)
                for non_mine_variable in equation[0]:
                    if not self.env.opened[non_mine_variable.row, non_mine_variable.column] and \
                            non_mine_variable not in self.non_mine_variables:
                        self.non_mine_variables.append(non_mine_variable)
                continue

            if len(equation[0]) == equation[1]:
                self.all_constraint_equations.remove(equation)
                for mine_variable in equation[0]:
                    if not self.env.flags[mine_variable.row, mine_variable.column] and \
                            mine_variable not in self.mine_variables:
                        self.mine_variables.append(mine_variable)

    def _remove_variable_from_other_equations(self, variable, is_mine_variable=False):
        for equation in self.all_constraint_equations:
            if variable in equation[0]:
                equation[0].remove(variable)
                if is_mine_variable and equation[1]:
                    equation[1] -= 1

    def _add_mine_flag(self, cell):
        self.env.add_mine_flag(cell.row, cell.column)
        self._remove_variable_from_other_equations(variable=cell, is_mine_variable=True)
        self.move_history.append({'type': 'flag', 'row': cell.row, 'column': cell.column, 'variable': cell})

    def _open_mine_cell(self, cell):
        self.env.open_mine_cell(cell.row, cell.column)
        self._remove_variable_from_other_equations(variable=cell, is_mine_variable=True)
        self.move_history.append({'type': 'click', 'row': cell.row, 'column': cell.column, 'variable': cell})

    def _click_square(self, cell):
        self.env.click_square(cell.row, cell.column)
        if self.env.mine_hit and not self.end_game_on_mine_hit:
            self._open_mine_cell(cell=cell)
            return

        self._create_constraint_equation_for_variable(variable=cell)
        self._remove_variable_from_other_equations(variable=cell)
        self.move_history.append({'type': 'click', 'row': cell.row, 'column': cell.column, 'variable': cell})

    def _check_solvable_csp(self):
        return not self.non_mine_variables and not self.mine_variables

    def _click_random_square_with_heuristic(self):
        unopened_cells = dict()
        open_cell_coords = list(zip(*np.where(self.env.opened & ~self.env.clicked_and_not_revealed)))

        for row, column in open_cell_coords:
            open_cell = self.env.variable_mine_ground_copy[row, column]
            number_of_cell_mines_found = open_cell.get_flagged_mines(env=self.env)
            risk = open_cell.value - number_of_cell_mines_found
            unopened_cell_neighbours = open_cell.get_unopened_neighbours(env=self.env, use_probability_agent=self.use_probability_agent)

            for cell_neighbour in unopened_cell_neighbours:
                if cell_neighbour not in unopened_cells:
                    unopened_cells[cell_neighbour] = 0
                unopened_cells[cell_neighbour] += risk

        if not unopened_cells:
            self.game_stuck = True
            return

        random_cell = min(unopened_cells, key=unopened_cells.get)
        self._click_square(random_cell)

    def _click_random_square(self):
        unopened_cells_coords = list(zip(*np.where(~self.env.clicked)))
        if not unopened_cells_coords:
            return

        random_cells = [self.env.variable_mine_ground_copy[row, col] for (row, col) in unopened_cells_coords]
        random_cell = np.random.choice(random_cells)
        self._click_square(random_cell)

    def _click_all_non_mine_cells(self):
        while self.non_mine_variables:
            non_mine_variable = self.non_mine_variables.pop(0)
            self._click_square(non_mine_variable)

    def _flag_all_mine_cells(self):
        for cell in self.mine_variables[:]:  # Iterate over a copy of the list
            probability = self._calculate_probability(cell)
            if probability >= self.probability_threshold:
                self._add_mine_flag(cell)
                self.mine_variables.remove(cell)  # Remove the cell after flagging

    def _basic_solver(self):
        self._click_all_non_mine_cells()
        self.all_constraint_equations = self._remove_duplicates(self.all_constraint_equations)
        self._flag_all_mine_cells()
        self._check_equations_for_mine_and_non_mine_variables()

    def _adjust_probability(self):
        remaining_cells = np.count_nonzero(~self.env.clicked)
        if remaining_cells < 10:  # Arbitrary threshold for when to increase risk-taking
            self.prob = min(self.prob + 0.1, 1.0)  # Increase probability up to 1.0

    def get_gameplay_metrics(self):
        metrics = dict()
        metrics["number_of_mines_hit"] = self.env.number_of_mines_hit
        metrics["number_of_mines_flagged_correctly"] = len(list(zip(*np.where(self.env.mines & self.env.flags.astype(bool)))))
        metrics["number_of_cells_flagged_incorrectly"] = len(list(zip(*np.where(~self.env.mines & self.env.flags.astype(bool)))))
        metrics["game_stuck"] = self.game_stuck
        metrics["game_won"] = self.game_won
        return metrics

    def play(self):
        self.non_mine_variables.append(self.env.variable_mine_ground_copy[0, 0])
        while True:
            self._basic_solver()

            all_flags_equal_to_mines = list(zip(*np.where(self.env.mines))) == list(zip(*np.where(self.env.flags)))
            all_clicked = np.all(self.env.clicked)

            if self.game_stuck:
                if not self._backtrack():  # Try backtracking if stuck
                    self.game_stuck = False
                    self._click_random_square()

            if self.env.mine_hit:
                self.game_won = False
                return

            if all_clicked:
                self.game_won = all_flags_equal_to_mines
                return

            if self._check_solvable_csp():
                self._resolve_subsets()

                if self._check_solvable_csp():
                    self._adjust_probability()
                    self._click_random_square_with_heuristic()
