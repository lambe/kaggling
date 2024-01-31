import ast
from collections import deque
import os
import re
import string
from typing import Iterable

import numpy as np
from numpy import ndarray
import pandas as pd
import tqdm

# A utility class to help with some puzzle operations
class PermutationPuzzle(object):
    def __init__(self, puzzle_series: pd.Series):
        self.raw_series = puzzle_series
        self.puzzle_type = puzzle_series["puzzle_type"]
        self.initial_state = np.array(puzzle_series["initial_state"].split(";"))
        self.solution_state = np.array(puzzle_series["solution_state"].split(";"))
        self.state = self.initial_state.copy()
        self.path = []
        self.num_wildcards = puzzle_series["num_wildcards"]
        self.base_solution = np.array(puzzle_series["moves"].split("."))

        self.colour_dict, self.colour_dict_reverse = make_forward_and_reverse_colour_maps(self.solution_state)

        # More fine-grained information about the puzzle type
        # puzzle_shape is either "cube", "globe", or "wreath"
        # dimensions is the puzzle size as a list of integers
        p_type_items = self.puzzle_type.split("_")
        self.puzzle_shape = p_type_items[0]
        numbers = p_type_items[1].split("/")
        self.dimensions = [int(k) for k in numbers]

        # Create the actions dictionaries
        # The expanded actions dictionary includes negative actions and repetitions of the same action
        # such that the best set of moves in every direction is included
        self.base_actions_dict = ast.literal_eval(puzzle_info_df.loc[puzzle_info_df['puzzle_type'] == self.puzzle_type, 'allowed_moves'].values[-1])
        self.actions_dict = {}

        # We need this object to use integers to index the actions, rather than the identifying string
        # Some off-the-shelf codes require actions to be integers, not strings
        # See also the self.actions_index_reverse object below...
        self.actions_index = {}

        # Puzzle-topology-specific code is added in child classes
        return

    def get_state_as_str(self) -> str:
        # Print the current state in the same format as the initial and solution states
        return ";".join(self.state)
    
    def get_state(self) -> list[str]:
        return self.state

    def reset_state(self) -> None:
        self.state = self.initial_state.copy()
        self.path = []
        return

    def possible_actions(self) -> list[int]:
        # Puzzle-topology-specific code is added in child classes
        pass

    def possible_actions_by_name(self) -> list[str]:
        if len(self.path) == 0:
            return list(self.actions_dict.keys())
        
        actions_list = self.possible_actions()
        return [self.actions_index[k] for k in actions_list]
    
    def take_action(self, action: int) -> None:
        # Add the action to the path
        self.path.append(self.actions_index[action])
        # Execute the permutation action
        perm = self.actions_dict[self.actions_index[action]]
        self.state = self.state[perm]
        return 
    
    def take_action_by_name(self, action: str) -> None:
        # Add the action to the path
        self.path.append(action)
        # Execute the permutation action
        perm = self.actions_dict[action]
        self.state = self.state[perm]
        return

    def is_terminated(self) -> bool:
        wc_count = 0
        for i in range(len(self.state)):
            if self.state[i] != self.solution_state[i]:
                wc_count += 1

        if wc_count > self.num_wildcards:
            return False
        else:
            return True


class WreathPuzzle(PermutationPuzzle):
    def __init__(self, puzzle_series: pd.Series):
        assert "wreath" in puzzle_series["puzzle_type"]
        super().__init__(puzzle_series)

        # Figure out the rotations for wreath puzzles
        # For example, the 6x6 wreath has 5 possible left ring rotations: -2l, -l, l, 2l, and 3l (same for the right ring)
        # The 7x7 wreath has 6 possible left ring rotations: -3l, -2l, -l, l, 2l, and 3l (same for the right ring)
        # Use the min_rot and max_rot variables to determine the range of rotations and the
        # perm_nums list to enumerate each rotation
        n_moves_per_rot = self.dimensions[0] - 1
        max_rot = self.dimensions[0] // 2
        min_rot = -1 * max_rot
        is_odd_dim = self.dimensions[0] % 2 == 1
        if not is_odd_dim:
            min_rot += 1
        perm_nums = list(range(min_rot, max_rot + 1))
        perm_nums.remove(0)

        for i, (k, v) in enumerate(self.base_actions_dict.items()):
            perm = np.array(v)
            inv_perm = np.argsort(perm)

            for q, j in enumerate(perm_nums):
                if j == 1:
                    new_k = k
                    new_v = perm.copy()
                elif j == -1:
                    new_k = "-" + k
                    new_v = inv_perm.copy()
                else:
                    new_k = str(j) + k
                    if j > 0:
                        new_v = perm.copy()
                        for _ in range(2, j+1):
                            new_v = new_v[perm]
                    else:
                        new_v = inv_perm.copy()
                        for _ in range(2, -j + 1):
                            new_v = new_v[inv_perm]

                self.actions_dict[new_k] = new_v
                self.actions_index[n_moves_per_rot*i + q] = new_k

        # Utilities: a dictionary of indices and the length of the action set
        self.actions_index_reverse = {v: k for k, v in self.actions_index.items()}
        self.actions_dict_length = len(self.actions_dict)

    def possible_actions(self) -> list[int]:
        actions_list = list(np.arange(self.actions_dict_length))
        if len(self.path) == 0:
            return actions_list

        # Prevent the reverse of the action we just took from being selected right away
        last_move = self.path[-1]

        # For wreath puzzles, it doesn't make sense to follow a left ring rotation with another left ring rotation
        # (or follow a right ring rotation with another right ring rotation) because the second move will always be
        # suboptimal. Therefore, remove any rotations of the same ring from consideration
        ring_rotated = "l" if "l" in last_move else "r"
        remove_list = [k for k in self.actions_dict.keys() if ring_rotated in k]

        for move in remove_list:
            actions_list.remove(self.actions_index_reverse[move])

        return actions_list


class CubePuzzle(PermutationPuzzle):
    def __init__(self, puzzle_series: pd.Series):
        assert "cube" in puzzle_series["puzzle_type"]
        super().__init__(puzzle_series)

        for i, (k, v) in enumerate(self.base_actions_dict.items()):
            # Add a second rotation in the same direction to the dictionary
            # The third rotation in the same direction would be the same as the negative of the first rotation, so do not add that one
            perm = np.array(v)
            k2 = '2' + k
            perm2 = perm[perm]

            self.actions_dict[k] = perm
            self.actions_dict[k2] = perm2
            self.actions_dict["-" + k] = np.argsort(perm)

            self.actions_index[3*i] = k
            self.actions_index[3*i+1] = k2
            self.actions_index[3*i+2] = "-" + k

        # Utilities: a dictionary of indices and the length of the action set
        self.actions_index_reverse = {v: k for k, v in self.actions_index.items()}
        self.actions_dict_length = len(self.actions_dict)

    def possible_actions(self) -> list[int]:
        actions_list = list(np.arange(self.actions_dict_length))
        if len(self.path) == 0:
            return actions_list

        # Prevent the reverse of the action we just took from being selected right away
        last_move = self.path[-1]

        # For cube puzzles, it is also suboptimal to follow, e.g, a "d0" move with another "d0" move.
        # Furthermore, all "d" moves occur in the same plane, so we want to avoid the case where, e.g., a "d0" move
        # is followed by a "d1" is followed by another "d0" move. In addition, the "d0" and "d1" moves can be taken
        # in either order to reach the same state. To force an algorithm to choose one order over the other, we
        # force subsequent "d" moves to have a higher following integer, e.g., "d0" -> "d1" -> "d2" -> "d3".
        # (The reverse order would also be an acceptable option if desired.)
        # Therefore, e.g., if the previous move was in the "d1" plane, do not allow the next move to be in the "d0" or "d1" planes.
        # The same logic applies to the other planes, i.e., "f" and "l" moves.
        plane_rotated = last_move[-2]
        plane_rotated_num = int(last_move[-1])
        remove_list = [k for k in self.actions_dict.keys() if (plane_rotated in k and int(k[-1]) <= plane_rotated_num)]

        for move in remove_list:
            actions_list.remove(self.actions_index_reverse[move])

        return actions_list


class GlobePuzzle(PermutationPuzzle):
    def __init__(self, puzzle_series: pd.Series):
        assert "globe" in puzzle_series["puzzle_type"]
        super().__init__(puzzle_series)

        # The rotations of the lateral layers of the globe puzzle are similar to the ring rotations of the wreath puzzle
        # For example, each layer of the 3x4 globe has 7 possible rotations: -3r0, -2r0, -r0, r0, 2r0, 3r0, and 4r0
        # Notice that we have 4 rotations in one direction and (4-1) in the other direction
        n_layers = self.dimensions[0] + 1
        n_moves_per_rot = 2*self.dimensions[1] - 1
        n_layerwise_rotations = n_layers * n_moves_per_rot
        max_rot = self.dimensions[1]
        min_rot = -1 * max_rot + 1
        perm_nums = list(range(min_rot, max_rot + 1))
        perm_nums.remove(0)

        for i, (k, v) in enumerate(self.base_actions_dict.items()):
            perm = np.array(v)
            if "r" in k:
                inv_perm = np.argsort(perm)

                for q, j in enumerate(perm_nums):
                    if j == 1:
                        new_k = k
                        new_v = perm.copy()
                    elif j == -1:
                        new_k = "-" + k
                        new_v = inv_perm.copy()
                    else:
                        new_k = str(j) + k
                        if j > 0:
                            new_v = perm.copy()
                            for _ in range(2, j+1):
                                new_v = new_v[perm]
                        else:
                            new_v = inv_perm.copy()
                            for _ in range(2, -j + 1):
                                new_v = new_v[inv_perm]

                    self.actions_dict[new_k] = new_v
                    self.actions_index[n_moves_per_rot*i + q] = new_k
            elif "f" in k:
                # The negative of the flip move is the same as the flip move, so do not add the negative move to the dictionary
                # Further, in the puzzle description, the rotation moves are described before the flip moves. Therefore, since
                # order is preserved in the base dictionary, the following indexing will work.
                self.actions_dict[k] = perm
                self.actions_index[n_layerwise_rotations + i - n_layers] = k
            else:
                raise ValueError("Invalid action: {}".format(k))
            
        # Utilities: a dictionary of indices and the length of the action set
        self.actions_index_reverse = {v: k for k, v in self.actions_index.items()}
        self.actions_dict_length = len(self.actions_dict)

    def possible_actions(self) -> list[int]:
        actions_list = list(np.arange(self.actions_dict_length))
        if len(self.path) == 0:
            return actions_list

        # Prevent the reverse of the action we just took from being selected right away
        last_move = self.path[-1]

        # For globe puzzles, the lateral layers follow the same logic as the planes in the cube puzzles.
        # E.g., if the previous move was in the "r1" plane, do not allow the next move to be in the "r0" or "r1" planes.
        # However, order matters for the flip moves. E.g., "f0" followed by "f1" is not the same as "f1" followed by "f0".
        # Therefore, if the last move is a flip, the remove list contains only that move.
        layer_rotated = last_move[-2]
        layer_rotated_num = int(last_move[-1])
        if layer_rotated == "r":
            remove_list = [k for k in self.actions_dict.keys() if ("r" in k and int(k[-1]) <= layer_rotated_num)]
        elif layer_rotated == "f":
            remove_list = [last_move]
        else:
            raise ValueError("Invalid action found: {}".format(last_move))

        for move in remove_list:
            actions_list.remove(self.actions_index_reverse[move])

        return actions_list


def convert_action_str_to_base_actions(actions: Iterable[str] | str) -> str:
    # Convert a sequence of "complex" actions into the base actions
    # defined in the problem submission example
    if isinstance(actions, str):
        actions = actions.split(".")

    base_actions_str = ""
    for action in actions:
        # Split the action string into the number of repetitions and the action type
        split_action = re.split("d|f|l|r", action)
        num_reps = num_reps = 1 if split_action[0] in {"", "-"} else int(split_action[0])
        action_type = action[len(split_action[0]):]
        if num_reps < 0:
            base_actions_str += ("-" + action_type + ".") * (-num_reps)
        else:
            base_actions_str += (action_type + ".") * num_reps

    # Remove the trailing period
    return base_actions_str[:-1]


def make_forward_and_reverse_colour_maps(state: Iterable[str]) -> np.ndarray:
    if "N0" in state:
        int_set = np.arange(len(state), dtype=np.uint16)
        colour_dict = {f"N{k}": k for k in int_set}
    else:
        int_set = np.arange(len(set(state)), dtype=np.uint8)
        colour_dict = {k: v for k, v in zip(string.ascii_uppercase[:len(int_set)], int_set)}

    return colour_dict, {v: k for k, v in colour_dict.items()}


def make_complementary_puzzle(p: PermutationPuzzle) -> PermutationPuzzle:
    # Create a new puzzle with the initial and solution states swapped
    # The path is reset to an empty list
    new_p = p.__class__(puzzle_series=p.raw_series)
    new_p.initial_state = p.solution_state.copy()
    new_p.solution_state = p.initial_state.copy()
    new_p.state = new_p.initial_state.copy()
    new_p.path = []
    return new_p
    

def play_n_random_moves(p: PermutationPuzzle, n: int) -> dict[str, int]:
    # Play n random moves on a puzzle and keep track of the "value" of
    # each position along the way.
    # We use a simple reward function: +1 if the move results in a solution and
    # -0.001 for every base action (e.g., quarter turn of a cube face) taken.
    # The idea is that the learning algorithm is penalized slightly for every action
    # taken, but is rewarded generously for reaching the solution.
    new_value = 1.0
    value_dict = {}
    for _ in range(n):
        action = np.random.choice(p.possible_actions())
        action_str = p.actions_index[action]
        # print(action_str)   # Test code

        # Determine the number of "base_action" moves occur in the action to update the value of the state
        split_action = re.split("d|f|l|r", action_str)
        num_reps = 1 if split_action[0] in {"", "-"} else int(split_action[0])
        action_value = 0.001 * num_reps if num_reps > 0 else -0.001 * num_reps
        new_value -= action_value

        p.take_action(action)
        if (s := p.get_state_as_str()) not in value_dict:
            value_dict[s] = new_value
    
    return value_dict

def play_all_moves_to_depth_d(p: PermutationPuzzle, d: int) -> dict[str, int]:
    # Play all possible moves to depth d and keep track of the "value" of
    # each position along the way.
    new_value = 1.0
    value_dict = {}
    for _ in range(d):
        actions = p.possible_actions()
        # TODO: Figure out how to keep track of value and global dictionary for each action taken so far

        for action in actions:
            # Determine the number of "base_action" moves occur in the action to update the value of the state
            split_action = re.split("d|f|l|r", action)
            num_reps = 1 if split_action[0] in {"", "-"} else int(split_action[0])
            action_value = 0.001 * num_reps if num_reps > 0 else -0.001 * num_reps
            new_value -= action_value

            p.take_action_by_name(action)
            if (s := p.get_state_as_str()) not in value_dict:
                value_dict[s] = new_value
    
    return value_dict

if __name__ == '__main__':
    # root = '/kaggle/input/santa-2023/'
    root = 'competition_data'
    puzzles_df = pd.read_csv(os.path.join(root, 'puzzles.csv'))
    puzzle_info_df = pd.read_csv(os.path.join(root, 'puzzle_info.csv'))
    base_sub_df = pd.read_csv(os.path.join(root, 'sample_submission.csv'))
    super_df = pd.merge(puzzles_df, puzzle_info_df, on='puzzle_type')
    super_df = pd.merge(super_df, base_sub_df, on='id')

    # Class unit tests
    test_cube = CubePuzzle(super_df.loc[super_df['puzzle_type'] == 'cube_3/3/3', :].iloc[0])
    test_globe = GlobePuzzle(super_df.loc[super_df['puzzle_type'] == 'globe_1/8', :].iloc[0])
    test_wreath = WreathPuzzle(super_df.loc[super_df['puzzle_type'] == 'wreath_7/7', :].iloc[0])

    test_cube_rev = make_complementary_puzzle(test_cube)
    # print("Initial puzzle:")
    # print(test_cube.initial_state)
    # print(test_cube.solution_state)
    # print("Reversed puzzle:")
    # print(test_cube_rev.initial_state)
    # print(test_cube_rev.solution_state)

    test_scramble = play_n_random_moves(test_cube_rev, 60)
    print(len(test_scramble))
    # print(test_scramble)
