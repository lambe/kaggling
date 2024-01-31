# %%
import ast
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# root_dir = "/kaggle/input/santa-2023"
root_dir = "competition_data"

# %%
puzzles_df = pd.read_csv(os.path.join(root_dir, 'puzzles.csv'))
puzzle_info_df = pd.read_csv(os.path.join(root_dir, 'puzzle_info.csv'))
submission_df = pd.read_csv(os.path.join(root_dir, 'sample_submission.csv'))

# %%
# Count the total number of moves in the submission from each puzzle type
submission_df['total_moves'] = submission_df['moves'].apply(lambda x: len(x.split(".")))
submission_df['puzzle_type'] = puzzles_df.loc[submission_df['id']]['puzzle_type'].values
submission_df['num_puzzles'] = 1
submission_group_df = submission_df.groupby('puzzle_type').agg({'total_moves': 'sum', 'num_puzzles': 'sum'}).reset_index()
submission_group_df = submission_group_df.sort_values(by=['total_moves'], ascending=False)
submission_group_df

# %%
# Sort submission by total moves
submission_df_sorted = submission_df.sort_values(by=['total_moves'], ascending=False)
submission_df_sorted.iloc[250:300]

# %% [markdown]
# ## Let's try substituting moves based on the solutions given

# %%
def simplify_wreath_path(dimension: int, solution_path: str):
    solution_arr = solution_path.split(".")
    new_solution_arr = solution_path.split(".")
    j = 1
    for i in range(1, len(solution_arr)):
        if "-" + solution_arr[i] == solution_arr[i-1] or solution_arr[i] == "-" + solution_arr[i-1]:
            # Remove redundant move, i.e., pop the two elements
            new_solution_arr.pop(j)
            new_solution_arr.pop(j)
        else:
            # Bump the corresponding index in the new solution array
            j += 1
            
    return ".".join(new_solution_arr)

# %%
def simplify_path(puzzle_type: str, dimensions: list[int], solution_path: str):
    if puzzle_type == 'wreath':
        return simplify_wreath_path(dimensions[0], solution_path)
    elif puzzle_type == 'cube':
        return solution_path
    elif puzzle_type == 'globe':
        return solution_path
    else:
        raise ValueError(f"Unknown puzzle type {puzzle_type}")
    
# %% [markdown]
# ## Now run the Examine the wreath puzzle

# %%
test_wreath_series = puzzles_df.iloc[284]


# %%
test_wreath_A = PermutationPuzzle(test_wreath_series)
test_wreath_B = PermutationPuzzle(test_wreath_series)

# %%
test_wreath_A.possible_actions_by_name()

# %%
# Apply 3 left ring rotations
for _ in range(4):
    test_wreath_A.take_action(2)
for _ in range(2):
    test_wreath_B.take_action(3)
print(test_wreath_A.get_state())
print(test_wreath_B.get_state())

# %%
test_wreath_A.reset_state()
test_wreath_B.reset_state()


