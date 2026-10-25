
# List all the files in a folder
import argparse
import json
import joblib
import math
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
from typing import List, Tuple


def split_into_blocks(csv_path):
    """Splits a CSV file into blocks based on game numbers.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        List[List[str]]: A list of blocks, where each block is a list of strings (lines). Block 0 is the header.
    """
    blocks = []
    current_block = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            # Strip newline but keep commas structure
            stripped = line.rstrip("\n")
            if not stripped.strip(): 
                # Skip completely empty lines
                continue

            # Try to detect the start of a new block
            parts = stripped.split(",")
            first_cell = parts[0].strip()

            # A block starts when first cell is an integer game number (1,2,...)
            if first_cell.isdigit():
                # If there is an existing block, save it
                if current_block:
                    blocks.append(current_block)
                # Start a new block
                current_block = [stripped]  # Start new block
            else:
                # Inside a block → append line
                if current_block is not None:
                    current_block.append(stripped)

        # Append last block if exists
        if current_block:
            blocks.append(current_block)

    return blocks
        
def parse_line(line: str) -> Tuple[int, str]:
    """Parses a line from the CSV file.

    Args:
        line (str): A line from the CSV file.

    Returns:
        Tuple[int, str]: A tuple containing the value and assignment.
    """
    parts = line.split(",")
    value = int(parts[9].strip())
    assignment = parts[10].strip()
    return value, assignment

def compute_value(assignments_player1:List[Tuple[int, str]], assignments_player2:List[Tuple[int, str]]) -> Tuple[float, float]:
    """Computes the values for both players based on their assignments.

    Args:
        assignments_player1 (List[Tuple[int, str]]): The assignments for player 1.
        assignments_player2 (List[Tuple[int, str]]): The assignments for player 2.

    Returns:
        Tuple[float, float]: The computed values for both players.
    """
    assert len(assignments_player1) == len(assignments_player2)
    v_player1, v_player2 = 0.0, 0.0
    for a1, a2 in zip(assignments_player1, assignments_player2):
        if a1[1] != a2[1]:
            v_player1 -= 0.2 * a1[0]
            v_player2 -= 0.2 * a1[0]
        elif a1[1] == a2[1] and a1[1] == "r1":
            v_player1 += a1[0]
        elif a1[1] == a2[1] and a1[1] == "r2":
            v_player2 += a1[0]

    return (v_player1, v_player2)

def game_signature(csv_file: str) -> str:
    """ 
    Generates a signature for the game based on the CSV file.
    It can be used to check whether two games can be compared.
    It extracts the X,Y position of each disk and interaction and concatenate them.
    Args:
        csv_file (str): The path to the CSV file.
    Returns:
        str: The game signature.
    """
    blocks = split_into_blocks(csv_file)[1:]  # Skip header block
    signature = ""
    for b in blocks:
        for row in b[3:]:  # Skip first 3 rows of each block
            row_split = row.split(',')
            x, y = row_split[-4], row_split[-3]  # XY positions in the original string
            signature += f"{x},{y};"
    return signature

def extract_svo_angle(csv_path: str) -> float:
    """
    Extracts the Player SVO angle (in degrees) from a CSV file.

    The function assumes the SVO angle is stored on a separate line
    as the first value, followed by descriptive text.
    """
    with open(csv_path, "r") as f:
        for line in f:
            if "SVO angle" in line:
                return float(line.split(",")[0])

    raise ValueError("SVO angle not found in file.")

def adaptive_agent_from_file(
    model_path: str = "./models/adaptive_agent_model.joblib", 
    X: np.ndarray = None
    ) -> int:
    model = joblib.load(model_path)

    probabilities = model.predict_proba(X)
    # label = (probabilities[:, 1] >= 0.5).astype(int)  # 50% threshold

    return probabilities[0][1]

def extract_assignments(response: str) -> List[Tuple[List[int], str]]:
    blue_str = ("b", "r1")  # Blue
    yellow_str = ("y", "r2")  # Yellow
    results = []
    for ic, c in enumerate(response.lower().strip()):
        if c == yellow_str[0]:
            try:
                x, y = int(response[ic-2]), int(response[ic-1])
                results.append((f"{x},{y}", yellow_str[1]))
            except:
                pass
        elif c == blue_str[0]:
            try:
                x, y = int(response[ic-2]), int(response[ic-1])
                results.append((f"{x},{y}", blue_str[1]))
            except:
                pass

    return results

def player_files(folder_path: str, file_prefix: str = "") -> List[str]:
    files = os.listdir(folder_path)
    player_files = []
    for f in files:
        if file_prefix in f:
            player_files.append(f)
    return sorted(player_files)

def sample_pairs(
    player1_files: List[str], 
    player2_files: List[str], 
    num_samples: int, 
    with_replacement: bool = False
    ) -> List[Tuple[str, str]]:
    assert len(player1_files) > 0 and len(player2_files) > 0, f"Cannot sample pairs! len(player1_files)={len(player1_files)}, len(player2_files)={len(player2_files)}"
    sampled_pairs = []
    available_pairs = [(f1, f2) for f1 in player1_files for f2 in player2_files]
    
    if with_replacement:
        for _ in range(num_samples):
            sampled_pairs.append(random.choice(available_pairs))
    else:
        assert len(player1_files)*len(player2_files) >= num_samples, f"Cannot sample more pairs than available ({len(player1_files)*len(player2_files)}) without replacement!"
        sampled_pairs = random.sample(available_pairs, min(num_samples, len(available_pairs)))
    
    return sampled_pairs

def plot_pretty_boxplot(
    data1: List[float],
    data2: List[float],
    label1: str = "Player 1",
    label2: str = "Player 2",
    save_path: str = "",
    annotate_stats: bool = True,
    stats_decimals: int = 2,
    variant_label: str | None = None,
):
    # --- Font setup ---
    rcParams['font.family'] = 'Times New Roman'

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    # --- Figure and style setup ---
    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9f9f9")

    # Custom rich colors for the bars
    box_colors = ["#0D53AA", "#D45105"]  # Blue & Purple, elegant, strong

    bp = ax.boxplot(
        [data1, data2],
        vert=False,
        patch_artist=True,
        widths=0.3,
        boxprops=dict(edgecolor="#000000", linewidth=1.6),
        whiskerprops=dict(color="#000000", linewidth=1.1),
        capprops=dict(color="#000000", linewidth=1.8),
        medianprops=dict(color="none"),  # we’ll draw our own
        flierprops=dict(marker="", linestyle="none")
    )

    # --- Fill boxes and draw mean/median lines ---
    for i, (data, color) in enumerate(zip([data1, data2], box_colors)):
        box = bp['boxes'][i]
        box.set_facecolor(color)
        box.set_alpha(0.7)
        box.set_linewidth(1.8)

        verts = box.get_path().vertices
        y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
        box_height = y_max - y_min
        pad = 0.0 * box_height

        mean_v = np.mean(data)
        median_v = np.median(data)

        # Strong mean and median markers
        ax.plot([mean_v, mean_v], [y_min + pad, y_max - pad],
                color="#57EB3D", linewidth=1.3, zorder=5)
        ax.plot([median_v, median_v], [y_min + pad, y_max - pad],
                color="red", linewidth=1.3, zorder=6)

        # Add a subtle marker (dot) for mean
        ax.scatter(mean_v, (y_min + y_max)/2, 
                   color="#57EB3D", 
                   edgecolor="black", 
                   zorder=7, 
                   s=40)
        if annotate_stats:
            text_margin = 0.2 * box_height
            mean_text = f"{mean_v:.{stats_decimals}f}"
            median_text = f"{median_v:.{stats_decimals}f}"
            ax.text(
                mean_v,
                y_max + text_margin,
                mean_text,
                color="#57EB3D",
                fontsize=9,
                ha="center",
                va="bottom",
            )
            ax.text(
                median_v,
                y_min - text_margin,
                median_text,
                color="red",
                fontsize=9,
                ha="center",
                va="top",
            )

    # --- Axis styling ---
    ax.set_yticks([1, 2])
    ax.set_yticklabels([label1, label2], fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=11)
    # ax.set_title("Bargaining Table – Payoff Statistics", fontsize=14, fontweight="bold", pad=12)

    # --- Grid ---
    ax.grid(True, linestyle="--", linewidth=0.6, color="#b0b0b0", alpha=0.7)

    if variant_label:
        ax.text(
            0.02,
            0.98,
            variant_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="#000000",
                boxstyle="round,pad=0.3",
                alpha=0.8,
            ),
        )

    # --- X-limits with padding ---
    # all_data = np.concatenate([data1, data2])
    # min_v, max_v = np.min(all_data), np.max(all_data)
    min_v, max_v = -30., 80.
    rng = max_v - min_v if max_v > min_v else 1
    ax.set_xlim(min_v - 0.1 * rng, max_v + 0.1 * rng)

    # --- Legend ---
    ax.plot([], [], color="#57EB3D", linewidth=1.3, label="Mean")
    ax.plot([], [], color="red", linewidth=1.3, label="Median")
    ax.legend(loc="upper right", frameon=True, fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()

def compute_payoff(
    sampled_pairs: List[Tuple[str, str]],
    folder_path_player1: str,
    folder_path_player2: str,
    strategy: str = "humans"
) -> dict:
    """
    Computes the payoff for both players based on the sampled pairs of games and the chosen strategy.

    Args:
        sampled_pairs (List[Tuple[str, str]]): The sampled pairs of game files.
        folder_path_player1 (str): The folder path for player 1's game files.
        folder_path_player2 (str): The folder path for player 2's game files.
        strategy (str, optional): The strategy to use for computing payoffs (defaults to "humans"). 
            The strategy can be:
            - "humans" (based on data for both players), 
            - "p1_greedy" (player 1 is greedy, player 2 is based on his data), 
            - "p2_greedy" (player 2 is greedy, player 1 is based on his data), 
            - "both_greedy" (not very interesting, both players are greedy),
            - "p1_cooperative" (player 1 chooses the assignments based on the Euclidean distance from the disk).
            - "p2_cooperative" (player 2 chooses the assignments based on the Euclidean distance from the disk).
            - "p1_SVO" (player 1 chooses the assignments based on Social Value Orientation). Given the angle a, computes 
            t1 = cos(a)*v1 + sin(a)*v2 and t2 = sin(a)*v1 + cos(a)*v2, where vi is the utility if they keep the disk. 
            Then assigns to himself if t1 > t2, otherwise to the other.
            - "p2_SVO" (player 2 chooses the assignments based on Social Value Orientation). Same but for player 2.
            - "p1_adaptive" (player 1 uses an adaptive agent based on a trained model, player 2 is based on his data).
            - "p1_llm": use the json of the LLM as player 1.
            - "p2_llm": use the json of the LLM as player 2.
    Raises:
        ValueError: If the game signatures do not match.
        ValueError: If the strategy is unknown.

    Returns:
        dict: A dictionary containing the total payoffs for both players.
    """
    global DATAFRAME_P1_ADAPTIVE, LLM_AS_P1, LLM_AS_P2
    global ADAPTIVE_PLAYER1_JOBLIB
    global SVO_ANGLE
    # Load additional data if needed
    if strategy == "p1_adaptive":
        df_adaptive = pd.read_csv(DATAFRAME_P1_ADAPTIVE)
    elif strategy == "p1_llm":
        with open(LLM_AS_P1, "r") as f:
            data_llm = json.load(f)
    elif strategy == "p2_llm":
        with open(LLM_AS_P2, "r") as f:
            data_llm = json.load(f)
    
    
    llm_missing_match_exception = 0  # Debug for when an llm is employed 
    player1_total_payoff, player2_total_payoff = 0.0, 0.0
    overall_player1_payoff = {i:[] for i in range(1,11)}  # Store payoffs per game type
    overall_player2_payoff = {i:[] for i in range(1,11)}  # Store payoffs per game type
    overall_assignments = {}
    for game_num in range(1,11):
        count = 0
        player1_payoff, player2_payoff = 0.0, 0.0
        total_choices, suboptimal_choices = 0, 0
        pay_off_left = 0.0
        overall_assignments[game_num] = []
        for f1, f2 in sampled_pairs:
            sign_f1 = game_signature(os.path.join(folder_path_player1, f1))
            sign_f2 = game_signature(os.path.join(folder_path_player2, f2))
            if sign_f1 != sign_f2:
                raise ValueError(f"Game signatures do not match for files {f1} and {f2}!")

            whole_block1 = split_into_blocks(os.path.join(folder_path_player1, f1))[game_num]
            whole_block2 = split_into_blocks(os.path.join(folder_path_player2, f2))[game_num]
            
            # Positions of the players in "cooperative" strategy
            if strategy in ["p1_cooperative", "p2_cooperative", "p1_adaptive"]:
                p1x = int(whole_block1[1].split(',')[-2].strip())
                p1y = int(whole_block1[1].split(',')[-1].strip())
                p2x = int(whole_block2[2].split(',')[-2].strip())
                p2y = int(whole_block2[2].split(',')[-1].strip())
                # print(os.path.join(folder_path_player1, f1))
                # print(os.path.join(folder_path_player2, f2))
                # print(f"Player 1 position: ({p1x},{p1y}), Player 2 position: ({p2x},{p2y})")
            
            blocks1 = whole_block1[3:]  # Skip first 3 rows (header)
            blocks2 = whole_block2[3:]  # Skip first 3 rows (header)
            
            game_payoff_p1 = 0.
            game_payoff_p2 = 0.
            # Iterate through corresponding blocks of both players, according to the strategy
            for (i,r1),r2 in zip(enumerate(blocks1), blocks2):
                if strategy == "humans":
                    r1_v = r1.split(',')
                    r2_v = r2.split(',')
                    r1_t = r1_v[-1].strip()
                    r2_t = r2_v[-1].strip()
                    
                elif strategy in ["p1_greedy", "p2_greedy", "both_greedy"]:
                    r1_v = r1.split(',')
                    r2_v = r2.split(',')
                    if strategy == "p1_greedy":
                        r1_t = "r1"  # Player 1 always chooses r1
                        r2_t = r2_v[-1].strip()
                    elif strategy == "p2_greedy":
                        r1_t = r1_v[-1].strip()
                        r2_t = "r2"  # Player 2 always chooses r2
                    elif strategy == "both_greedy":
                        r1_t = "r1"  # Player 1 always chooses r1
                        r2_t = "r2"  # Player 2 always chooses r2
                        
                elif strategy in ["p1_cooperative", "p2_cooperative"]:
                    r1_v = r1.split(',')
                    r2_v = r2.split(',')
                    # Disk position
                    dx = int(r1_v[-4].strip())
                    dy = int(r1_v[-3].strip())
                    dist_p1_d = ((p1x - dx)**2 + (p1y - dy)**2)**0.5
                    dist_p2_d = ((p2x - dx)**2 + (p2y - dy)**2)**0.5
                    if strategy == "p1_cooperative":
                        if dist_p1_d >= dist_p2_d:
                            r1_t = "r2"
                        else:
                            r1_t = "r1"
                        r2_t = r2_v[-1].strip()
                    else:  # p2_cooperative
                        r1_t = r1_v[-1].strip()
                        if dist_p2_d >= dist_p1_d:
                            r2_t = "r1"
                        else:
                            r2_t = "r2"
                        r1_t = r1_v[-1].strip()
                    
                elif strategy in ["p1_SVO", "p2_SVO"]:
                    r1_v = r1.split(',')
                    r2_v = r2.split(',')
                    v1_r1, v2_r1 = int(r1_v[-2].strip()), int(r2_v[-2].strip())
                    # SVO angle fixed to SVO_ANGLE degrees
                    angle_rad = SVO_ANGLE * (math.pi / 180)
                    threshold_p1 = math.cos(angle_rad) * v1_r1 # + math.sin(angle_rad) * 0.
                    threshold_p2 = math.sin(angle_rad) * v2_r1  # + math.cos(angle_rad) * 0.
                    if strategy == "p1_SVO":
                        if threshold_p1 > threshold_p2:
                            r1_t = "r1"
                        else:
                            r1_t = "r2"
                        r2_t = r2_v[-1].strip()
                    else:
                        if threshold_p1 > threshold_p2:
                            r2_t = "r2"
                        else:
                            r2_t = "r1"
                        r1_t = r1_v[-1].strip()
                        
                elif strategy == "p1_adaptive":
                    r1_v = r1.split(',')
                    r2_v = r2.split(',')
                    
                    game_match_row = df_adaptive.loc[df_adaptive["game_number"] == f2].iloc[i]
                    adaptive_X = game_match_row[["x1", "x2", "x3", "x4", "x5", "x6"]].to_numpy().flatten().reshape(1, -1)
                    adaptive_Y = adaptive_agent_from_file(ADAPTIVE_PLAYER1_JOBLIB, adaptive_X)
                    r = random.random()
                    if adaptive_Y > r:
                        # Disk position
                        dx = int(r1_v[-4].strip())
                        dy = int(r1_v[-3].strip())
                        dist_p1_d = ((p1x - dx)**2 + (p1y - dy)**2)**0.5
                        dist_p2_d = ((p2x - dx)**2 + (p2y - dy)**2)**0.5
                        if dist_p1_d >= dist_p2_d:
                            r1_t = "r2"
                        else:
                            r1_t = "r1"
                        r2_t = r2_v[-1].strip()
                    else:
                        r1_t = "r1"
                    r2_t = r2_v[-1].strip()
                    
                elif strategy == "p1_llm":
                    r1_v = r1.split(',')
                    r2_v = r2.split(',')
                    
                    response = random.choice(data_llm[game_num-1]["responses"])
                    answer_reg = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
                    match = answer_reg.search(response)
                    
                    try:
                        llm_answers = extract_assignments(match.group(1).strip())
                        r1_t = llm_answers[i][1].strip()
                    except:
                        r1_t = "r1"
                        llm_missing_match_exception += 1
                    r2_t = r2_v[-1].strip()
                    
                elif strategy == "p2_llm":
                    r1_v = r1.split(',')
                    r2_v = r2.split(',')
                    
                    response = random.choice(data_llm[game_num-1]["responses"])
                    answer_reg = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
                    match = answer_reg.search(response)
                    
                    try:
                        llm_answers = extract_assignments(match.group(1).strip())
                        r1_t = llm_answers[i][1].strip()
                    except:
                        r1_t = "r2"
                        llm_missing_match_exception += 1
                    r2_t = r2_v[-1].strip()
                        
                else:
                    raise ValueError(f"Unknown mode: {strategy}")

                # Assign payoffs based on each player choices
                if r1_t == r2_t == "r1":
                    player1_payoff += int(r1_v[-2].strip())
                    game_payoff_p1 += int(r1_v[-2].strip())
                    game_payoff_p2 += 0.
                elif r1_t == r2_t == "r2":
                    player2_payoff += int(r2_v[-2].strip())
                    game_payoff_p1 += 0.
                    game_payoff_p2 += int(r2_v[-2].strip())
                else:
                    player1_payoff -= 0.2 * int(r1_v[-2].strip())
                    player2_payoff -= 0.2 * int(r2_v[-2].strip())
                    pay_off_left += int(r1_v[-2].strip())
                    suboptimal_choices += 1
                    game_payoff_p1 -= 0.2 * int(r1_v[-2].strip())
                    game_payoff_p2 -= 0.2 * int(r2_v[-2].strip())
                total_choices += 1
                
                # Overall assignments
                overall_assignments[game_num].append((r1_t, r2_t))
                
            overall_player1_payoff[game_num].append(game_payoff_p1)
            overall_player2_payoff[game_num].append(game_payoff_p2)
            count += 1

        player1_total_payoff += player1_payoff
        player2_total_payoff += player2_payoff
        percentage_suboptimal_choices = (suboptimal_choices / total_choices * 100) if total_choices > 0 else 0
        
        print("--------------------------------------------------")
        print(f"Strategy: {strategy}")
        print(f"Analyzed {count} pairs of games of type {game_num}.")
        print(f"\tAverage payoff for Player 1: {player1_payoff/count if count>0 else 0.0}")
        print(f"\tAverage payoff for Player 2: {player2_payoff/count if count>0 else 0.0}")
        print(f"\tNumber of suboptimal choices: {suboptimal_choices}/{total_choices} ({percentage_suboptimal_choices} %)")
        print(f"\tAverage payoff left: {pay_off_left/count if count>0 else 0.0}")
        print(f"\tPayoff ratio (P1/P2): {(player1_payoff/count)/(player2_payoff/count) if player2_payoff/count != 0 else 'inf'}")
        print(f"\tPayoff ratio (P2/P1): {(player2_payoff/count)/(player1_payoff/count) if player1_payoff/count != 0 else 'inf'}")
        print(f"[Debug] Missing match with LLMs: {llm_missing_match_exception}")
    
    # Compute the normalised coordination index (NCI) of each game and the average NCI
    ncis = []
    for game_num in range(1,11):
        denominator = len(overall_assignments[game_num]) * (len(overall_assignments[game_num]) - 1)
        m1 = m2 = m3 = 0
        for assign in overall_assignments[game_num]:
            a1, a2 = assign
            if a1 == a2 == "r1":
                m1 += 1
            elif a1 == a2 == "r2":
                m2 += 1
                
        nci = 2 * (m1 * (m1 - 1) + m2 * (m2 - 1) + m3 * (m3 - 1)) / denominator if denominator > 0 else -1.
        ncis.append(nci)

    print(f"Total payoff - across games - Player 1: {player1_total_payoff/(count)}, Player 2: {player2_total_payoff/(count)}")
    print(f"Average NCI across games: {np.mean(ncis)} ± {np.std(ncis)}")
    print("--------------------------------------------------")

    return {
        "mode": strategy,
        "samples_per_game": count,
        "player1_payoff": player1_payoff,
        "player2_payoff": player2_payoff,
        "suboptimal_choices": suboptimal_choices,
        "percentage_suboptimal_choices": percentage_suboptimal_choices,
        "total_choices": total_choices,
        "pay_off_left": pay_off_left,
        "NCI-per-game": ncis,
        "NCI": f"{np.mean(ncis)} ± {np.std(ncis)}",
        "overall_payoff_p1": overall_player1_payoff,
        "overall_payoff_p2": overall_player2_payoff
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Run sampling strategy")

    parser.add_argument(
        "--strategy",
        type=str,
        default="humans",
        help="Strategy to use (default: p1_adaptive)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of random pairs to sample (default: 30)",
    )

    parser.add_argument(
        "--sample-with-replacement",
        type=bool,
        default=False,
        help="Sample with replacement (default: False)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    
    parser.add_argument(
        "--file-player-as-llm",
        type=str,
        default="",
        help="The folder where LLM data is.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    Strategy can be:
        - "humans" (based on data for both players), 
        - "p1_greedy" (player 1 is greedy, player 2 is based on his data), 
        - "p2_greedy" (player 2 is greedy, player 1 is based on his data), 
        - "both_greedy" (not very interesting, both players are greedy),
        - "p1_cooperative" (player 1 chooses the assignments based on the Euclidean distance from the disk).
        - "p2_cooperative" (player 2 chooses the assignments based on the Euclidean distance from the disk).
        - "p1_SVO" (player 1 chooses the assignments based on Social Value Orientation).
        - "p2_SVO" (player 2 chooses the assignments based on Social Value Orientation).
        - "p1_adaptive" (player 1 uses an adaptive agent based on a trained model, player 2 is based on his data).
        - "p1_llm": use the json of the LLM as player 1.
        - "p2_llm": use the json of the LLM as player 2.
    """
    # Global parameters
    DATAFRAME_P1_ADAPTIVE = "./data/Dor-humans/bargaining_games_player_blue.csv"
    SVO_ANGLE = 22.5  # Angle of the SVO agent 
    FOLDER_PLAYER1_DATA = "./data/Dor-humans/stage-2-analysis/Number1players/"
    FOLDER_PLAYER2_DATA = "./data/Dor-humans/stage-2-analysis/Number2players/"
    PLOT_FOLDER = "./plots/bargaining_table"
    RESULTS_FOLDER = "./results/bargaining_table"
    ADAPTIVE_PLAYER1_JOBLIB = "./data/Dor-humans/bargaining_table_rf.joblib"
    # LLM_AS_P1 = "./data/bargaining_table_llms/blue/gpt-oss-120b/bargaining_table_realdata-vanilla_responses_problem.jsonl"
    # LLM_AS_P2 = "./data/bargaining_table_llms/orange/gpt-oss-120b/bargaining_table_realdata-vanilla_responses_problem.jsonl"

    
    strategy_to_label = {
        "humans": ["Orange Human", "Blue Human"],
        "p1_greedy": ["Orange Human", "Blue Greedy"],
        "p2_greedy": ["Orange Greedy", "Blue Human"],
        "both_greedy": ["Orange Greedy", "Blue Greedy"],
        "p1_cooperative": ["Orange Human", "Blue Cooperative"],
        "p2_cooperative": ["Orange Cooperative", "Blue Human"],
        "p1_SVO": ["Orange Human", "Blue SVO"],
        "p2_SVO": ["Orange SVO", "Blue Human"],
        "p1_adaptive": ["Orange Human", "Blue Adaptive"],
        "p1_llm": ["Orange Human", "Blue LLM"],
        "p2_llm": ["Orange LLM", "Blue Human"],
    }
    
    args = parse_args()

    strategy = args.strategy
    num_samples = args.num_samples
    sample_with_replacement = args.sample_with_replacement
    seed = args.seed

    LLM_AS_P1 = LLM_AS_P2 = args.file_player_as_llm

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print(f"Running sampling with strategy={strategy}, num_samples={num_samples}, sample_with_replacement={sample_with_replacement}, seed={seed}")

    # Retrieve all the games played by both players
    player1_files = player_files(FOLDER_PLAYER1_DATA, "BT")
    player2_files = player_files(FOLDER_PLAYER2_DATA, "BT")
    
    sampled_pairs = sample_pairs(
        player1_files,
        player2_files,
        num_samples,
        with_replacement=sample_with_replacement,
    )

    results = compute_payoff(
        sampled_pairs,
        FOLDER_PLAYER1_DATA,
        FOLDER_PLAYER2_DATA,
        strategy=strategy
    )

    # Save results in JSON format
    if strategy not in ["p1_llm", "p2_llm"]:
        filename_results = strategy
    else:
        filename_results = strategy
        # "./data/bargaining_table_llms/blue/gpt-oss-120b-high/bargaining_table_realdata-vanilla_responses_problem.jsonl"
        if "blue" in LLM_AS_P1:
            filename_results += "_blue"
        elif "yellow" in LLM_AS_P1:
            filename_results += "_yellow"

        if "gpt-oss" in LLM_AS_P1:
            if "120b" in LLM_AS_P1:
                filename_results += "_gpt-oss-120b"
            elif "20b" in LLM_AS_P1:
                filename_results += "_gpt-oss-20b"
                
            if "high" in LLM_AS_P1:
                filename_results += "-high"
            elif "medium" in LLM_AS_P1:
                filename_results += "-medium"
            elif "low" in LLM_AS_P1:
                filename_results += "-low"
                
            filename_results += "_" + LLM_AS_P1.split('/')[-1].split('.jsonl')[0]

    os.makedirs(f"{RESULTS_FOLDER}/{strategy}", exist_ok=True)
    with open(f"{RESULTS_FOLDER}/{strategy}/{filename_results}.json", "w") as f:
        json.dump(results, f)

    # Plot and save
    # Order games by (game_type, sample_number)
    games_p1 = np.array([v for v in results["overall_payoff_p1"].values()])
    games_p2 = np.array([v for v in results["overall_payoff_p2"].values()])
    
    # Reduce over the game_type axis
    sum_games_p1 = games_p1.sum(axis=0)
    sum_games_p2 = games_p2.sum(axis=0)

    os.makedirs(f"{PLOT_FOLDER}/{strategy}", exist_ok=True)
    plot_pretty_boxplot(sum_games_p1, 
                        sum_games_p2,
                        label1=strategy_to_label[strategy][1] + ": ",
                        label2=strategy_to_label[strategy][0] + ": ",
                        save_path=f"{PLOT_FOLDER}/{strategy}/{filename_results}.png")
