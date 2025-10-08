# =============================================================================
# PRE-FIXED VALUES (DO NOT EDIT)
# =============================================================================

# -------------------- REGION---------------------------#
# Map Grid Nodes to Regions
TARGET_ZONE = [84, 85, 86]
ENTRY_ZONE = [47, 46]
LOOPS = [
    33,
    45,
    57,
    58,
    59,
    71,
    83,
    95,
    70,
    69,
    68,
    56,
    44,
    78,
    79,
    80,
    81,
    82,
    94,
    106,
    118,
    105,
    93,
    92,
    104,
    116,
    117,
    91,
    90,
    52,
    53,
    41,
    42,
    43,
    55,
    67,
    54,
    66,
    65,
    64,
    38,
    37,
    36,
    25,
    24,
    13,
    12,
    0,
    1,
]
NEUTRAL_ZONE = [107, 119, 131, 143]

# P1, P2, P3 are the 3 sections of the Reward Path
P1 = [22, 21, 34, 20, 32, 31, 30, 29, 17, 5, 4, 3, 2, 14, 26]
P2 = [27, 39, 51, 63, 62, 61, 60, 72, 73, 74, 75, 76, 77, 89, 101]
P3 = [102, 103, 115, 114, 113, 125, 137, 136, 135, 123, 111, 110, 109, 108, 96, 97, 98]
REWARD_PATH = P1 + P2 + P3

LEFT_DEAD_ENDS = [10, 11, 23, 35, 9, 8, 6, 7, 19, 18, 15, 16, 28, 40, 50, 49, 48]  # Left Dead Ends
RIGHT_DEAD_ENDS = [
    128,
    129,
    130,
    142,
    141,
    140,
    139,
    127,
    126,
    138,
    87,
    88,
    100,
    112,
    124,
    99,
    122,
    134,
    121,
    133,
    132,
    120,
]  # Right Dead Ends
DEAD_ENDS = LEFT_DEAD_ENDS + RIGHT_DEAD_ENDS
# ------------------------------------------------------#

# -------CHOOSE REGION NAMES (KEY), MAPPED TO LIST OF GRID NODES IN THAT REGION (VALUE)--------#
REGION_MAPPING = {
    "target_zone": TARGET_ZONE,
    "entry_zone": ENTRY_ZONE,
    "reward_path": REWARD_PATH,
    "dead_ends": DEAD_ENDS,
    "neutral_zone": NEUTRAL_ZONE,
    "loops": LOOPS,
}

# -------CHOOSE REGION NAMES (KEY), MAPPED TO LENGTH OF GRID NODES IN THAT REGION (VALUE)--------#
REGION_LENGTHS = {
    "entry_zone": len(ENTRY_ZONE),
    "loops": len(LOOPS),
    "dead_ends": len(DEAD_ENDS),
    "neutral_zone": len(NEUTRAL_ZONE),
    "reward_path": len(REWARD_PATH),
    "target_zone": len(TARGET_ZONE),
}

# ----------------NODE-TYPES-------------------------#
DECISION_REWARD = [20, 32, 17, 14, 39, 51, 63, 60, 77, 89, 115, 114, 110, 109, 98]
NONDECISION_REWARD = [34, 21, 31, 30, 4, 3, 62, 61, 73, 74, 75, 76, 102, 125, 136, 123, 97]
CORNER_REWARD = [22, 29, 5, 2, 26, 27, 72, 101, 103, 113, 137, 135, 111, 108, 96]
DECISION_NONREWARD = [100, 71, 12, 24, 42, 106, 92, 119]
NONDECISION_NONREWARD = [
    35,
    23,
    18,
    15,
    28,
    49,
    127,
    140,
    141,
    129,
    126,
    122,
    121,
    99,
    112,
    45,
    58,
    70,
    69,
    83,
    56,
    44,
    13,
    38,
    52,
    64,
    65,
    54,
    55,
    78,
    79,
    80,
    81,
    94,
    91,
    90,
    104,
    131,
]
CORNER_NONREWARD = [
    11,
    10,
    9,
    8,
    6,
    7,
    19,
    16,
    40,
    48,
    50,
    139,
    142,
    130,
    128,
    138,
    134,
    133,
    132,
    120,
    88,
    87,
    124,
    33,
    57,
    59,
    95,
    68,
    0,
    1,
    36,
    25,
    37,
    66,
    53,
    41,
    43,
    67,
    82,
    105,
    93,
    116,
    117,
    118,
    107,
    143,
]
ENTRY_ZONE_NODES = [47, 46]
TARGET_ZONE_NODES = [84, 85, 86]
DECISION_3WAY = [20, 17, 39, 51, 63, 60, 77, 89, 115, 114, 110, 109, 98]
DECISION_4WAY = [32, 14]

NODE_TYPE_MAPPING = {
    "decision_reward": DECISION_REWARD,
    "nondecision_reward": NONDECISION_REWARD,
    "corner_reward": CORNER_REWARD,
    "decision_nonreward": DECISION_NONREWARD,
    "nondecision_nonreward": NONDECISION_NONREWARD,
    "corner_nonreward": CORNER_NONREWARD,
    "entry_zone": ENTRY_ZONE_NODES,
    "target_zone": TARGET_ZONE_NODES,
    "decision_3way": DECISION_3WAY,
    "decision_4way": DECISION_4WAY,
}
