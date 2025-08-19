# =============================================================================
# FILE PATHS/INITIALIZATIONS
# =============================================================================

BASE_PATH = r'C:\Users\PalopLabPortal\Gladstone Dropbox\Palop Lab\Shreya\Labyrinth Project Folders\TESTER FOR STREAMLINE_WT_DSI_Labyrinth'

# Metadata file paths
METADATA_PATH = r'C:\Users\PalopLabPortal\Gladstone Dropbox\Palop Lab\Shreya\Labyrinth Project Folders\TESTER FOR STREAMLINE_WT_DSI_Labyrinth'
METADATA_FILE = '20241028_WT_DSI_Labyrinth_DLC_InfoSheet_v1.xlsx'  

# Trial configuration
TRIAL_TYPE = 'Labyrinth_DSI' 

# DLC scorer (DLC file subtext)
DLC_SCORER = 'DLC_resnet50_LabyrinthMar13shuffle1_1000000'

# Specify color of plot in graph
PALETTE = ['grey']

# Specify Value map path # In the Resources folder, the file can be found
VALUE_MAP_PATH = r"C:\Users\PalopLabPortal\Gladstone Dropbox\Palop Lab\Shreya\Reinforcement learning\Value_Function_perGridCell.csv"   


# =============================================================================
# PRE-FIXED VALUES
# =============================================================================

#---------------------------INITIALIZATIONS--------------------------#

# Dictionary of each reference grid node mapped to a list of nodes 

CLOSE_REF={11:10,
47:[11,23,35],
46:47,
22:[34,46],
20:[22,21,8,32,44,56,68],
32:[20,33,71,44,56,68],
8:9,
33:[45,57],
57:[58,59],
59:[83,95],
68:[69,70],
29:[30,31,32],
5:[17,29],
7:6,
19:7,
17:[18,19],
2:[3,4,5],
26:[2,14],
16:[28,40],
14:[15,16,12,13,24],
0:1,
12:[36,0],
39:[27,37,38,25],
27:26,
51:[27,39,52,53,42],
63:[51,64,65,66,54],
53:41,
41:43,
43:[55,67],
60:[61,62,63],
72:[48,60],
48:[49,50],
77:[72,73,74,75,76,78,79,80,81,82],
101:[77,89],
89:[90,91,92,93,94,106],
92:[104,116],
116:117,
93:105,
106:118,
118:119,
119:[107,131,143],
103:[101,102],
115:[103,127,139],
139:[140,141,142],
142:130,
130:[128,129],
113:[114,115],
114:[126,138],
137:[113,125],
135:[136,137],
111:[123,135],
108:[111,110,109],
110:[122,134],
132:120,
133:132,
109:[121,133],
96:108,
98:[96,97,99,100],
88:87,
100:[88,112,124],
86:98,
84:[85,86,84]
}


NODES=range(0,144)
POS_X=[11,23,35,107,130,8,20,7,103,41,77,89,113,125,88,27,39,51,2,14,25,0,48,60,120]
POS_Y=[10,6,101,76,75,87,26,38,74,13,25,37,73,12,24,72,132]
NEG_X=[83,95,131,143,46,34,118,57,45,105,44,56,68,104,116,55,67,127,139,126,138,17,29,28,40,112,124,123,135,98,122,134,121,133,108,36]
NEG_Y=[x for x in NODES if x not in POS_X+POS_Y+NEG_X]

X_Y_MAPPING={
   'pos_x':POS_X,
    "pos_y":POS_Y,
    "neg_x":NEG_X,
    "neg_y":NEG_Y
}


