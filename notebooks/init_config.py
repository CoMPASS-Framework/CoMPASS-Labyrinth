from pathlib import Path
import os
 # =============================================================================
# FILE PATHS/INITIALIZATIONS
# =============================================================================

##### CHANGE THE PATHS AS PER USER LOCAL DRIVE

BASE_PATH = r'D:\Gladstone Dropbox\Palop Lab\Patrick\DeepLabCut Projects\AppSAA_DSI_CoMPASS_Test'  # TODO: CHANGE THIS TO YOUR BASE PATH LOCATION

# Location of original raw video locations from 2 computers (Copy original videos to central VIDEOFILE_PATH location)
# Specific to Palop Lab, IGNORE FOR MOST CASES
VIDEO_PATH_1 = r'D:\Gladstone Dropbox\Palop Lab\Patrick\Machine Learning Behavioral Analysis\Labyrinth\Noldus\20251019_AppSAA_DSI_Labyrinth\Media Files'
# VIDEO_PATH_2 = ''
print(f"Location of Computer 1 Videos: {VIDEO_PATH_1}")
# print(f"Location of Computer 2 Videos: {VIDEO_PATH_2}")

# Central video location (where all videos are copied for processing)
VIDEOFILE_PATH = os.path.join(BASE_PATH, 'videos', 'original_videos')
print(f"Central Video Location: {VIDEOFILE_PATH}")

# DeepLabCut CONFIG PATH, if running DLC from Palop labyrinth 'supernetwork'
DLC_CONFIG_PATH = Path(r'D:\Gladstone Dropbox\Palop Lab\Patrick\Machine Learning Behavioral Analysis\Labyrinth\DeepLabCut Projects\Labyrinth-Nick-2023-03-13\config.yaml') # Specific to Palop Lab, IGNORE FOR MOST CASES

# Pose estimation CSV outputs filepath
POSE_EST_CSV_PATH = os.path.join(BASE_PATH, 'data', 'dlc_results') #Don' edit

# Path for all grid based files for a particular Session, as part of Level-1 Post-Analysis --> Plot 1: Heatmap Representations of HMM States
GRID_PATH = os.path.join(BASE_PATH, 'data', 'grid_files') # Don't edit

# Metadata file paths
METADATA_PATH = os.path.join(BASE_PATH, 'data', 'metadata') # Don't edit
METADATA_FILE = os.path.join(METADATA_PATH,'20250725_LG124KI3_Cohort4_DLC_InfoSheet_v1.xlsx') # TODO: EDIT

# Trial configuration
TRIAL_TYPE = 'Labyrinth_DSI' # TODO: EDIT
FILE_EXT = '.csv'
VIDEO_TYPE = '.mp4'

# DLC scorer (DLC file subtext)
DLC_SCORER = 'DLC_resnet50_LabyrinthMar13shuffle1_1000000' # TODO: EDIT
print(f"DLC Scorer: {DLC_SCORER}")
BODYPARTS = ['nose', 'belly', 'sternum', 'leftflank', 'rightflank', 'tailbase']
print(f"Tracking bodyparts: {', '.join(BODYPARTS)}")

# Experimental groups
GROUPS = ['WT', 'AppSAA'] # TODO: EDIT THIS - Define your experimental groups
print(f"Experimental groups: {GROUPS}")

# Specify color of plot in graph
PALETTE=['grey'] # TODO: EDIT


# =============================================================================
# PRE-FIXED VALUES (DO NOT EDIT)
# =============================================================================


#-------------------- REGION---------------------------#
# Map Grid Nodes to Regions
Target_Zone=[84,85,86]
Entry_Zone=[47,46]
Loops=[33,45,57,58,59,71,83,95,70,69,68,56,44,78,79,80,81,82,94,106,118,105,93,92,104,116,117,91,90,52,53,41,42,43,55,67,54,66,65,64,38,37,36,25,24,13,12,0,1]
Neutral_Zone=[107,119,131,143]

# P1, P2, P3 are the 3 sections of the Reward Path
P1=[22,21,34,20,32,31,30,29,17,5,4,3,2,14,26]
P2=[27,39,51,63,62,61,60,72,73,74,75,76,77,89,101]
P3=[102,103,115,114,113,125,137,136,135,123,111,110,109,108,96,97,98]
Reward_Path=P1+P2+P3

Left_Dead_Ends=[10,11,23,35,9,8,6,7,19,18,15,16,28,40,50,49,48]   #Left Dead Ends
Right_Dead_Ends=[128,129,130,142,141,140,139,127,126,138,87,88,100,112,124,99,122,134,121,133,132,120]   #Right Dead Ends
Dead_Ends=Left_Dead_Ends+Right_Dead_Ends
#------------------------------------------------------#



#-------CHOOSE REGION NAMES (KEY), MAPPED TO LIST OF GRID NODES IN THAT REGION (VALUE)--------#
region_mapping = {
    'Target Zone': Target_Zone,
    'Entry Zone': Entry_Zone,
    'Reward Path': Reward_Path,
    'Dead Ends': Dead_Ends,
    'Neutral Zone': Neutral_Zone,
    'Loops': Loops
}
#---------------------------------------------------------------------------------------------#



#-------CHOOSE REGION NAMES (KEY), MAPPED TO LENGTH OF GRID NODES IN THAT REGION (VALUE)--------#
region_lengths = {
    'Entry Zone': len(Entry_Zone),
    'Loops': len(Loops),
    'Dead Ends': len(Dead_Ends),
    'Neutral Zone': len(Neutral_Zone),
    'Reward Path': len(Reward_Path),
    'Target Zone': len(Target_Zone)
}
#------------------------------------------------------------------------------------------------#


#----------------NODE-TYPES-------------------------#
Decision_Reward=[20,32,17,14,39,51,63,60,77,89,115,114,110,109,98]
NonDecision_Reward=[34,21,31,30,4,3,62,61,73,74,75,76,102,125,136,123,97]
Corner_Reward=[22,29,5,2,26,27,72,101,103,113,137,135,111,108,96]
Decision_NonReward=[100,71,12,24,42,106,92,119]
NonDecision_NonReward=[35,23,18,15,28,49,127,140,141,129,126,122,121,
99,112,45,58,70,69,83,56,44,13,38,52,64,65,54,55,78,79,80,81,94,91,90,104,131]
Corner_NonReward=[11,10,9,8,6,7,19,16,40,48,50,139,142,130,128,138,134,133,132,120,88,87,124,
33,57,59,95,68,0,1,36,25,37,66,53,41,43,67,82,105,93,116,117,118,107,143]
Entry_Zone=[47,46]
Target_Zone=[84,85,86]

Decision_3way=[20,17,39,51,63,60,77,89,115,114,110,109,98]
Decision_4way=[32,14]

#---------------------------------------------------------#
