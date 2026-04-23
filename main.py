import pandas as pd
from CBM import community_based_rf

data = pd.read_csv("data.csv")

LABEL_COLS = ['next_hour_tv_used',
              'next_hour_atv_used',
              'next_hour_lights_used',
              'next_hour_sbar_used',
              ]

###############################################

# non_feature_cols = ['patient_id', 'date', '@timestamp', 'id', 'room_number',
#                     'function', 'group',
#                     'room_ltg_state', 'opaque_blind_state', 'blkout_blind_state',
#                     'season', 'room_facing', 'discharge_ts', 'current_channel',
#                     'next_hour_door_used', 'next_hour_hvac_used', 'next_hour_blinds_used'
#                     ] + LABEL_COLS
#
# feature_cols = [c for c in data.columns if c not in non_feature_cols]
#
# for col in feature_cols:
#     if col in data.columns:
#         data[col] = pd.to_numeric(data[col], errors='coerce')
#
# print(f"Total features: {len(feature_cols)}")
# print(feature_cols)
#
# community_results = community_based_rf(
#     df=data,
#     feature_cols=feature_cols,
#     label_cols=LABEL_COLS,
#     similarity_measurement=2,  # cosine
#     thresholds=[0.75, 0.85, 0.90, 0.95, 0.98],
#     n_iterations=2,
# )
#
# print(community_results)
# community_results.to_csv("results_2.csv")

################################################ aggregated data


data_aggr = pd.read_csv("data_aggr.csv")

non_feature_cols_aggr = ['patient_id', 'date', '@timestamp', 'id', 'room_number',
                         'function', 'group', 'month', 'day', 'year', 'minute',
                         'room_ltg_state', 'opaque_blind_state', 'blkout_blind_state',
                         'hour', 'day_of_week', 'is_weekend', 'month',
                         'next_hour_door_used', 'next_hour_hvac_used', 'next_hour_blinds_used'
                         ] + LABEL_COLS

feature_cols_aggr = [c for c in data_aggr.columns if c not in non_feature_cols_aggr]
for col in feature_cols_aggr:
    if col in data_aggr.columns:
        data_aggr[col] = pd.to_numeric(data_aggr[col], errors='coerce')

print(f"Total features: AGGR : {feature_cols_aggr}")

roll_cols = [c for c in data_aggr.columns if 'roll24h' in c or 'roll7d' in c]
data_aggr[roll_cols] = data_aggr[roll_cols].fillna(0)

community_results_aggr = community_based_rf(
    df=data_aggr,
    feature_cols=feature_cols_aggr,
    label_cols=LABEL_COLS,
    similarity_measurement=2,  # cosine
    thresholds=[0.75, 0.85, 0.90, 0.95, 0.98],
    n_iterations=5,
)

community_results_aggr.to_csv("results_aggr_3.csv")
