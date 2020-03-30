import pandas as pd

def parse_experiments_results(explore, exploit, metrics_trial_freq):
    explore_df = pd.DataFrame(explore)
    exploit_df = pd.DataFrame(exploit)

    explore_df['phase'] = 'explore'
    exploit_df['phase'] = 'exploit'

    df = pd.concat([explore_df, exploit_df], ignore_index=True)
    df['trial'] = df.index * metrics_trial_freq
    df.set_index('trial', inplace=True)
    return df
