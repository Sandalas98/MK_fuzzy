import pandas as pd
from lcs.agents.acs2 import ACS2, Configuration
from tqdm import tqdm


def parse_experiments_results(explore, exploit, metrics_trial_freq):
    explore_df = pd.DataFrame(explore)
    exploit_df = pd.DataFrame(exploit)

    explore_df['phase'] = 'explore'
    exploit_df['phase'] = 'exploit'

    df = pd.concat([explore_df, exploit_df], ignore_index=True)
    df['trial'] = df.index * metrics_trial_freq
    df.set_index('trial', inplace=True)
    return df


def start_single_experiment(env, explore_trials, exploit_trials, **kwargs):
    # Prepare the environment
    env.reset()

    cfg = Configuration(**kwargs)

    explorer = ACS2(cfg)
    metrics_explore = explorer.explore(env, explore_trials)

    exploiter = ACS2(cfg, explorer.population)
    metrics_exploit = explorer.exploit(env, exploit_trials)

    # Parse results into DataFrame
    df = parse_experiments_results(metrics_explore, metrics_exploit,
                                   cfg.metrics_trial_frequency)

    return df


def avg_experiments(n, env, explore_trials, exploit_trials, **kwargs):
    dfs = []
    print(f"{kwargs}\n")

    for i in tqdm(range(n), desc='Experiment', disable=n == 1):
        df = start_single_experiment(env, explore_trials, exploit_trials, **kwargs)
        dfs.append(df)

    bar = pd.concat(dfs)
    perf_df = bar.groupby(['trial', 'phase']).mean().reset_index(level='phase')

    return perf_df
