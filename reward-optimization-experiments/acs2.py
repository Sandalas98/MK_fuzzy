#%%
from lcs.agents.acs2 import Configuration, ACS2

#%%
cfg = Configuration(2, 2)
agent = ACS2(cfg=cfg)

#%%
print(cfg.chi)
