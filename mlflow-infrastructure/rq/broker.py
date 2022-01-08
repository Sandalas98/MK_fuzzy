import itertools
import os
import random

from redis import Redis
from rq import Queue

from job_executor import job

modulos = [4, 8, 16, 32]
rmpxes = [3, 6, 11, 20]
hashes = ['md5', 'sha256']
algorithms = ['acs', 'acs2', 'acs2ga', 'yacs']

combinations = list(itertools.product(algorithms, rmpxes, modulos, hashes))
print(f'Got {len(combinations)} combinations')

combinations = [
    ('acs2', 11, 4, 'sha256'),
    ('acs2', 11, 4, 'md5'),

    # ('acs2', 20, 4, 'sha256'),
    # ('acs2', 20, 4, 'md5'),
    #
    # ('acs2', 20, 8, 'sha256'),
    # ('acs2', 20, 8, 'md5'),
]

random.shuffle(combinations)


experiment = 1
trials = 50_000

base_cmd = 'docker run --rm --env-file .env --network="host" khozzy/alcs-rmpx-experiments:1 --no-conda'
commands = []

for idx, (alg, rmpx, modulo, hash) in enumerate(combinations):
    commands.append(f"{base_cmd} -P trials={trials} -P rmpx-size={rmpx} -P hash={hash} -P agent={alg} -P modulo={modulo} -P experiment={experiment}")


q = Queue(connection=Redis(os.getenv('REDIS_HOST', 'localhost'), password='z@4:vB34F+p[~L!'))

for cmd in commands:
    print(f'Adding to queue: [{cmd}]')
    q.enqueue(job, cmd, job_timeout='50h')
