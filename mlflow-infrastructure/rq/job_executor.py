import subprocess


def job(cmd):
    return subprocess.run(cmd, shell=True).returncode