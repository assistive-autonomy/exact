# Executable Activity Models


## Installation

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. 


## Usage

Run a random policy:

```bash
uv run scripts./random_policy.py 
```

Run a policy with a custom reward:

```bash
uv run scripts./reward_policy.py --reward move-ego-0-2
```



