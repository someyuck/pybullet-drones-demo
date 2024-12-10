# pybullet-drones-demo
A demonstration of multidrone control using PyBullet and the gym-pybullet-drones package.

## Dependencies

`gym-pybullet-drones` is used as the simulation environment, as it provides easy-to-use control of
multiple drones in an agent-environment setting.

Install it by following the steps in the [official repository](https://github.com/utiasDSL/gym-pybullet-drones).
**Note:** Use the `main` branch, and not the `master` branch.

## Running the simulation

After activating the virtual environment, run the simulations as follows:

### One drone reaching a target point

```bash
python -m src.simple_path --num_drones=1
```

### Two drones reaching their target points

```bash
python -m src.simple_path --num_drones=2
```

### Four drones reaching consensus

- Starting with all drones in a line: `python -m src.swarm --init_pos=0`
- Starting with all drones in a square shape: `python -m src.swarm --init_pos=1`
- Starting with all drones in random positions: `python -m src.swarm --init_pos=2`


## Cleanup

Run ```bash clean.sh``` to remove the saved results and videos.
