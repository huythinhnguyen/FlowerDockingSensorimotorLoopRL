# Experiment Description
## [Exp 1] 1 bat 1 flower, randomized init configs
### Description
Initialize bat pose `(x_bat_init, y_bat_init, theta_bat_init)`:
- `(x_bat_init, y_bat_init)` is fixed to `(0.,0.)`.
- `theta_bat_init` is randomized uniformly within $[-60\degree, 60\degree)$ or $[-\frac{\pi}{3}, \frac{\pi}{3})$.
    - &rarr; Bat will always have the flower within its field of view.
Initialize flower pose `(x_flower0, y_flower0, theta_flower0)`:
- `(x_flower0, y_flower0)` is fixed to `(FLOWER_DISTANCE, 0.)`
- `theta_flower0` is randomized uniformly within $[-180\degree, 180\degree)$ or $[-\pi, \pi)$.
- `FLOWER_DISTANCE` is within the settings:
    ```python
    FLOWER_DISTANCE_SETTINGS={'close': 0.8, # meter 
                              'mid':   1.6, # meter
                              'far':   2.4, # meter
    ```
- Run `close` and `far` settigns first, `mid` is optional. Each setting run for 1000 trials.

Endings:
- Flower Collision Distance = 0.22 meter
- Flower Dockzone: 60&deg;,  -30&deg; &rarr; 30&deg;.
- Out-Zone: Bat xy should be within -1, 4.

***
## [Exp 2] 1 bat 1 flower, fixed init configs
We found that:

- Facing the front of the flower (+/- 20 degree from the front), docking is highly likely.
- Facing the back of the flower has higher hit likelihood. → Let’s ignore this.
- Facing the side of the flower between 45 degree and 65 degree from the flower opening. → higher hit likely hood.

We will construct 5 configs:

- A1: theta_bat = 0°, theta_flower = -180°
- A2: theta_bat = 45°, theta_flower = -180°
- B1: theta_bat = 0°, theta_flower = 135°
- B2: theta_bat = 45°, theta_flower = 135°
- B3: theta_bat = -45°, theta_flower = 135°

Maybe: 200 episodes each config?

What to collect? Enough to accurately reconstruct the each episodes?

***Data to collect:***

- For every single steps:
    1. Bat poses ***→ Historical trajectories of bat***
    2. Echoes (Either raw or compress, that’s fine —> Let’s do compress) Cheaper mem cost.
    → No need for model input??
    ***→ Historical echoes progression***
    3. Estimate flower pose (in world coordinates)
    → Consider the case of nan: either keep use previous estimate if previous course is used or use nan if walking randomly
    ***→ Historical record of estimate flower pose***
    4. Keep track of object presence detection ***→ Keep track of object presence. Also keep track of random walk (of flower est is nan and presence is False)***
    5. Keep track of which step re-planning is performed ***→ Keep track of where estimate is used.***
    6. Translational and rotational errors ***→ For convenience. Also see how this number change through the episodes. Also, does the errors reduced as it I come closer.
    Matching with the re-planning steps, did I replan in a better place?***
    7. Misc tracking:
        1. Linear velocity, angular velocity
        2. Use prediction (maybe not needed for 1 flower case but will be use full for 2 flower case)
- For episode:
    1. Init bat pose, init flower pose
    2. angle of arrival, ending azimuth of flower
    3. outcome
    4. path length
    5. optimal path length