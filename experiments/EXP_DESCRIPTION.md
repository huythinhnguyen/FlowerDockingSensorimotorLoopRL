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
