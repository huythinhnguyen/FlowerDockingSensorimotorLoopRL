### To-Do:
- Build `py_environment` for the simpliest docking task.
- Build `ModelTrainer.py`, these features should be included.
    + Saving checkpoint.
    + Train model from checkpoint.
    + Inference and demonstrate episodes from checkpoint. (run a episode, and visualize with matplotlib)

### Implement Epsilon Decay:
Use `tf.keras.optimizers.schedules.LearningRateSchedule` to implement schedule learning rate.
https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/LearningRateSchedule

```python
# use this format
starting_episilon=0.8
final_epislon=0.001
number_of_exploration_steps 
episilon_schedule = MyCustomEpsilonSchedule(starting_epislon, final_epislon, number_of_exploration_steps)

agent = DqnAgent(
    ...
    epsilon_greedy = epsilon_schedule,
    ...
)
```