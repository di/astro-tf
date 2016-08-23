# astro

Detecting Asteroids with Neural Networks, as presented at PromptWorks on August
23rd, 2016

Extended presentation available at: <>

# Setup

Install TensorFlow: https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html

Then:

```
$ pip install numpy pandas pillow
```

# Training and Running

Training and trial data are in the `/training_data` and `/test_data/trial*`
directories.

To train and evaluate the network, run:

```
$ python astro.py
```

This will output the results of the evaluations of each of the three trials.
