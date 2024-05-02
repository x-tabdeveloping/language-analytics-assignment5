# language-analytics-assignment5
Fifth assignment for Language Analytics in Cultural Data Science

This assignment is oriented at figuring out how much CO2 was emitted as a result of the code used in the exam portfolio.
All measurements were conducted with the [CodeCarbon Python package](https://mlco2.github.io/codecarbon/index.html) for substasks in each assignment.
All code for producing my results can be found in the other assignments' repositories.

## Usage

All emission files are available in the `emissions/` folder in this repository.

To reproduce the figures used in this README, you will have to install requirements:
```bash
pip install -r requirements.txt
```

Then run the visualization script:
```bash
python3 src/produce_visualizations.py
```

## Analysis

### System Parameters
All assignments were run on my HP EliteBook 840 G8 work computer with the following parameters:

| Parameter | Value |
| - | - |
| CPU | 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz (8 cores) |
| OS | Ubuntu 20.04 LTS (Linux-5.15.0 kernel) |
| Available RAM | 15.305 GB |

### Total Emissions

Doing one run of all assignments resulted in a total estimated emission of approximately 0.001 kilograms of CO2.
To put this in context an average human produces about 1kg of CO2 simply [through metabolism](https://www.sciencefocus.com/planet-earth/how-much-does-human-breathing-contribute-to-climate-change).
A single run of all experiments amounts to about 0.1% of the emissions of a human being breathing normally.
I would thus deem these experiments rather sustainable.

> Note that this only includes a single run for each experiment.
> These are overly optimistic estimates in the context of real development.
> The computer's emissions throughout development time, numerous test runs and the previous runs would probably amount to multiple times this value.

### Emissions per task

To gain insight into which kinds of tasks are most emission-intensive we can look at the summed emissions of subtasks in each assignment.

| Total Emissions | Mean Emission Rate |
| - | - |
| ![Bar Emissions]("./figures/projects_bar.png") | ![Bar Emission Intensity]("./figures/projects_intensity_bar.png") |

#### Most emissions: Assignment 4
We can see that by far the most emissions were produced by the fourth assignment, where emotion labels needed to be extracted based on passages from Game of Thrones, amounting to twice as much CO2 as the second heaviest task.
This is likely due to the fact that inference with a transformer model had to be run.
Looking at the subtask emissions suggests that all other parts of the assignment were insignificant in comparison to inference.
The corpus was also relatively large.
I doubt that this process could be made more efficient, as a relatively small model was used, and I used batch processing.
Perhaps implementation of the model in another framework could have made it more efficient, especially on CPU (e.g. in JAX).

| Assignment4 | Assignment2 |
| - | - |
| ![Emotion Arcs]("./figures/emotion_arcs_pie.png") | ![Fake News]("./figures/fake_news_classification_pie.png") |

#### Second place: Assignment 2

The second most poluting task was assignment 2 where fake news had to be classified.
As we can see from the subtask emissions, training the neural network classifier is responsible for almost 99% of the emissions, while only marginally improving on the Logistic Regression's performance.
Inference on the other hand was approximately equally efficient with LR and NN classifiers.

This inefficiency is most likely due to a number of factors:
 - Vast parameter space. Even a single hidden layer with a hundred nodes results in an incredibly large amount of parameters to be trained, when BoW representations are used as model input.
 Larger parameter spaces are simply harder to optimize ([curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)).
 It probably does not help, that bag-of-words representations are in discrete space, not a continuous one.
 - Scikit-learn's neural network implementations are not written in a tensor framework, but rely on numpy and scipy instead.
 This does not allow for optimizing computation graphs or [JIT compilation](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#using-jit-to-speed-up-functions).

Possible remedies:
 1. Reducing the dimensionality of bag-of-words representations with matrix factorization (LSA, NMF) to around a 100 dimensions would likely result in just as good or higher performance, as:
    - The number of parameters would be significantly lower.
    - Models would not have to deal with highly collinear word occurrences.
    - Input values would be in continuous space.
 2. Reimplementing the neural network in FLAX, Torch or TensorFlow would likely increase efficiency to a great extent due to better optimized computational graphs.

#### Third place: Assignment 1

Assignment 1 was significantly less poluting than the two other assignments so far, cutting the emissions of assignment 2 almost in half.
Subtask data was more coarse on this task than other ones, as processing happenned while streaming data from disk as opposed to loading data first and then processing it.
This was done out of considerations for efficiency.

Most of the emissions were due to processing documents with SpaCy, which is, yet again, utilizing neural networks to do most tasks.

A potential step to increase efficiency would be to remove not-needed components from the SpaCy pipeline.
The lemmatizer and parser components could have been turned off to reduce CO2 emissions, as they were not directly utilized in the assignment.

| Assignment1 | Assignment3 |
| - | - |
| ![SpaCy processing]("figures/pos_ner_spacy_pie.png") | ![Query Expansion]("figures/query_expansion_pie.png") |

#### Lowest emissions: Assignment 3

Assignment 3 was by far the least poluting.
This is likely due to the fact that no inference or model training was conducted using neural networks, and the bulk of the emissions were produced by loading the model and data.
Static word embedding models are incredibly efficient as producing an embedding for a word amounts to a dictionary lookup and then indexing an array (both are constant time complexity operations).

Loading the model was likely the heaviest subtask, as static word embedding models are relatively large due to the number of tokens.
If necessary, this could potentially be made more efficient with [Bloom embeddings](https://explosion.ai/blog/bloom-embeddings), but the expected gains are very modest.
Storing and loading the model from disk instead of downloading it from Gensim's repositories would also improve efficiency if multiple runs are done.

### Conclusion

An overarching pattern we can see with the tasks involved in my portfolio is that training and running artificial neural networks results in significantly higher emissions than most other subtasks, sometimes only marginally improving performance.
This, I believe, is a more widespread problem in the machine learning community, and more effort should be directed at finding alternatives that are "good enough" for the task at hand, but are more computationally efficient.
The popularity of deep learning also likely results in their usage even in scenarios where they are known to be a suboptimal solution (e.g. [tabular classification](https://arxiv.org/abs/2207.08815)).
Careful considerations in modeling choices, efficient implementations and research are needed to reduce machine learning CO2 emissions worldwide.
