# Voice Verification Using Resemblyzer

This project contains a Jupyter Notebook (`voice_verification.ipynb`) that demonstrates how to perform **Speaker Verification** by leveraging a pre-trained deep learning model called [Resemblyzer](https://github.com/resemble-ai/Resemblyzer).

The notebook compares a set of known speaker voice recordings (training data) with alternative recordings of the same speakers (test data). By utilizing a highly optimized deep learning architecture, it calculates and visualizes the similarity scores, confirming how well a voice can be matched against a known profile.

## Machine Learning Concepts & Mathematics

The process of voice verification in this notebook relies heavily on the concepts of **speech embeddings** and **cosine similarity**.

### 1. Voice Embeddings (D-Vectors)

Machine learning models, and standard Artificial Neural Networks, cannot directly process raw audio waves effortlessly. The audio must first be converted into a meaningful numerical representation.

Resemblyzer employs a variant of the **Generalized End-to-End (GE2E)** loss-based model. It processes audio as follows:

- **Preprocessing:** Audio samples are read, the volume is normalized, and long silences are truncated. The signal is converted into **mel-spectrograms**, a visual representation of the spectrum of frequencies of a signal as it varies with time (mimicking human ear perception).
- **Recurrent Neural Network (LSTM/GRU):** These spectrogram frames are passed into an LSTM (Long Short-Term Memory) network which excels at capturing sequential data.
- **Embeddings Calculation:** The output of this network over the timeframes is collapsed (usually averaged) into a fixed-length numeric vector representing the *characteristics of the voice itself*, ignoring the actual words said. This vector is called a **d-vector** or **speaker embedding**. In this notebook, `VoiceEncoder.embed_speaker()` handles this logic.

### 2. Inner Product and Cosine Similarity

To figure out if "Speaker A" is the same person as "Unknown Speaker B", we need a mathematical way to compare their voice embeddings.

If we have two embedding vectors: $\mathbf{a}$ from the training set and $\mathbf{b}$ from the test set, we calculate the similarity. Because Resemblyzer normalizes these embedded vectors so that their magnitude (length) is exactly 1 ($||\mathbf{a}|| = 1$ and $||\mathbf{b}|| = 1$), taking their **dot product** (or inner product) is strictly equivalent to calculating the **Cosine Similarity**.

$$
\text{Similarity}(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = ||\mathbf{a}|| ||\mathbf{b}|| \cos(\theta) = \cos(\theta)
$$

- If the similarity is close to **$1$**, the vectors are pointed in the same direction, meaning the voices are very similar.
- If the similarity is closer to **$0$** (or negative), they are pointing in different directions, meaning the speakers are likely different.

In the notebook, the cross-similarity matrix between train and test distributions is calculated using:

```python
spk_sim_matrix = np.inner(spk_embeds_a, spk_embeds_b)
```

### 3. Evaluation & Accuracy

The resulting `spk_sim_matrix` provides a pairwise comparison of all training identities against all testing identities. The matrix diagonal contains the similarities matching "Speaker X" from the train set to "Speaker X" in the test set.

The code analyzes this by comparing the true identity similarity predictions (`spk_sim_matrix.diagonal()`) against the highest similarity recorded along each row (`np.amax()`). A correctly verified voice means that the known recording of Speaker X was most similar to the test recording of Speaker X over *all* other possible speakers in the set.

## Requirements

To execute this notebook properly, you'll need the following dependencies:

- `resemblyzer` (pulled directly from their [GitHub repository](https://github.com/resemble-ai/Resemblyzer.git))
- `numpy`
- `matplotlib`
- `tqdm`

You can start the setup process directly in the notebook by running the first cell, which clones the necessary utilities.
