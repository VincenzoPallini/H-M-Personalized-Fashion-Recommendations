**Project Title:** H&M Recommendation System with TensorFlow Recommenders 

**General Description**

This personal project implements a fashion product recommendation system based on the "H&M Personalized Fashion Recommendations" dataset available on Kaggle. The goal is to generate personalized item suggestions for customers using a *Retrieval* model built with the TensorFlow Recommenders (TFRS) library.

The system focuses on the *Retrieval* stage, which aims to select an initial subset of relevant candidates from the entire item catalog for a given user. This approach is efficient for handling large datasets like H&M's.

**Methodology and Implementation**

1.  **Model:** A *factorized retrieval* model was used. This model consists of two sub-models (or "towers"):
    * **Query Model:** Processes user features (in this case, `customer_id`) to generate a user embedding (numerical representation).
    * **Candidate Model:** Processes item features (in this case, `article_id`) to generate an item embedding.
2.  **Embeddings:** Both models use `StringLookup` layers to map categorical IDs (customer and article) to integer indices, followed by `Embedding` layers to learn dense vector representations (embeddings) of dimension 64 for users and items.
3.  **Task and Loss:** The `tfrs.tasks.Retrieval` task was employed, which computes the loss and appropriate metrics for retrieval models, such as `FactorizedTopK`, to evaluate if relevant items are present among the top K suggestions.
4.  **Training:** The model was trained using the Adagrad optimizer on a subset of historical transaction data (`transactions_train.csv` filtered by date). The data was processed using `tf.data.Dataset` for efficient data pipeline management.
5.  **Inference/Recommendation:** After training, Google's `ScaNN` (Scalable Nearest Neighbors) library, integrated into TFRS, was used to index the item embeddings and efficiently retrieve the top-K (in this case, K=12) candidate items closest (most relevant) to a given user's embedding.

**Technologies Used**

* **Language:** Python
* **Core Libraries:**
    * **TensorFlow:** Base framework for machine learning.
    * **TensorFlow Recommenders (TFRS):** Specific library for building recommendation systems in TensorFlow.
    * **ScaNN:** Library for efficient large-scale nearest neighbor search, used for inference.
    * **Pandas:** For data manipulation and analysis (CSV reading, data filtering).
    * **NumPy:** For numerical operations.
    * **Matplotlib & OpenCV (cv2):** For results visualization (displaying recommended item images).
    * **CUDA (implied):** The setup suggests GPU usage for accelerated training and inference, common in environments like Kaggle.

**Results Obtained**

* Successfully trained a retrieval model capable of learning meaningful representations (embeddings) for H&M customers and items based on their past interactions.
* The system can generate a list of the top 12 recommended items for a given `customer_id`, leveraging ScaNN's efficiency for search.
* Demonstrated the ability to visualize recommended items by fetching their corresponding images.
* The notebook includes examples of recommendations generated for specific users and also demonstrates finding similar items based on embeddings ("When customers buy this, they also buy...").
* Extracted and displayed the specific embeddings learned for certain items.
