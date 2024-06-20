# Machine Learning Library for Edge AI support in Lingua Franca

## Popular Edge AI Algorithms

Although the deployment of AI models on edge devices has become increasingly popular, enabling local computation rather than relying solely on remote cloud servers, it presents several challenges. Edge devices often have limited computing resources, so the complexity and size of models can significantly impact their performance. Balancing the algorithm's performance with the available resources is crucial to ensure that the edge device can execute the AI model effectively.

Therefore, it is important to select appropriate AI algorithms that can be successfully utilized on edge devices. The most popular and suitable algorithms for deployment on edge devices currently include (i) **classification**, (ii) **detection**, (iii) **segmentation**, and (iv) **tracking** algorithms. These four types of algorithms offer practical solutions for various applications, ranging from object recognition and tracking to quality control and predictive maintenance.

However, recent advancements in edge device hardware and model reduction techniques have demonstrated the feasibility of deploying **clustering** and **natural language processing** (NLP) algorithms as well. Clustering algorithms enable edge devices to group data points into clusters based on their similarities, while NLP algorithms allow edge devices to understand and respond to natural language commands.

## Library Implementation

The objective is to develop several LF programs incorporating multiple reactors, each implementing a specific AI algorithm optimized for edge devices. These AI algorithms will utilize [TensorFlow Lite](https://www.tensorflow.org/lite/guide), a suite of tools designed for on-device machine learning, enabling developers to run models on mobile, embedded, and edge devices. Specifically, each algorithm will leverage the [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview), which consists of powerful and user-friendly task-specific libraries that facilitate the creation of ML experiences with TFLite. This library offers optimized out-of-the-box model interfaces for popular machine learning tasks, such as image classification and question answering. These model interfaces are designed to maximize performance and usability for each specific task.

The AI algorithms to be included in the library are as follows:

- **[`Audio.lf`](lib/Audio.lf)**
    - [x] `AudioClassifier`
- **[`ComputerVision.lf`](lib/ComputerVision.lf)**
    - [x] `ImageClassifier`
    - [x] `ImageSegmenter`
    - [x] `ObjectDetector`
    <!-- - [ ] `ImageSearcher` (?)
    - [ ] `ImageEmbedder` (?) -->
- **[`NLP.lf`](lib/NLP.lf)**
    - [x] `NLClassifier`
    - [x] `BertQuestionAnswer`
    <!-- - [ ] `TextSearcher`
    - [ ] `TextEmbedder` -->

>  **Note**: The `NLClassifier` supports both **BERT-based** and **Average Word Embedding** (AWE) model architectures.

For each specific task library, a machine learning model is provided in the [`models/`](models/) folder. However, you can train and use your own model with the single reactor. Just be sure to carefully read the [documentation](https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview) for the specific library task API you intend to use to verify model compatibility. Regardless of the model you use, it is important to specify the model's absolute path when instantiating a reactor library in your main reactor. For example:
```Python
cls = new AudioClassifier(model="/absolute/path/to/model.tflite");
```

### Example Usage

In the folder [`src/`](src/), you can find several example LF programs that demonstrate the usage of the library's reactors for each specific task.

- **`Audio Classification`**
    - [`Real-Time Audio Classification`](src/RTAudioClassification.lf)
- **`Computer Vision`**
    - [`Image Classification`](src/SimpleImageClassification.lf)
    - [`Image Segmentation`](src/SimpleImageSegmentation.lf)
    - [`Real-Time Image Segmentation`](src/RTImageSegmentation.lf)
    - [`Object Detection`](src/SimpleObjectDetection.lf)
    - [`Real-Time Object Detection`](src/RTObjectDetection.lf)
- **`Natural Language Processing`**
    - [`Text Classification`](src/TextClassification.lf)
    - [`Real-Time Speech-to-Text Sentiment Analysis`](src/SentimentAnalysisSpeech.lf)
    - [`Question & Answer`](src/BertQA.lf)
