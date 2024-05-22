# Machine Learning Library for Edge AI support in Lingua Franca

## Popular Edge AI Algorithms

Although the deployment of AI models on edge devices has become increasingly popular, enabling local computation rather than relying solely on remote cloud servers, it presents several challenges. Edge devices often have limited computing resources, so the complexity and size of models can significantly impact their performance. Balancing the algorithm's performance with the available resources is crucial to ensure that the edge device can execute the AI model effectively.

Therefore, it is important to select appropriate AI algorithms that can be successfully utilized on edge devices. The most popular and suitable algorithms for deployment on edge devices currently include (i) **classification**, (ii) **detection**, (iii) **segmentation**, and (iv) **tracking** algorithms. These four types of algorithms offer practical solutions for various applications, ranging from object recognition and tracking to quality control and predictive maintenance.

However, recent advancements in edge device hardware and model reduction techniques have demonstrated the feasibility of deploying **clustering** and **natural language processing** (NLP) algorithms as well. Clustering algorithms enable edge devices to group data points into clusters based on their similarities, while NLP algorithms allow edge devices to understand and respond to natural language commands.

## Library Implementation

The goal is to develop several LF programs that incorporates multiple reactors, each implementing a specific AI algorithm optimized for edge devices. All AI algorithms will utilize TensorFlow Lite, a set of tools designed to enable on-device machine learning by allowing developers to run their models on mobile, embedded, and edge devices. The AI algorithms to be included in the library are as follows:

- **ComputerVision.lf**
    - [ ] ImageClassifier
    - [ ] ImageSegmenter
    - [ ] ObjectDetector
    - [ ] ImageSearcher (?)
    - [ ] ImageEmbedder (?)
- **NLP.lf**
    - [ ] NLClassifier
    - [ ] BertNLClassifier
    - [ ] BertQuestionAnswer
    - [ ] TextSearcher
    - [ ] TextEmbedder
- **Audio.lf**
    - [x] AudioClassifier
