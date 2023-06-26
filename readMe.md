1. We use Javaparser to construct our datasets and Liu et al.'s datasets.
    1) We use 'CG-LSMN/Code/src/main/java/DataProcess_Class_Class.java' to generate our Dataset 1.
    2) We use 'CG-LSMN/Code/src/main/java/DataProcess_Method_Class.java' to generate our Dataset 2.
    3) We use "CG-LSMN/Code/src/main/java/StatinceCalculator.java" to assist in generating datasets in the data format defined by Liu et al.

2. The source code for our approach and state-of-the-art baselines is located under the "CG-LSMN/Code" path.
    1) We use "Code/src/test/code2vec master" and "Code/src/test/SDNE-master" to assist in generating datasets in the data format defined by Cui et al. and Kurbatova et al.
    2) Execute 'python Code/src/test/FeatureEnvyDetectionModels/ours-train.py' to run our CG-LSMN.
    3) Execute 'python Code/src/test/FeatureEnvyDetectionModels/Liu-CNN-Sense.py' to run the model of Liu et al.
    4) Execute 'python Code/src/test/FeatureEnvyDetectionModels/Wang-BiLSTM-Attention.py' to run the model of Wang et al.
    5) Execute 'python Code/src/test/FeatureEnvyDetectionModels/Zhang-PE-Self-Att-LSTM.py' to run the model of Zhang et al.
    6) Execute 'python Code/src/test/FeatureEnvyDetectionModels/Yin-local-global.py' to run the model of Yin et al.
    7) Execute 'python Code/src/test/FeatureEnvyDetectionModels/Cui-RMove.py' to run the model of Cui et al.
    8) Execute 'python Code/src/test/FeatureEnvyDetectionModels/Kurbatova-path-based.py' to run the model of Kurbatova et al.

3. Under the "CG-LSMN/Data" path, we collected feature envy refactoring cases on GitHub.
    1) View the details of the collected feature Envy cases from the file "Data/readMe.md".

4. The file 'result.xlsx' contains all experimental results for cross validation of all experiments. The file 'env36 test. yaml' is a detailed dependency of our experimental environment. Execute 'conda env create - f env36 test.yaml' to create a running environment. Note that before running, check all file paths to ensure that the absolute paths in the code are modified to your correct path.