import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances; 
import weka.core.converters.ConverterUtils.DataSource;

public class WekaTest {

  public static void main(String[] args) {
    try {
      // import data
      DataSource loader = new DataSource("taste_emotions.arff");
      Instances datasetInstances = loader.getDataSet();
      
      // data pre-processing, training/testing data split 80-20
      datasetInstances.randomize(new java.util.Random(0));
      int trainingDataSize = (int) Math.round(datasetInstances.numInstances() * 0.80);
      int testDataSize = (int) datasetInstances.numInstances() - trainingDataSize;
      Instances trainingInstances = new Instances(datasetInstances, 0, trainingDataSize);
      Instances testInstances = new Instances(datasetInstances, trainingDataSize, testDataSize);

      // Set target class
      trainingInstances.setClassIndex(trainingInstances.numAttributes() - 1);
      testInstances.setClassIndex(testInstances.numAttributes() - 1);
      // build
      RandomForest model = new RandomForest();
      model.buildClassifier(trainingInstances);
      model.setNumIterations(100);

      System.out.println(model);

      // Evaluation
      Evaluation evaluation = new Evaluation(trainingInstances);
      evaluation.evaluateModel(model, testInstances);
      System.out.println(evaluation.toSummaryString());
      System.out.println(evaluation.toMatrixString());

    } catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }
}

