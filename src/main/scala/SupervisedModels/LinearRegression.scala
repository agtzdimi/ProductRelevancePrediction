package SupervisedModels

import Preprocessing.LabelPoints
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, SparkSession}

class LinearRegression extends Serializable {

   def runPrediction(trainingDfData: DataFrame, sc: SparkContext, ss: SparkSession): Unit = {
      val labelpoint = new LabelPoints()
      labelpoint.createLabelPoints(trainingDfData, ss)
      val data = MLUtils.loadLibSVMFile(sc, "src/main/resources/trainLabeledVectors.csv")

      /** Creation of ML Model, as Labeled file, we can create it, save it to memory, for future uses */

      val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
      val (trainingData, testData) = (splits(0), splits(1))

      trainingData.cache()
      testData.cache()

      println("Applying Linear regression...")

      val model = LinearRegressionWithSGD.train(trainingData, 100, 1.0)

      /** Evaluation of Linear Regression on test instances and compute test error */
      val labelsAndPredictions = testData.map { point =>
         val prediction = model.predict(point.features)
         (point.label, prediction)
      }
      val testMSE = labelsAndPredictions.map { case (v, p) => math.pow(v - p, 2) }.mean()
      println("Test Mean Squared Error = " + testMSE)
      sc.stop()

   }


}