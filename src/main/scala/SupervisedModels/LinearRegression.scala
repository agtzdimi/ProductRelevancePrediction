package SupervisedModels

import Preprocessing.{DataframeManager, LabelPoints}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

class LinearRegression extends Serializable {

  def runPrediction(trainingDfData: DataFrame, sc: SparkContext, ss: SparkSession): Unit = {
    val labelpoint = new LabelPoints()
    labelpoint.createLabelPoints(trainingDfData, ss)

    val dataframeManager = new DataframeManager()
    val filteredData: RDD[LabeledPoint] = dataframeManager.getTopFeatures(sc, 500)

    val splits = filteredData.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val (trainingData, testData) = (splits(0), splits(1))

    trainingData.cache()
    testData.cache()
    println("")
    println("Applying Linear regression...")

    val model = LinearRegressionWithSGD.train(trainingData, 100, 0.3)

    /** Evaluation of Linear Regression on test instances and compute test error */
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map { case (v, p) => math.pow(v - p, 2) }.mean()
    println("Test Mean Squared Error = " + testMSE)

  }


}