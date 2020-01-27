package SupervisedModels

import Preprocessing._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

class NaiveBayes {

  def runPrediction(trainingDfData: DataFrame, sc: SparkContext, ss: SparkSession): Unit = {
    val labelpoint = new LabelPoints()
    labelpoint.createLabelPoints(trainingDfData, ss)

    val dataframeManager = new DataframeManager()
    val filteredData: RDD[LabeledPoint] = dataframeManager.getTopFeatures(sc, 500)

    /** Creation of ML Model, as Labeled file, we can create it, save it to memory, for future uses */

    val splits = filteredData.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val (trainingData, testData) = (splits(0), splits(1))
    println("")
    println("Applying Naive Bayes regression...")

    trainingData.cache()
    testData.cache()

    val model = NaiveBayes.train(trainingData, lambda = 0.1, modelType = "multinomial")

    /** Evaluation of Linear Regression on test instances and compute test error */
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map { case (v, p) => math.pow(v - p, 2) }.mean()
    println("Test Mean Squared Error = " + testMSE)

  }

}
