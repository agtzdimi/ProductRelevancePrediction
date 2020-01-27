package SupervisedModels

import java.nio.file.{Files, Paths}

import Preprocessing.DataframeManager
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
//import org.apache.spark.ml.feature.LabeledPoint
import Preprocessing.LabelPoints
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.sql.{DataFrame, SparkSession}

class RandomForest extends Serializable {

  def runPrediction(trainingDfData: DataFrame, sc: SparkContext, ss: SparkSession): Unit = {
    val labelpoint = new LabelPoints()
    labelpoint.createLabelPoints(trainingDfData, ss)

    val dataframeManager = new DataframeManager()
    val filteredData: RDD[LabeledPoint] = dataframeManager.getTopFeatures(sc, 500)

    val splits = filteredData.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val (trainingData, testData) = (splits(0), splits(1))
    trainingData.cache()
    testData.cache()

    var modelExist = Files.exists(Paths.get("target/tmp/myRandomForestRegressionModel"))

    println("")
    println("Applying Random Forest regression...")

    if (!modelExist) {

      val numClasses = 2
      val categoricalFeaturesInfo = Map[Int, Int]()
      val numTrees = 50
      val featureSubsetStrategy = "auto" // Let the algorithm choose.
      val impurity = "variance"
      val maxDepth = 20
      val maxBins = 100

      val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
        numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

      model.save(sc, "target/tmp/myRandomForestRegressionModel")

    }

    val model = RandomForestModel.load(sc, "target/tmp/myRandomForestRegressionModel")

    /** Evaluation of RandomForest model on test instances and compute test error */
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map { case (v, p) => math.pow((v - p), 2) }.mean()
    println("Test Mean Squared Error = " + testMSE)

  }

}
