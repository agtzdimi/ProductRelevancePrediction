package MlModels

import java.nio.file.{Files, Paths}

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, SparkSession}
import Preprocessing.LabelPoints

class RandomForest extends Serializable {

  def runPrediction(trainingDfData: DataFrame, sc: SparkContext, ss: SparkSession): Unit = {
    val labelpoint = new LabelPoints()
    labelpoint.createLabelPoints(trainingDfData, ss)
    val data = MLUtils.loadLibSVMFile(sc, "src/main/resources/trainLabeledVectors.csv")

    /** Creation of ML Model, as Labeled file, we can create it, save it to memory, for future uses */

    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val (trainingData, testData) = (splits(0), splits(1))
    trainingData.cache()
    testData.cache()

    var modelExist = Files.exists(Paths.get("target/tmp/myRandomForestRegressionModel"))

    if (!modelExist) {

      println("Applying Random Forest regression...")

      val numClasses = 2
      val categoricalFeaturesInfo = Map[Int, Int]()
      val numTrees = 5
      val featureSubsetStrategy = "auto" // Let the algorithm choose.
      val impurity = "variance"
      val maxDepth = 5
      val maxBins = 32

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
