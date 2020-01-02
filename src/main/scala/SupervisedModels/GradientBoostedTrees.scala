package SupervisedModels

import java.nio.file.{Files, Paths}

import Preprocessing.{DataframeManager, LabelPoints}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

class GradientBoostedTrees {

  def runPrediction(trainingDfData: DataFrame, sc: SparkContext, ss: SparkSession): Unit = {
    val labelpoint = new LabelPoints()
    labelpoint.createLabelPoints(trainingDfData, ss)

    val dataframeManager = new DataframeManager()
    val filteredData: RDD[LabeledPoint] = dataframeManager.getTopFeatures(sc, 500)

    val splits = filteredData.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val (trainingData, testData) = (splits(0), splits(1))
    trainingData.cache()
    testData.cache()

    var modelExist = Files.exists(Paths.get("target/tmp/myGradientBoostedTrees"))
    println("")
    println("Applying Gradient Boosted Trees regression...")

    if (!modelExist) {

      val boostingStrategy = BoostingStrategy.defaultParams("Regression")
      boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
      boostingStrategy.treeStrategy.maxDepth = 5
      // Empty categoricalFeaturesInfo indicates all features are continuous.
      boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

      val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

      model.save(sc, "target/tmp/myGradientBoostedTrees")

    }

    val model = GradientBoostedTreesModel.load(sc, "target/tmp/myGradientBoostedTrees")

    /** Evaluation of GradientBoostedTrees model on test instances and compute test error */
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map { case (v, p) => math.pow((v - p), 2) }.mean()
    println("Test Mean Squared Error = " + testMSE)

  }

}
