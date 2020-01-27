package SupervisedModels

import java.nio.file.{Files, Paths}

import Preprocessing.{DataframeManager, LabelPoints}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

class DecisionTree extends Serializable {

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
    println("Applying Desicion tree regression...")
    var modelExist = Files.exists(Paths.get("target/tmp/myDecisionTreeRegressionModel"))

    if (!modelExist) {
      val categoricalFeaturesInfo = Map[Int, Int]()
      val impurity = "variance"
      val maxDepth = 20
      val maxBins = 100
      val model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
      model.save(sc, "target/tmp/myDecisionTreeRegressionModel")

    }

    val model = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeRegressionModel")

    /** Evaluation of DecisionTree model on test instances and compute test error */
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map { case (v, p) => math.pow(v - p, 2) }.mean()
    println("Test Mean Squared Error = " + testMSE)

  }

}
