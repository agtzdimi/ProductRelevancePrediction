package SupervisedModels

import java.nio.file.{Files, Paths}

import Preprocessing.LabelPoints
import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, SparkSession}

class DecisionTree extends Serializable {

   def runPrediction(trainingDfData: DataFrame, sc: SparkContext, ss: SparkSession): Unit = {
      val labelpoint = new LabelPoints()
      labelpoint.createLabelPoints(trainingDfData, ss)
      val data = MLUtils.loadLibSVMFile(sc, "src/main/resources/trainLabeledVectors.csv")

      /** Creation of ML Model, as Labeled file, we can create it, save it to memory, for future uses */

      val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
      val (trainingData, testData) = (splits(0), splits(1))
      trainingData.cache()
      testData.cache()

      var modelExist = Files.exists(Paths.get("target/tmp/myDecisionTreeRegressionModel"))

      if (!modelExist) {

         println("Applying Desicion tree regression...")

         val categoricalFeaturesInfo = Map[Int, Int]()
         val impurity = "variance"
         val maxDepth = 5
         val maxBins = 32
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
