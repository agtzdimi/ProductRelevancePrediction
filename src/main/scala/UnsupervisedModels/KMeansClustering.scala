package UnsupervisedModels

import org.apache.spark.SparkContext
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions.{col, max, not}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

class KMeansClustering {

  def runPrediction(trainingData: DataFrame): DataFrame = {

    val kmeans = new KMeans().setK(2).setFeaturesCol("Similarity_as_vector").setPredictionCol("prediction")
    val model = kmeans.fit(trainingData)
    val prediction_df = model.transform(trainingData)

    val final_prediction_df = prediction_df.withColumn("invert_prediction", not(col("prediction").cast("Boolean")).cast("Double"))
      .withColumn("prediction", col("prediction").cast("Double"))
      .withColumn("binarized_relevance", col("binarized_relevance").cast("Double"))

    return final_prediction_df

  }

  def getMetrics(prediction_df: DataFrame, metric: String, sparkSession: SparkSession, sc: SparkContext): Unit = {
    import sparkSession.implicits._ // For implicit conversions like converting RDDs to DataFrames

    val predictionAndLabels = prediction_df.select("prediction", "binarized_relevance").rdd.map { case Row(p: Double, l: Double) => (p, l) }

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val f1_score = metrics.fMeasureByThreshold.toDF("threshold", "f1-score")
    val max_f1_score = f1_score.select(max("f1-score")).head().getDouble(0)

    val recall = metrics.recallByThreshold.toDF("threshold", "recall")
    val max_recall_score = recall.select(max("recall")).head().getDouble(0)

    val precision = metrics.precisionByThreshold.toDF("threshold", "precision")
    val max_precision_score = precision.select(max("precision")).head().getDouble(0)

    metric match {
      case "precision" => println("The precision is: ", max_precision_score)
      case "recall" => println("The recall is: ", max_recall_score)
      case "f1-score" => println("The f1-score is: ", max_f1_score)
      case _ => println("Invalid metric: \"precision | recall | f1-score\"")
    }


  }
}
