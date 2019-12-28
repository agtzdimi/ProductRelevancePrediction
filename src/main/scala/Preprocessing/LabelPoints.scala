package Preprocessing

import java.nio.file.{Files, Paths}

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, SparkSession}

class LabelPoints {

  def createLabelPoints(trainingDfData: DataFrame, ss: SparkSession): Unit = {
    import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

    var labeledPointsExist = Files.exists(Paths.get("src/main/resources/trainLabeledVectors.csv"))

    if (!labeledPointsExist) {
      // Creating Labeled data out of training Dataframe because that is the form ml algorithms accept them
      val labeled = trainingDfData.rdd.map(row => LabeledPoint(
        row.getAs[Double]("label"),
        row.getAs[org.apache.spark.ml.linalg.Vector]("features")
      )).toDS()

      // Save file for later use
      val convertedVecDF = MLUtils.convertVectorColumnsToML(labeled)
      convertedVecDF.write.format("libsvm").save("src/main/resources/trainLabeledVectors.csv")

    }

  }

}
