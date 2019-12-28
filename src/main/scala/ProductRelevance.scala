import MlModels.{DecisionTree, LinearRegression, RandomForest}
import Preprocessing.DataframeManager
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}


object TextProcessingNew {


  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local[*]")
      .setAppName("Text Mining Project").setSparkHome("src/main/resources")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    conf.set("spark.executor.instances", "1")
    conf.set("spark.executor.cores", "8")
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val sc = new SparkContext(conf)
    val ss = SparkSession.builder().master("local[*]").appName("Text Mining Project").getOrCreate()
    import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

    val dataframeManager = new DataframeManager()

    /** We create separate Dataframes for the different use cases
      * To split the code and make it easier to use we created a Dataframe Manager to handle
      * different actions. All dataframes will remove the stopwords clarified in a csv file
      * in resources directory
      */
    val attributesDF = dataframeManager.getAttrDF("src/main/resources/attributes_sample.csv", ss, sc)
    val trainingDF = dataframeManager.getTrainDF("src/main/resources/train_sample.csv", ss, sc)
    val descriptionDF = dataframeManager.getProdDescrDF("src/main/resources/product_descriptions_sample.csv", ss, sc)
    val searchDF = dataframeManager.getSearchTermDF("src/main/resources/train_sample.csv", ss, sc)

    val joinTrainTestDF = dataframeManager.joinDataframes(trainingDF, attributesDF, ss, sc)


    /** Create a dataframe containing all the dataframes joined to access every piece of information
      */
    val joinDescTestTrainDF = dataframeManager.joinDataframes(joinTrainTestDF, descriptionDF, ss, sc)
    joinTrainTestDF.take(5).foreach(println)

    /** In order to calculate the results of the ML Models
      * we need to prepare the data for fit & predict.
      * This means that we need to have the data in a LabelPoints format
      * applying labels and features
      */

    val idf = dataframeManager.getIDF(joinDescTestTrainDF, ss, sc)
    val trainingDfDataInit = idf.select($"rProductUID", $"rFilteredWords", $"rFeatures", $"rRelevance", $"rrLabel").withColumnRenamed("rFeatures", "features").withColumnRenamed("rrLabel", "label")
    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Decision Tree ====================*/
    /* ======================================================= */

    println("Defining features and label for the model...")
    var trainingDfData = trainingDfDataInit.select("label", "features")
    val decisionTree_Regressor = new DecisionTree()
    decisionTree_Regressor.runPrediction(trainingDfData, sc, ss)

    //
    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Random Forest ================== */
    /* ======================================================= */
    println("Defining features and label for the model...")
    trainingDfData = trainingDfDataInit.select("features", "label")
    val random_forest_Regressor = new RandomForest()
    random_forest_Regressor.runPrediction(trainingDfData, sc, ss)

    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Linear Regression ================== */
    /* ======================================================= */
    println("Defining features and label for the model...")
    trainingDfData = trainingDfDataInit.select("features", "label")
    val linearRegression = new LinearRegression()
    linearRegression.runPrediction(trainingDfData, sc, ss)

    sc.stop()
  }


}