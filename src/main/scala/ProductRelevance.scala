import Preprocessing.DataframeManager
import SupervisedModels._
import UnsupervisedModels.KMeansClustering
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}


object ProductRelevance {


  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local[*]")
      .setAppName("Product Relevance Project").setSparkHome("src/main/resources")
    conf.set("spark.driver.memory", "14g")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    conf.set("spark.executor.instances", "1")
    conf.set("spark.executor.cores", "8")
    conf.set("spark.executor.memory", "2g")
    conf.set("spark.cores.max", "8")
    conf.set("spark.eventLog.enabled", "true")
    conf.set("spark.eventLog.dir", "src/main/resources/spark-logs")
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val sc = new SparkContext(conf)
    val ss = SparkSession.builder().master("local[*]").appName("Text Mining Project").getOrCreate()
    import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

    val train_input = "src/main/resources/train.csv"
    val attributes_input = "src/main/resources/attributes.csv"
    val descriptions_input = "src/main/resources/product_descriptions.csv"

    val dataframeManager = new DataframeManager()


    /** We create separate Dataframes for the different use cases
      * To split the code and make it easier to use we created a Dataframe Manager to handle
      * different actions. All dataframes will remove the stopwords clarified in a csv file
      * in resources directory
      */
    val attributesDF = dataframeManager.getAttrDF(attributes_input, ss, sc)
    val trainingDF = dataframeManager.getTrainDF(train_input, ss, sc)
    val descriptionDF = dataframeManager.getProdDescrDF(descriptions_input, ss, sc)
    val searchDF = dataframeManager.getSearchTermDF(train_input, ss, sc)

    val joinTrainTestDF = dataframeManager.joinDataframes(trainingDF, attributesDF, ss, sc)


    /** Create a dataframe containing all the dataframes joined to access every piece of information
      */
    val joinDescTestTrainDF = dataframeManager.joinDataframes(joinTrainTestDF, descriptionDF, ss, sc)

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

    //
    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Gradient Boosted Trees ================== */
    /* ======================================================= */
   println("Defining features and label for the model...")
   trainingDfData = trainingDfDataInit.select("features", "label")
   val gradient_boosted_trees = new GradientBoostedTrees()
   gradient_boosted_trees.runPrediction(trainingDfData, sc, ss)

    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Linear Regression ================== */
    /* ======================================================= */
   println("Defining features and label for the model...")
   trainingDfData = trainingDfDataInit.select("features", "label")
   val linearRegression = new LinearRegression()
   linearRegression.runPrediction(trainingDfData, sc, ss)

    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Naive Bayes ================== */
    /* ======================================================= */
    println("Defining features and label for the model...")
    trainingDfData = trainingDfDataInit.select("features", "label")
    val nbRegression = new NaiveBayes()
    nbRegression.runPrediction(trainingDfData, sc, ss)

    /* ================= UNSUPERVISED CASE =================== */

    /* ======================================================= */
    /* ===================== CLUSTERING ====================== */
    /* ================== KMeans Clustering ================== */
    /* ======================================================= */
    println("Defining features and label for the model...")
    val prod_df = dataframeManager.unsupervisedProdPreprocessing(train_input, ss, sc)
    val descr_df = dataframeManager.unsupervisedDescrPreprocessing(descriptions_input, ss, sc)

    // Calculate Cosine Similarity and apply KMeans

    val kmeans_features_df = dataframeManager.unsupervisedFeatureSelection(prod_df, descr_df, ss, sc)
    val kmeans_training_data_df = dataframeManager.kmeansPreprocessing(kmeans_features_df)

    val kmeans = new KMeansClustering()
    val prediction_df = kmeans.runPrediction(kmeans_training_data_df)
    kmeans.getMetrics(prediction_df, "f1-score", ss, sc, "Cosine")

    // Calculate Jacccard Similarity and apply KMeans

    val jaccardSimilarityDF = dataframeManager.calcJacccardSimilarity(searchDF,joinDescTestTrainDF,ss,sc)
    val kmeans_training_data_df2 = dataframeManager.kmeansPreprocessing(jaccardSimilarityDF)

    val prediction_df2 = kmeans.runPrediction(kmeans_training_data_df2)
    kmeans.getMetrics(prediction_df2, "f1-score", ss, sc, "Jaccard")

    // Calculate Euclidean Similarity and apply KMeans

    val euclSimilarityDF = dataframeManager.calcEuclideanSimilarity(searchDF,joinDescTestTrainDF,ss,sc)
    val kmeans_training_data_df3 = dataframeManager.kmeansPreprocessing(euclSimilarityDF)

    val prediction_df3 = kmeans.runPrediction(kmeans_training_data_df3)
    kmeans.getMetrics(prediction_df3, "f1-score", ss, sc, "Euclidean")

    sc.stop()
  }


}