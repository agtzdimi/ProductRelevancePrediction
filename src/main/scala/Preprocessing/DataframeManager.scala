package Preprocessing

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.feature.ChiSqSelector
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, MatrixEntry}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.Pipeline


class DataframeManager() extends Serializable {

  def getProdDescrDF(inputFilePath: String, sparkSession: SparkSession, sc: SparkContext): DataFrame = {

    val stopWordsInput = sc.textFile("src/main/resources/stopwords.csv")
    val stopWords = stopWordsInput.flatMap(x => x.split("\\W+")).collect()

    import sparkSession.implicits._ // For implicit conversions like converting RDDs to DataFrames

    // Read the contents of the csv file in a dataframe
    val basicTrainDF: DataFrame = sparkSession.read.option("header", "true").csv(inputFilePath)

    // Rename the columns of the dataframe
    val newColumnNames = Seq("rProductUID", "rProductDescription")
    val renamedDF = basicTrainDF.toDF(newColumnNames: _*)

    val renamedRDD: RDD[Row] = renamedDF.rdd
    val reshapedColumnsRDD = renamedRDD.map(x => (x(0).toString, x(1).toString))
    val finalReshapedDf = sparkSession.createDataFrame(reshapedColumnsRDD)
      .toDF("rProductUID", "rProductDescription")

    val trainDF = finalReshapedDf

    // Create tokens of words
    val tokenizer = new RegexTokenizer().setInputCol("rProductDescription")
      .setOutputCol("rProductDescriptionWords")
      .setPattern("[\\W_]+")
    val tokenizedWordsDf = tokenizer.transform(trainDF)

    val rows: RDD[Row] = tokenizedWordsDf.rdd
    val changedRowsRDD = rows.map(x => (x(0).toString, changeImportantWords(x(1).toString)))

    val changedSpecialWordsDf = sparkSession.createDataFrame(changedRowsRDD)
      .toDF("rProductUID", "rProductDescriptionWords")
    val tokenizerForChangedWordsDf = new RegexTokenizer().setInputCol("rProductDescriptionWords")
      .setOutputCol("rChangedProductDescriptionWords")
      .setPattern("[\\W_]+")
    val tokenizedChangedSpecialWordsDf = tokenizerForChangedWordsDf.transform(changedSpecialWordsDf)

    val stopWordRemover = new StopWordsRemover().setStopWords(stopWords)
      .setInputCol("rChangedProductDescriptionWords")
      .setOutputCol("rFilteredProductDescriptionWords")

    val filteredWordsDf = stopWordRemover.transform(tokenizedChangedSpecialWordsDf)

    val finalFilteredWordsDf = filteredWordsDf.select($"rProductUID", $"rFilteredProductDescriptionWords")
      .withColumnRenamed("rFilteredProductDescriptionWords", "rFilteredWords")
    finalFilteredWordsDf.take(5).foreach(println)
    return finalFilteredWordsDf
  }

  /** Function for changing words that seem trivial e.g the degrees symbol to degrees word */

  def changeImportantWords(x: String): String = {

    val result = x
      .replaceAll("\\bSome.\\b", " ")
      .replaceAll("°", "degrees")
      .replaceFirst("WrappedArray", "")
    //         .replaceAll("""[\p{Punct}&&[^.]]""", "")

    return result
  }

  def getTrainDF(inputFilePath: String, sparkSession: SparkSession, sc: SparkContext): DataFrame = {

    val stopWordsInput = sc.textFile("src/main/resources/stopwords.csv")
    val stopWords = stopWordsInput.flatMap(x => x.split("\\W+")).collect()

    import sparkSession.implicits._ // For implicit conversions like converting RDDs to DataFrames

    // Read the contents of the csv file in a dataframe
    val basicTrainDF: DataFrame = sparkSession.read.option("header", "true").csv(inputFilePath)

    // Rename the columns of the dataframe
    val newColumnNames = Seq("rId", "rProductUID", "rProductTitle", "rSearchTerm", "rRelevance")
    val renamedDF = basicTrainDF.toDF(newColumnNames: _*)

    val renamedRDD: RDD[Row] = renamedDF.rdd
    val reshapedColumnsRDD = renamedRDD.map(x => (x(0).toString, x(1).toString, x(2).toString + " " + x(3).toString, x(4).toString))
    val finalReshapedDf = sparkSession.createDataFrame(reshapedColumnsRDD)
      .toDF("rId", "rProductUID", "rProductTitle", "rRelevance")

    val trainDF = finalReshapedDf

    // Create tokens of words with RegexTokenizer
    val tokenizer = new RegexTokenizer().setInputCol("rProductTitle")
      .setOutputCol("rProductTitleWords")
      .setPattern("[\\W_]+")
    val tokenizedWordsDf = tokenizer.transform(trainDF)

    val rows: RDD[Row] = tokenizedWordsDf.rdd
    val changedRowsRDD = rows.map(x => (x(0).toString, x(1).toString, x(2).toString, x(3).toString, changeImportantWords(x(4).toString)))

    val changedSpecialWordsDf = sparkSession.createDataFrame(changedRowsRDD)
      .toDF("rId", "rProductUID", "rProductTitle", "rRelevance", "rProductTitleWords")
    val tokenizerForChangedWordsDf = new RegexTokenizer().setInputCol("rProductTitleWords")
      .setOutputCol("rChangedProductTitleWords")
      .setPattern("[\\W_]+")
    val tokenizedChangedSpecialWordsDf = tokenizerForChangedWordsDf.transform(changedSpecialWordsDf)

    val stopWordRemover = new StopWordsRemover().setStopWords(stopWords)
      .setInputCol("rChangedProductTitleWords")
      .setOutputCol("rFilteredProductTitleWords")

    val filteredWordsDf = stopWordRemover.transform(tokenizedChangedSpecialWordsDf)

    val finalFilteredWordsDf = filteredWordsDf.select($"rId", $"rProductUID", $"rFilteredProductTitleWords", $"rRelevance")
      .withColumnRenamed("rFilteredProductTitleWords", "rFilteredWords")

    return finalFilteredWordsDf
  }

  def getSearchTermDF(inputFilePath: String, sparkSession: SparkSession, sc: SparkContext): DataFrame = {

    import sparkSession.implicits._ // For implicit conversions like converting RDDs to DataFrames

    val basicTrainDF: DataFrame = sparkSession.read.option("header", "true").csv(inputFilePath)

    val newColumnNames = Seq("rId", "rProductUID", "rProductTitle", "rSearchTerm", "rRelevance")
    val renamedDF = basicTrainDF.toDF(newColumnNames: _*).select("rId", "rProductUID", "rSearchTerm")

    val trainDF = renamedDF

    // Create tokens of words with RegexTokenizer
    val tokenizer = new RegexTokenizer().setInputCol("rSearchTerm").setOutputCol("rSearchTermFeatures")
      .setPattern("[\\W_]+")
    val tokenizedWordsDf = tokenizer.transform(trainDF)

    val finalFilteredWordsDf = tokenizedWordsDf.select($"rId", $"rProductUID", $"rSearchTerm", $"rSearchTermFeatures")

    return finalFilteredWordsDf

  }

  def getIDF(trainDataframe: DataFrame, sparkSession: SparkSession, sc: SparkContext): DataFrame = {

    import sparkSession.implicits._

    val hashingTF = new HashingTF().setInputCol("rFilteredWords").setOutputCol("rRawFeatures").setNumFeatures(20000)
    val featurizedDF = hashingTF.transform(trainDataframe)

    val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("rFeatures")
    val idfM = idf.fit(featurizedDF)
    val completeDF = idfM.transform(featurizedDF)

    val udf_toDouble = udf((s: String) => s.toDouble)
    val finalIDF = completeDF.select($"rProductUID", $"rFilteredWords", $"rFeatures", $"rRelevance", udf_toDouble($"rRelevance").as("rrLabel"))

    return finalIDF


  }

  def joinDataframes(trainingDataframe: DataFrame, testDataframe: DataFrame, sparkSession: SparkSession, sc: SparkContext): DataFrame = {

    val stopWordsInput = sc.textFile("src/main/resources/stopwords.csv")
    val stopWords = stopWordsInput.flatMap(x => x.split("\\W+")).collect()

    val trainingText = trainingDataframe.select("rProductUID", "rFilteredWords", "rRelevance")
    val testText = testDataframe.select("rProductUID", "rFilteredWords")

    val trainingRDD: RDD[Row] = trainingDataframe.rdd
    val testRDD: RDD[Row] = testDataframe.rdd

    // Change to rdd to groupByKey and bring all search terms together //
    val changedRenamedTestRDD = testRDD.
      map(x => (x(0).toString, x(1).toString)).
      groupByKey().sortBy(x => x._1).map(x => (x._1, changeImportantWords(x._2.toString.replace("CompactBuffer", ""))))

    // In training dataframe the columns we need are in index 1 and 2, productUID and filteredWords respectively
    val changedRenamedTrainingRDD = trainingRDD.
      map(x => (x(1).toString, x(2).toString)).
      sortBy(x => x._1).map(x => (x._1, changeImportantWords(x._2.toString.replace("CompactBuffer", ""))))

    val joinedRDDResult = changedRenamedTrainingRDD.leftOuterJoin(changedRenamedTestRDD)

    val newRDD = joinedRDDResult.
      map(x => (x._1.toString, x._2.toString))
    print("newRDD count : " + newRDD.count())

    val finalRenamedDF = sparkSession.createDataFrame(newRDD).toDF("rProductUID", "rFilteredWords")

    val testDF = finalRenamedDF
    testDF.repartition(4)

    val tokenizer = new RegexTokenizer().setInputCol("rFilteredWords")
      .setPattern("[\\W_]+")
    val tokenizedWordsDf = tokenizer.transform(testDF)
    tokenizedWordsDf.select("rFilteredWords")

    val rows: RDD[Row] = tokenizedWordsDf.rdd
    val changedRowsRDD = rows.map(x => (x(0).toString, changeImportantWords(x(1).toString)))

    val changedSpecialWordsDf = sparkSession.createDataFrame(changedRowsRDD).toDF("rProductUID", "rFilteredWords")
    val tokenizerForChangedWordsDf = new RegexTokenizer().setInputCol("rFilteredWords")
      .setOutputCol("rFilteredWordsChangedRegex")
      .setPattern("[\\W_]+")
    val tokenizedChangedSpecialWordsDf = tokenizerForChangedWordsDf.transform(changedSpecialWordsDf)

    val stopWordRemover = new StopWordsRemover()
      .setStopWords(stopWords)
      .setInputCol("rFilteredWordsChangedRegex").setOutputCol("rFilteredWordsChangedStopWords")
    val filteredWordsDf = stopWordRemover.transform(tokenizedChangedSpecialWordsDf)

    val finaResultDF = filteredWordsDf.select("rProductUID", "rFilteredWordsChangedStopWords")
      .withColumnRenamed("rFilteredWordsChangedStopWords", "rFilteredWords")
      .orderBy("rProductUID")

    println("Adding column rID and rRelevance to the final joined Dataframe")
    val relevence = trainingDataframe.select("rID", "rProductUID", "rRelevance")
      .orderBy("rProductUID")

    var relevence2 = relevence.withColumn("rowId1", monotonically_increasing_id())
    var finaResultDF2 = finaResultDF.withColumn("rowId2", monotonically_increasing_id())

    val joinedWithRelevanceFinalDF = finaResultDF2.as("df1")
      .join(relevence2.as("df2"), finaResultDF2("rowId2") === relevence2("rowId1"), "inner")
      .select("df2.rID", "df1.rProductUID", "df1.rFilteredWords", "df2.rRelevance")
      .orderBy("rProductUID")

    return joinedWithRelevanceFinalDF

  }

  def getAttrDF(inputFilePath: String, sparkSession: SparkSession, sc: SparkContext): DataFrame = {

    val stopWordsInput = sc.textFile("src/main/resources/stopwords.csv")
    val stopWords = stopWordsInput.flatMap(x => x.split("\\W+")).collect()

    import sparkSession.implicits._ // For implicit conversions like converting RDDs to DataFrames

    val basicTrainDF: DataFrame = sparkSession.read.option("header", "true").csv(inputFilePath)

    val newColumnNames = Seq("rProductUID", "rName", "rValue")
    val renamedDF = basicTrainDF.toDF(newColumnNames: _*)

    val removedNullDF = renamedDF.filter($"rProductUID".isNotNull)
    val renamedRDD: RDD[Row] = removedNullDF.rdd

    val reshapedColumnsRDD = renamedRDD.map(x => (x(0).toString, x(1).toString + " " + x(2).toString))
      .groupByKey()
      .sortBy(x => x._1)
      .map(x => (x._1, changeImportantWords(x._2.toString.replace("CompactBuffer", ""))))

    val finalReshapedDf = sparkSession.createDataFrame(reshapedColumnsRDD)
      .toDF("rProductUID", "rNameValues")
    finalReshapedDf.printSchema()
    val trainDF = finalReshapedDf

    val tokenizer = new RegexTokenizer().setInputCol("rNameValues")
      .setOutputCol("rNameValuesWords")
      .setPattern("[\\W_]+")
    val tokenizedWordsDf = tokenizer.transform(trainDF)

    val rows: RDD[Row] = tokenizedWordsDf.rdd
    val changedRowsRDD = rows.map(x => (x(0).toString, changeImportantWords(x(1).toString)))

    val changedSpecialWordsDf = sparkSession.createDataFrame(changedRowsRDD).toDF("rProductUID", "rNameValuesWords")
    val tokenizerForChangedWordsDf = new RegexTokenizer().setInputCol("rNameValuesWords")
      .setOutputCol("rChangedNameValuesWords")
      .setPattern("[\\W_]+")
    val tokenizedChangedSpecialWordsDf = tokenizerForChangedWordsDf.transform(changedSpecialWordsDf)

    val stopWordRemover = new StopWordsRemover().setStopWords(stopWords) // This parameter is optional
      .setInputCol("rChangedNameValuesWords")
      .setOutputCol("rFilteredNameValuesWords")

    val filteredWordsDf = stopWordRemover.transform(tokenizedChangedSpecialWordsDf)


    val finalFilteredWordsDf = filteredWordsDf.select($"rProductUID", $"rFilteredNameValuesWords")
      .withColumnRenamed("rFilteredNameValuesWords", "rFilteredWords")

    finalFilteredWordsDf
  }

  def unsupervisedProdPreprocessing(prodFilePath: String, sparkSession: SparkSession, sc: SparkContext): DataFrame = {
    val stopWordsInput = sc.textFile("src/main/resources/stopwords.csv")
    val stopWords = stopWordsInput.flatMap(x => x.split("\\W+")).collect() // For implicit conversions like converting RDDs to DataFrames

    val prod_df = sparkSession.read.option("header", true).csv(prodFilePath)
    // Create tokens of words
    val tokenizer = new RegexTokenizer().setInputCol("search_term")
      .setOutputCol("tokenized_search_term")
      .setPattern("[\\W_]+")
    val tokenized_df = tokenizer.transform(prod_df)

    // Remove stopwords
    val stopWordRemover = new StopWordsRemover().setStopWords(stopWords)
      .setInputCol("tokenized_search_term")
      .setOutputCol("filtered_search_term")
    val filtered_words_df = stopWordRemover.transform(tokenized_df)

    // Hasing TF
    val hashingTF = new HashingTF().setInputCol("filtered_search_term")
      .setOutputCol("raw_features")
      .setNumFeatures(144)
    val raw_feature_df = hashingTF.transform(filtered_words_df)

    // Apply IDF
    val idf = new IDF().setInputCol("raw_features")
      .setOutputCol("prod_features")
    val idf_model = idf.fit(raw_feature_df)
    val prod_feature_df = idf_model.transform(raw_feature_df)

    // Feauture normalization
    val normalizer = new Normalizer()
      .setInputCol("prod_features")
      .setOutputCol("normalized_prod_features")
    val norm_features = normalizer.transform(prod_feature_df)

    return norm_features
  }

  def unsupervisedDescrPreprocessing(descrFilePath: String, sparkSession: SparkSession, sc: SparkContext): DataFrame = {
    val stopWordsInput = sc.textFile("src/main/resources/stopwords.csv")
    val stopWords = stopWordsInput.flatMap(x => x.split("\\W+")).collect() // For implicit conversions like converting RDDs to DataFrames

    val descr_df = sparkSession.read.option("header", true).csv(descrFilePath)

    // Create tokens of words
    val tokenizer = new RegexTokenizer().setInputCol("product_description")
      .setOutputCol("tokenized_product_description")
      .setPattern("[\\W_]+")
    val tokenized_df = tokenizer.transform(descr_df)

    // Remove stopwords
    val stopWordRemover = new StopWordsRemover().setStopWords(stopWords)
      .setInputCol("tokenized_product_description")
      .setOutputCol("filtered_description")
    val filtered_words_df = stopWordRemover.transform(tokenized_df)

    // Hasing TF
    val hashingTF = new HashingTF().setInputCol("filtered_description")
      .setOutputCol("raw_features")
      .setNumFeatures(144)
    val raw_feature_df = hashingTF.transform(filtered_words_df)

    // Apply IDF
    val idf = new IDF().setInputCol("raw_features")
      .setOutputCol("descr_features")
    val idf_model = idf.fit(raw_feature_df)
    val descr_feature_df = idf_model.transform(raw_feature_df)

    // Feauture normalization
    val normalizer = new Normalizer()
      .setInputCol("descr_features")
      .setOutputCol("normalized_descr_features")
    val norm_features = normalizer.transform(descr_feature_df)

    return norm_features
  }

  def unsupervisedFeatureSelection(prod_df: DataFrame, descr_df: DataFrame, sparkSession: SparkSession, sc: SparkContext): DataFrame = { // For implicit conversions like converting RDDs to DataFrames

    // Create a joined data view
    val joined_df = prod_df.join(descr_df, "product_uid")
    // Assemple feature vector
    val assembler = new VectorAssembler()
      .setInputCols(Array("normalized_descr_features", "normalized_prod_features"))
      .setOutputCol("features")
    //
    val output = assembler.transform(joined_df).select("features")

    // Apply cosine similarity in the "features" column, to create a new "similarity" column
    val outputRDD = output.rdd
    val indexedRDD = outputRDD.map(_.getAs[org.apache.spark.ml.linalg.Vector](0))
      .map(org.apache.spark.mllib.linalg.Vectors.fromML)
      .zipWithIndex.map { case (v, i) => IndexedRow(i, v) }

    val similarity_matrix = new IndexedRowMatrix(indexedRDD).toCoordinateMatrix.transpose.toRowMatrix.columnSimilarities

    val temp_rdd = similarity_matrix.entries
      .map { case MatrixEntry(row: Long, col: Long, sim: Double) => Array(sim).mkString(",") }
      .map { a => Row(a) }
    val dfschema = StructType(Array(StructField("Similarity", StringType)))
    val rddToDF = sparkSession.createDataFrame(temp_rdd, dfschema)
      .select("Similarity")
      .withColumn("rowIndex", monotonically_increasing_id())

    val relevanceDF = joined_df.select("product_uid", "relevance")
      .withColumn("rowIndex", monotonically_increasing_id())

    // Binarize relevance: 0<- 1.5 <
    // relevance < 1.5 -> 0
    // relevance =>1.5 -> 1
    val relevance_binarizer = udf((x: Double) => if (x < 1.5) 0 else 1)

    val cosineDF = rddToDF.join(relevanceDF, rddToDF("rowIndex") === relevanceDF("rowIndex"), "inner")
      .withColumn("binarized_relevance", lit(relevance_binarizer(col("relevance").cast("Double"))))
      .withColumn("Similarity", col("Similarity").cast("Double"))
      .drop("rowIndex")

    return cosineDF

  }

  def kmeansPreprocessing(df: DataFrame): DataFrame = {
    val vectorizer = new VectorAssembler()
      .setInputCols(Array("Similarity"))
      .setOutputCol("Similarity_as_vector")

    val trainingData = vectorizer.transform(df)
    return trainingData
  }

  def getTopFeatures(sc: SparkContext, numOfFeat: Int): RDD[LabeledPoint] = {
    val data = MLUtils.loadLibSVMFile(sc, "src/main/resources/trainLabeledVectors.csv")

    val selector = new ChiSqSelector(numOfFeat)
    // Create ChiSqSelector model (selecting features)
    val transformer = selector.fit(data)
    // Filter the top 50 features from each feature vector
    val filteredData: RDD[LabeledPoint] = data.map { lp =>
      LabeledPoint(lp.label, transformer.transform(lp.features))
    }

    return filteredData
  }

  def calcJacccardSimilarity(searchDataframe: DataFrame,featuresDataframe:DataFrame,ss : SparkSession,sc: SparkContext) : DataFrame = {
    /*First Find the TF using Hashing TF. CountVectorizer can also be used */
    val hashingTF = new HashingTF().setInputCol("rFilteredWords").setOutputCol("rTFFeatures").setNumFeatures(20000)
    val featurizedDF = hashingTF.transform(featuresDataframe)
    val featuresTF = featurizedDF.select("rID","rTFFeatures","rRelevance")

    val joinedDF = featuresTF.join(searchDataframe,"rID").select("rID","rProductUID","rSearchTermFeatures","rTFFeatures","rRelevance").orderBy("rID")

    /*Create MinHash object*/
    val lsh = new MinHashLSH().setInputCol("rTFFeatures").setOutputCol("LSH").setNumHashTables(3)

    /*Create Pipeline model*/
    val pipe = new Pipeline().setStages(Array(lsh))
    val pipeModel = pipe.fit(joinedDF)

    /*Create the new DF*/

    val transformedDF = pipeModel.transform(joinedDF)

    /*Create Transformer*/
    val transformer = pipeModel.stages

    /*MinHashModel*/
    val tempMinHashModel = transformer.last.asInstanceOf[MinHashLSHModel]
    val threshold = 1.5

    /*Just a udf for converting string to double*/
    val udf_toDouble = udf( (s: String) => s.toDouble )


    /*Perform the Similarity with self-join*/
    /*Find the distance of pairs which is lower than the given threshold*/

    val preSimilarityDF = tempMinHashModel.approxSimilarityJoin(transformedDF,transformedDF,threshold)
      .select(udf_toDouble(col("datasetA.rRelevance")).alias("relevance"),
        col("distCol"))

    /*Make a vector of the distCol and name it Similarity. It will be needed when using the df for the ML Models*/
    val vectorAssem = new VectorAssembler()
      .setInputCols(Array("distCol"))
      .setOutputCol("Similarity")

    val relevance_binarizer = udf((x: Double) => if (x < 1.5) 0 else 1)

    val jaccardSimilarityDF = vectorAssem.transform(preSimilarityDF).select("Similarity","relevance")
      .withColumn("binarized_relevance", lit(relevance_binarizer(col("relevance").cast("Double"))))

    jaccardSimilarityDF
  }

  def calcEuclideanSimilarity(searchDataframe: DataFrame,featuresDataframe:DataFrame,ss : SparkSession,sc: SparkContext) : DataFrame ={
    /*First Find the TF using Hashing TF. CountVectorizer can also be used */
    val hashingTF = new HashingTF().setInputCol("rFilteredWords").setOutputCol("rTFFeatures").setNumFeatures(20000)
    val featurizedDF = hashingTF.transform(featuresDataframe)

    val featuresTF = featurizedDF.select("rID","rTFFeatures","rRelevance")
    val joinedDF = featuresTF.join(searchDataframe,"rID").select("rID","rSearchTermFeatures","rTFFeatures","rRelevance").orderBy("rID")

    val brp = new BucketedRandomProjectionLSH()
      .setBucketLength(2.0)
      .setNumHashTables(3)
      .setInputCol("rTFFeatures")
      .setOutputCol("BRPHashes")

    /*Create Pipeline model*/
    val pipe = new Pipeline().setStages(Array(brp))
    val pipeModel = pipe.fit(featuresTF)

    /*Transform the Dataframe*/
    val transformedDF = pipeModel.transform(joinedDF)

    /*Create Transformer*/
    val transformer = pipeModel.stages
    /*MinHashModel*/
    val tempMinHashModel = transformer.last.asInstanceOf[BucketedRandomProjectionLSHModel]

    /*Threshold*/
    val threshold = 1.5

    /*Just a udf for converting string to double*/
    val udf_toDouble = udf( (s: String) => s.toDouble )

    val preEuclideanSimilarityDF = tempMinHashModel.approxSimilarityJoin(transformedDF,transformedDF,threshold,"distCol")
      .select(udf_toDouble(col("datasetA.rRelevance")).alias("relevance"),
        col("distCol"))

    /*Make a vector of the distCol and name it Similarity. It will be needed when using the df for the ML Models*/
    val vectorAssem2 = new VectorAssembler()
      .setInputCols(Array("distCol"))
      .setOutputCol("Similarity")

    val relevance_binarizer = udf((x: Double) => if (x < 1.5) 0 else 1)
    val euclideanSimilarityDF = vectorAssem2.transform(preEuclideanSimilarityDF).select("Similarity","relevance")
      .withColumn("binarized_relevance", lit(relevance_binarizer(col("relevance").cast("Double"))))

    euclideanSimilarityDF
  }

}