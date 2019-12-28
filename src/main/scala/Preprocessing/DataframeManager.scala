package Preprocessing

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}


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
    val finalReshapedDf = sparkSession.createDataFrame(reshapedColumnsRDD).toDF("rProductUID", "rProductDescription")

    val trainDF = finalReshapedDf

    // Create tokens of words
    val tokenizer = new RegexTokenizer().setInputCol("rProductDescription").setOutputCol("rProductDescriptionWords")
      .setPattern("[\\W_]+")
    val tokenizedWordsDf = tokenizer.transform(trainDF)

    val rows: RDD[Row] = tokenizedWordsDf.rdd
    val changedRowsRDD = rows.map(x => (x(0).toString, changeImportantWords(x(1).toString)))

    val changedSpecialWordsDf = sparkSession.createDataFrame(changedRowsRDD).toDF("rProductUID", "rProductDescriptionWords")
    val tokenizerForChangedWordsDf = new RegexTokenizer().setInputCol("rProductDescriptionWords").setOutputCol("rChangedProductDescriptionWords")
      .setPattern("[\\W_]+")
    val tokenizedChangedSpecialWordsDf = tokenizerForChangedWordsDf.transform(changedSpecialWordsDf)

    val stopWordRemover = new StopWordsRemover()
      .setStopWords(stopWords)
      .setInputCol("rChangedProductDescriptionWords")
      .setOutputCol("rFilteredProductDescriptionWords")

    val filteredWordsDf = stopWordRemover.transform(tokenizedChangedSpecialWordsDf)

    val finalFilteredWordsDf = filteredWordsDf.select($"rProductUID", $"rFilteredProductDescriptionWords").withColumnRenamed("rFilteredProductDescriptionWords", "rFilteredWords")

    return finalFilteredWordsDf
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
    val finalReshapedDf = sparkSession.createDataFrame(reshapedColumnsRDD).toDF("rId", "rProductUID", "rProductTitle", "rRelevance")

    val trainDF = finalReshapedDf

    // Create tokens of words with RegexTokenizer
    val tokenizer = new RegexTokenizer().setInputCol("rProductTitle").setOutputCol("rProductTitleWords")
      .setPattern("[\\W_]+")
    val tokenizedWordsDf = tokenizer.transform(trainDF)

    val rows: RDD[Row] = tokenizedWordsDf.rdd
    val changedRowsRDD = rows.map(x => (x(0).toString, x(1).toString, x(2).toString, x(3).toString, changeImportantWords(x(4).toString)))

    val changedSpecialWordsDf = sparkSession.createDataFrame(changedRowsRDD).toDF("rId", "rProductUID", "rProductTitle", "rRelevance", "rProductTitleWords")
    val tokenizerForChangedWordsDf = new RegexTokenizer().setInputCol("rProductTitleWords").setOutputCol("rChangedProductTitleWords")
      .setPattern("[\\W_]+")
    val tokenizedChangedSpecialWordsDf = tokenizerForChangedWordsDf.transform(changedSpecialWordsDf)

    val stopWordRemover = new StopWordsRemover()
      .setStopWords(stopWords)
      .setInputCol("rChangedProductTitleWords")
      .setOutputCol("rFilteredProductTitleWords")

    val filteredWordsDf = stopWordRemover.transform(tokenizedChangedSpecialWordsDf)


    val finalFilteredWordsDf = filteredWordsDf.select($"rId", $"rProductUID", $"rFilteredProductTitleWords", $"rRelevance").withColumnRenamed("rFilteredProductTitleWords", "rFilteredWords")

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
    val tokenizerForChangedWordsDf = new RegexTokenizer().setInputCol("rFilteredWords").setOutputCol("rFilteredWordsChangedRegex")
      .setPattern("[\\W_]+")
    val tokenizedChangedSpecialWordsDf = tokenizerForChangedWordsDf.transform(changedSpecialWordsDf)

    val stopWordRemover = new StopWordsRemover()
      .setStopWords(stopWords)
      .setInputCol("rFilteredWordsChangedRegex").setOutputCol("rFilteredWordsChangedStopWords")
    val filteredWordsDf = stopWordRemover.transform(tokenizedChangedSpecialWordsDf)

    val finaResultDF = filteredWordsDf.select("rProductUID", "rFilteredWordsChangedStopWords").withColumnRenamed("rFilteredWordsChangedStopWords", "rFilteredWords")
      .orderBy("rProductUID")

    println("Adding column rID and rRelevance to the final joined Dataframe")
    val relevence = trainingDataframe.select("rID", "rProductUID", "rRelevance").orderBy("rProductUID")

    var relevence2 = relevence.withColumn("rowId1", monotonically_increasing_id())
    var finaResultDF2 = finaResultDF.withColumn("rowId2", monotonically_increasing_id())

    val joinedWithRelevanceFinalDF = finaResultDF2.as("df1").join(relevence2.as("df2"), finaResultDF2("rowId2") === relevence2("rowId1"), "inner")
      .select("df2.rID", "df1.rProductUID", "df1.rFilteredWords", "df2.rRelevance").orderBy("rProductUID")

    return joinedWithRelevanceFinalDF

  }

  /** Function for changing words that seem trivial e.g the degrees symbol to degrees word */

  def changeImportantWords(x: String): String = {

    val result = x
      .replaceAll("\\bSome.\\b", " ")
      .replaceAll("°", "degrees")
      .replaceFirst("WrappedArray", "")
      .replaceAll("""[\p{Punct}&&[^.]]""", "")

    return result
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

    val reshapedColumnsRDD = renamedRDD
      .map(x => (x(0).toString, x(1).toString + " " + x(2).toString))
      .groupByKey()
      .sortBy(x => x._1)
      .map(x => (x._1, changeImportantWords(x._2.toString.replace("CompactBuffer", ""))))

    val finalReshapedDf = sparkSession.createDataFrame(reshapedColumnsRDD).toDF("rProductUID", "rNameValues")
    finalReshapedDf.printSchema()
    val trainDF = finalReshapedDf

    val tokenizer = new RegexTokenizer().setInputCol("rNameValues").setOutputCol("rNameValuesWords")
      .setPattern("[\\W_]+")
    val tokenizedWordsDf = tokenizer.transform(trainDF)

    val rows: RDD[Row] = tokenizedWordsDf.rdd
    val changedRowsRDD = rows.map(x => (x(0).toString, changeImportantWords(x(1).toString)))

    val changedSpecialWordsDf = sparkSession.createDataFrame(changedRowsRDD).toDF("rProductUID", "rNameValuesWords")
    val tokenizerForChangedWordsDf = new RegexTokenizer().setInputCol("rNameValuesWords").setOutputCol("rChangedNameValuesWords")
      .setPattern("[\\W_]+")
    val tokenizedChangedSpecialWordsDf = tokenizerForChangedWordsDf.transform(changedSpecialWordsDf)

    val stopWordRemover = new StopWordsRemover()
      .setStopWords(stopWords) // This parameter is optional
      .setInputCol("rChangedNameValuesWords")
      .setOutputCol("rFilteredNameValuesWords")

    val filteredWordsDf = stopWordRemover.transform(tokenizedChangedSpecialWordsDf)


    val finalFilteredWordsDf = filteredWordsDf.select($"rProductUID", $"rFilteredNameValuesWords").withColumnRenamed("rFilteredNameValuesWords", "rFilteredWords")
    finalFilteredWordsDf.printSchema()

    finalFilteredWordsDf
  }

}