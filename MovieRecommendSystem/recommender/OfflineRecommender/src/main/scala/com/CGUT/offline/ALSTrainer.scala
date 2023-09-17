package com.CGUT.offline

import breeze.numerics.sqrt
import com.CGUT.offline.OfflineRecommender.MONGODB_RATING_COLLECTION
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object ALSTrainer {
  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://CentOS-7-107:27017/recommender",
      "mongo.db" -> "recommender"
    )
    val sparkConf = new SparkConf().setAppName("OfflineRecommender").setMaster(config("spark.cores"))
    //创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
    // 声明一个隐式的配置对象
    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    //加载评分数据
    val ratingRDD = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]
      .rdd
      .map(rating => Rating(rating.uid, rating.mid, rating.score))
      .cache()
    // 随机切分数据集，生成训练集和测试集
    val splits = ratingRDD.randomSplit(Array(0.8, 0.2))
    val trainingRDD = splits(0)
    val testRDD = splits(1)
    //模型参数选择，输出最优参数
    adjustALSParam(trainingRDD, testRDD)
    spark.close()
  }

  def adjustALSParam(trainData: RDD[Rating], testData: RDD[Rating]): Unit = {
    val result = for (rank <- Array(20, 50, 100); lambda <- Array(0.1, 0.01, 0.001))
      yield {
        val model = ALS.train(trainData, rank, 5, lambda)
        // 计算当前参数对应模型的rmse，返回Double
        val rmse = getRMSE(model, testData)
        (rank, lambda, rmse)
      }
    // 按照 rmse 排序
    println(result.minBy(_._3))
  }

  def getRMSE(model: MatrixFactorizationModel, data: RDD[Rating]): Double = {
    //计算预测评分
    val userMovies = data.map(item => (item.user, item.product))
    val predictRating = model.predict(userMovies)
    val real = data.map(item => ((item.user, item.product), item.rating))
    val predict = predictRating.map(item => ((item.user, item.product), item.rating))
    // 计算 RMSE
    sqrt(
      real.join(predict).map {
        case ((uid, mid), (real, pre)) =>
          // 真实值和预测值之间的差
          val err = real - pre
          err * err
      }.mean()
    )
  }
}
