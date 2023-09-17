package com.CGUT.content

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

//需要的数据源是电影内容信息
case class Movie(mid: Int, name: String, descri: String, timelong: String,
                 issue: String, shoot: String, language: String,
                 genres: String, actors: String, directors: String)
case class MongoConfig(uri: String, db: String)

// 标准推荐对象（基准推荐对象），mid,score
case class Recommendation(mid: Int, score: Double)

// 基于电影内容提取出的特征向量的电影相似度（电影推荐）列表
case class MovieRecs(mid: Int, recs: Seq[Recommendation])

object contentRecommender {
  // 定义表名
  val MONGODB_MOVIE_COLLECTION = "Movie"

  val CONTENT_MOVIE_RECS = "ContentMovieRecs"

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
    // 载入电影数据集
    val movieTagsDF = spark
      .read
      .option("uri",mongoConfig.uri)
      .option("collection",MONGODB_MOVIE_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Movie]
      .map(x => (x.mid, x.name, x.genres.map(c => if(c == '|') ' ' else c)))
      .toDF("mid", "name", "genres")
      .cache()

    // 实例化一个分词器，默认按空格分
    val tokenizer = new Tokenizer().setInputCol("genres").setOutputCol("words")
    // 用分词器做转换
    val wordsData = tokenizer.transform(movieTagsDF)
    // 定义一个 HashingTF 工具
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(50)
    // 用 HashingTF 做处理
    val featurizedData = hashingTF.transform(wordsData)
    // 定义一个 IDF 工具
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    // 将词频数据传入，得到 idf 模型（统计文档）
    val idfModel = idf.fit(featurizedData)
    // 用 tf-idf 算法得到新的特征矩阵
    val rescaledData = idfModel.transform(featurizedData)
//    rescaledData.show(truncate = false)
    // 从计算得到的 rescaledData 中提取特征向量
    val movieFeatures = rescaledData.map{
      row => ( row.getAs[Int]("mid"), row.getAs[SparseVector]("features").toArray )
    }
      .rdd
      .map(x =>
        (x._1, new DoubleMatrix(x._2) )
      )
//    movieFeatures.collect().foreach(println)

    // 对所有电影俩俩计算他们的相似度，先做笛卡尔积
    val movieRecs = movieFeatures.cartesian(movieFeatures)
      .filter {
        case (a, b) => a._1 != b._1
      }
      .map {
        case (a, b) => val simScore = this.consinSim(a._2, b._2) // 求余弦相似度
          (a._1, (b._1, simScore))
      }
      .filter(_._2._2 > 0.6)
      .groupByKey()
      .map {
        case (mid, items) => MovieRecs(mid, items.toList.sortWith(_._2 > _._2).map(x => Recommendation(x._1, x._2)))
      }
      .toDF()

    movieRecs.write
      .option("uri", mongoConfig.uri)
      .option("collection", CONTENT_MOVIE_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    spark.stop()
  }
  //求向量余弦相似度
  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix): Double = {
    movie1.dot(movie2) / (movie1.norm2() * movie2.norm2())
  }
}
