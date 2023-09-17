package com.CGUT.recommender

import java.net.InetAddress

import com.mongodb.casbah.commons.MongoDBObject
import com.mongodb.casbah.{MongoClient, MongoClientURI}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.elasticsearch.action.admin.indices.create.CreateIndexRequest
import org.elasticsearch.action.admin.indices.delete.DeleteIndexRequest
import org.elasticsearch.action.admin.indices.exists.indices.IndicesExistsRequest
import org.elasticsearch.common.settings.Settings
import org.elasticsearch.common.transport.InetSocketTransportAddress
import org.elasticsearch.transport.client.PreBuiltTransportClient


/*
  Movie 数据集
  mid         Int              电影的 ID
  name        String           电影的名称
  descri      String           电影的描述
  timelong    String           电影的时长
  shoot       String           电影拍摄时间
  issue       String           电影发布时间
  language    Array[String]    电影语言 每一项用“|”分割
  genres      Array[String]    电影所属类别 每一项用“|”分割
  director    Array[String]    电影的导演 每一项用“|”分割
  actors      Array[String]    电影的演员 每一项用“|”分割
 */
case class Movie(mid: Int, name: String, descri: String, timelong: String,
                 issue: String, shoot: String, language: String,
                 genres: String, actors: String, directors: String)

/*
  Rating 数据集
  uid         Int       用户的 ID
  mid         Int       电影的 ID
  score       Double    电影的分值
  timestamp   Long      评分的时间
 */
case class Rating(uid: Int, mid: Int, score: Double, timestamp: Int)

/*
 Tag 数据集
 uid        Int         用户的 ID
 mid        Int         电影的 ID
 tag        String      电影的标签
 timestamp  Long        标签的时间
 */
case class Tag(uid: Int, mid: Int, tag: String, timestamp: Int)

//把MongoDB和ES的配置封装成样例类

/**
 *
 * @param uri MongoDB连接
 * @param db  MongoDB数据库
 */
case class MongoConfig(uri: String, db: String)

/**
 *
 * @param httpHosts      http主机列表，逗号分割
 * @param transportHosts transport主机列表
 * @param index          需要操作的索引
 * @param clustername    集群名称，默认elasticsearch
 */
case class ESConfig(httpHosts: String, transportHosts: String, index: String, clustername: String)

object DataLoader {
  val MOVIE_DATA_PATH = "E:\\MovieRecommendSystem\\recommender\\DataLoader\\src\\main\\resources\\movies.csv"
  val RATING_DATA_PATH = "E:\\MovieRecommendSystem\\recommender\\DataLoader\\src\\main\\resources\\ratings.csv"
  val TAG_DATA_PATH = "E:\\MovieRecommendSystem\\recommender\\DataLoader\\src\\main\\resources\\tags.csv"

  val MONGODB_MOVIE_COLLECTION = "Movie"
  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_TAG_COLLECTION = "Tag"
  val ES_MOVIE_INDEX = "Movie"

  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://CentOS-7-107:27017/recommender",
      "mongo.db" -> "recommender",
      "es.httpHosts" -> "CentOS-7-107:9200",
      "es.transportHosts" -> "CentOS-7-107:9300",
      "es.index" -> "recommender",
      "es.cluster.name" -> "es-cluster"

    )
    //创建一个sparkConf
    val sparkConf = new SparkConf().setAppName("DataLoader").setMaster(config("spark.cores"))

    //创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._

    //加载数据
    val movieRDD = spark.sparkContext.textFile(MOVIE_DATA_PATH)
    //    转为DataFrame
    val movieDF = movieRDD.map(
      item => {
        val attr = item.split("\\^")
        Movie(attr(0).toInt, attr(1).trim, attr(2).trim, attr(3).trim, attr(4).trim, attr(5).trim, attr(6).trim, attr(7).trim, attr(8).trim, attr(9).trim)
      }
    ).toDF()

    val ratingRDD = spark.sparkContext.textFile(RATING_DATA_PATH)
    //将 ratingRDD 转换为 DataFrame
    val ratingDF = ratingRDD.map(item => {
      val attr = item.split(",")
      Rating(attr(0).toInt, attr(1).toInt, attr(2).toDouble, attr(3).toInt)
    }).toDF()


    val tagRDD = spark.sparkContext.textFile(TAG_DATA_PATH)
    //将 tagRDD 装换为 DataFrame
    val tagDF = tagRDD.map(item => {
      val attr = item.split(",")
      Tag(attr(0).toInt, attr(1).toInt, attr(2).trim, attr(3).toInt)
    }).toDF()

    // 声明一个隐式的配置对象
    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    // 将数据保存到 MongoDB 中
    storeDataInMongoDB(movieDF, ratingDF, tagDF)

    //数据预处理, 把movie对应的tag信息添加进去，加一列tag1|tag2|tag3...
    import org.apache.spark.sql.functions._
    /**
     * mid ,tags
     * tags: tag1|tag2|tag3...
     */
    val newTag = tagDF.groupBy($"mid")
      .agg(concat_ws("|", collect_set($"tag")).as("tags"))
      .select("mid", "tags")

    //newTag和movie做join，数据合在一起
    val movieWithTagsDF = movieDF.join(newTag, Seq("mid"), "left")

    implicit val esConfig = ESConfig(config("es.httpHosts"), config("es.transportHosts"), config("es.index"), config("es.cluster.name"))

    //保存数据到ES
    storeDataInES(movieWithTagsDF)

    spark.stop()
  }

  def storeDataInMongoDB(movieDF: DataFrame, ratingDF: DataFrame, tagDF: DataFrame)(implicit mongoConfig: MongoConfig): Unit = {
    // 新建一个MongoDB的连接
    val mongoClient = MongoClient(MongoClientURI(mongoConfig.uri))

    //如果mongodb中已经有相应的数据库，先删除
    mongoClient(mongoConfig.db)(MONGODB_MOVIE_COLLECTION).dropCollection()
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).dropCollection()
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).dropCollection()

    //    将DF数据写入对应的MongoDB表中
    movieDF
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()
    ratingDF
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()
    tagDF
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_TAG_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    //对数据表建索引
    mongoClient(mongoConfig.db)(MONGODB_MOVIE_COLLECTION).createIndex(MongoDBObject("mid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).createIndex(MongoDBObject("uid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).createIndex(MongoDBObject("mid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).createIndex(MongoDBObject("uid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).createIndex(MongoDBObject("mid" -> 1))
    //关闭数据库
    mongoClient.close()
  }

  def storeDataInES(movieDF: DataFrame)(implicit eSConfig: ESConfig): Unit = {
    //新建es配置
    val settings: Settings = Settings.builder().put("cluster.name", eSConfig.clustername).build()

    //新建一个es客户端
    val esClient = new PreBuiltTransportClient(settings)

    val REGEX_HOST_PORT = "(.+):(\\d+)".r
    eSConfig.transportHosts.split(",").foreach{
      case REGEX_HOST_PORT(host:String, port:String) => {
        esClient.addTransportAddress(new InetSocketTransportAddress(InetAddress.getByName(host), port.toInt))
      }
    }
    //需要清除掉 ES 中遗留的数据
    if(esClient.admin().indices().exists(new IndicesExistsRequest(eSConfig.index))
      .actionGet().
      isExists
    ){
      esClient.admin().indices().delete(new DeleteIndexRequest(eSConfig.index))
    }

    esClient.admin().indices().create(new CreateIndexRequest(eSConfig.index))
    //将数据写入到 ES 中
    movieDF
      .write
      .option("es.nodes",eSConfig.httpHosts)
      .option("es.http.timeout","100m")
      .option("es.mapping.id","mid")
      .mode("overwrite")
      .format("org.elasticsearch.spark.sql")
      .save(eSConfig.index + "/" + ES_MOVIE_INDEX)
  }
}
