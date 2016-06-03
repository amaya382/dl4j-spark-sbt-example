package dl4jsparksbtexample.mlp

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.canova.api.records.reader.impl.CSVRecordReader
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.canova.RecordReaderFunction
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import scala.io.Source

object Iris {
  def main(args: Array[String]) {
    val log = LoggerFactory.getLogger(getClass)
    val conf = new SparkConf()
      .setAppName("Example of DL4J on Spark(MLP-Iris)")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)

    val seed = 12344
    val nIters = 1
    val nEpoch = 10
    val labelIdx = 4
    val nOutputClasses = 3
    val nInputs = 4
    val nOutputs = 3

    log.info("loading data")

    val trainRDD = {
      val recordReader = new CSVRecordReader(0, ",")
      val irisDataLines = getIrisDataLines(sc)
      irisDataLines.map(l => new RecordReaderFunction(recordReader, labelIdx, nOutputClasses).call(l))
    }

    log.info("building network")

    val nn = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(nIters)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.1)
      .l1(0.01)
      .regularization(true)
      .l2(0.001)
      .list
      .layer(0, new DenseLayer.Builder()
        .nIn(nInputs)
        .nOut(3)
        .activation("tanh")
        .weightInit(WeightInit.XAVIER)
        .build)
      .layer(1, new DenseLayer.Builder()
        .nIn(3)
        .nOut(2)
        .activation("tanh")
        .weightInit(WeightInit.XAVIER)
        .build)
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .nIn(2)
        .nOut(nOutputs)
        .activation("softmax")
        .weightInit(WeightInit.XAVIER)
        .build)
      .backprop(true)
      .pretrain(false)
      .build)

    nn.init()
    nn.setUpdater(null)

    val sparkNN = new SparkDl4jMultiLayer(sc, nn)

    log.info("starting training")

    (0 until nEpoch).map { _ =>
      sparkNN.fitDataSet(trainRDD)
      nn.params.data.asFloat.clone
    }.foreach(res => log.info(res.mkString(",")))

    log.info("finished")
  }

  private def getIrisDataLines(sc: SparkContext): RDD[String] = {
    val irisDataContents = Source.fromURL(
      getClass.getResource("/iris_shuffled_normalized_csv.txt"))
      .getLines.takeWhile(_ != "").toSeq
    sc.parallelize(irisDataContents)
  }
}

