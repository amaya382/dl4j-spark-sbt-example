package dl4jsparksbtsample

import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

object Main {
  def main(args: Array[String]) {
    val log = LoggerFactory.getLogger(this.getClass)
    val conf = new SparkConf()
      .setAppName("Example of DL4J on Spark")
      .setMaster("local[*]")
      .set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true))
    val sc = new SparkContext(conf)

    val seed = 12344
    val nIters = 1
    val nEpoch = 10
    val nTrain = 50000
    val nTest = 10000
    val nOut = 10
    val nSamples = nTrain + nTest
    val nCores = 8
    val batchSize = 20

    log.info("loading data")

    val (trainRDD, test) = {
      val mnist = new MnistDataSetIterator(1, nSamples, true)
      val _train = collection.mutable.ArrayBuffer.empty[DataSet]
      val _test = collection.mutable.ArrayBuffer.empty[DataSet]

      {
        var i = 0
        while (mnist.hasNext) {
          i += 1
          (if (i < 50000) _train else _test) += mnist.next
        }
      }

      val tr = sc.parallelize(_train)
      tr.persist(StorageLevel.MEMORY_ONLY)
      tr -> _test
    }

    log.info("building network")

    val nn = {
      val builder = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .iterations(nIters)
        .regularization(true)
        .l2(0.0005)
        .learningRate(0.1)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(Updater.ADAGRAD)
        .list
        .layer(0, new ConvolutionLayer.Builder(5, 5)
          .nIn(1)
          .stride(1, 1)
          .nOut(20)
          .weightInit(WeightInit.XAVIER)
          .activation("relu")
          .build)
        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
          .kernelSize(2, 2)
          .build)
        .layer(2, new ConvolutionLayer.Builder(5, 5)
          .nIn(20)
          .nOut(50)
          .stride(2, 2)
          .weightInit(WeightInit.XAVIER)
          .activation("relu")
          .build)
        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
          .kernelSize(2, 2)
          .build)
        .layer(4, new DenseLayer.Builder()
          .activation("relu")
          .weightInit(WeightInit.XAVIER)
          .nOut(200)
          .build)
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
          .nOut(10)
          .weightInit(WeightInit.XAVIER)
          .activation("softmax")
          .build)
        .backprop(true)
        .pretrain(false)
      new ConvolutionLayerSetup(builder, 28, 28, 1)
      new MultiLayerNetwork(builder.build)
    }
    nn.init()
    nn.setUpdater(null)

    val sparkNN = new SparkDl4jMultiLayer(sc, nn)

    log.info("starting training")

    (0 until nEpoch).foreach { i =>
      sparkNN.fitDataSet(trainRDD, nCores * batchSize, nCores)
      val eval = new Evaluation

      test.foreach { ds =>
        val output = nn.output(ds.getFeatureMatrix)
        eval.eval(ds.getLabels, output)
      }

      log.info(eval.stats)
    }

    log.info("finished")
  }
}

