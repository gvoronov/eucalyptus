package eucalyptus.decisiontree

import scala.language.implicitConversions
import scala.util.Random
import scala.util.control.Breaks._
import scala.collection.mutable.{Map => MutableMap}
import scala.collection.mutable.Buffer

import breeze.stats.distributions.Uniform

import koalas.dataframe.DataFrame
import koalas.series.Series
import koalas.row.Row
import koalas.datavalue._
import koalas.numericalops.NumericalOps._

import eucalyptus.tree._
import eucalyptus.util.TreeUtil

object DecisionTree {
  implicit def wrapToOption[T](x: T) = Option[T](x)
}

abstract class DecisionTree(
    val maxSplitPoints: Int = 10, val minSplitPoints: Int = 1, val maxDepth: Int = 100,
    val minSamplesSplit: Int = 2, val minSamplesLeaf: Int = 1) {
  var tree: Option[BiTree] = None
  var predictorColNames: Option[List[String]] = None
  var responseColName: Option[String] = None
  var weightColName: Option[String] = None
  var maxFeaturesPerSplit: Option[Int] = None

  protected type FeatureSupportDict = Option[MutableMap[String, Option[SupportDict]]]
  protected type BestSplit = Tuple4[
    Option[NumericalValue], Option[NumericalValue], Option[DataFrame],  Option[DataFrame]]

  protected case class SupportDict(
      val bins: Option[Vector[NumericalValue]] = None, val maxRefined: Option[Boolean] = None,
      val catMap: Option[Map[DataValue, NumericalValue]] = None,
      val catMappedData: Option[Series[NumericalValue]] = None)

  /**
   * [x description]
   * @param x
   * @param y
   * @param predictors
   * @param response
   * @param weights
   */
  def fit(
      x: DataFrame, y: Option[Series[DataValue]] = None, predictors: Option[List[String]] = None,
      response: Option[String] = None, weights: Option[Series[NumericalValue]] = None): Unit = {
    var parsedX: DataFrame = (y, predictors, response) match {
      case (None, Some(predictors), Some(response)) => {
        // Perform basic checks
        if (! predictors.toSet.subsetOf(x.getColumns.toSet))
          throw new RuntimeException("Invalid preditors argument!")
        if (! x.getColumns.toSet.contains(response))
          throw new RuntimeException("Invalid response argument!")

        this.predictorColNames = Some(predictors)
        this.responseColName = Some(response)

        x.select(predictors :+ response)
      }
      case (None, Some(predictors), None) => throw new RuntimeException("Not implemented yet!")
      case (None, None, Some(response)) => throw new RuntimeException("Not implemented yet!")
      case (None, None, None) => throw new RuntimeException("Not implemented yet!")
      case (Some(y), Some(predictors), None) => throw new RuntimeException("Not implemented yet!")
      case (Some(y), None, None) => throw new RuntimeException("Not implemented yet!")
      case _ => throw new RuntimeException("Improper arguments passed to fit method!")
    }

    weightColName = Some(getDistinctColumnName(this.predictorColNames.get :+ this.responseColName.get, "weight"))

    parsedX = weights match {
      case Some(weights) => parsedX.update(weightColName.get, weights)
      case None => parsedX.update(weightColName.get, NumericalValue(1))
    }

    maxFeaturesPerSplit = Some(getMaxFeaturesPerSplit)
    // fitRecursive(parsedX, 0)
  }
  // def predict[T](x: DataFrame): T
  // def predict[T](x: Row): T

  /**
   * [data description]
   * @type {[type]}
   */
  private def fitRecursive(
      data: DataFrame, depth: Int, support: FeatureSupportDict = None,
      parent: Option[BiNode] = None, key: Option[String] = None): Unit = {
    val mySupport = support.getOrElse(
      MutableMap(
        predictorColNames.get.map(predictor => (predictor -> (None: Option[SupportDict]))).toSeq: _*
      )
    )
    var costImprovement: NumericalValue = NumericalValue(0)
    var feature: Option[String] = None
    var split: Option[NumericalValue] = None
    var leftData: Option[DataFrame] = None
    var rightData: Option[DataFrame] = None

    // Find optional split, if enough samples in parent node to split and max depth not exceeded.
    if (data[NumericalValue](weightColName.get).sum >= minSamplesSplit && depth < maxDepth) {
      var numFeaturesConsidered: Int = 0
      breakable {
        for (feature <- Random.shuffle(predictorColNames.get)) {
          mySupport(feature) = preprocessData(feature, data, mySupport(feature))
        }
      }

      // Create child node or leaf (figure out by context)
      // val child: Node = if (costImprovement > 0) {
      //   if support(selectFeature).catMap.isDefined
      //     throw new RuntimeException("Not implemnted yet!")
      //   else {}
      // } else {}

      // Mount child in tree

      // Fit deeper branches if current child node is not a leaf
      // if (! child.isLeaf) {
      //   fitRecursive(leftData, depth+1, support?,, child, false)
      //   fitRecursive(rightData, depth+1, support?,, child, true)
      // }
    }

  }

  /**
   * Internal method used to prepare data for efficeint optimal split search as part of the tree
   * fitging procedure.
   * @param feature current feature being considered
   * @param data is just the portion o fth edata contained in the current node region
   * @param auxData contains the result of preprocessing produced in previous splits
   */
  private def preprocessData(feature: String, data: DataFrame, auxData: Option[SupportDict]): Option[SupportDict] = {
    data.getSchema.nameToField(feature).fieldType match {
      case "Numerical" => {
        val featureData = data[NumericalValue](feature).filter(!_.isNaN)

        auxData match {
          case Some(auxData) => {
            val binsOnNewPartition = auxData.bins.get.filter(x => x >= featureData.min && x <= featureData.max)
            if (auxData.bins.isEmpty && binsOnNewPartition.length <= minSplitPoints) {
              val bins = TreeUtil.equidepthHist(featureData, maxSplitPoints)
              val maxRefined: Boolean = bins.length == featureData.distinct.length

              Some(SupportDict(bins = Some(bins), maxRefined = Some(maxRefined)))
            } else
              Some(SupportDict(bins = Some(binsOnNewPartition), maxRefined = auxData.maxRefined))
          }
          case None => {
            val bins = TreeUtil.equidepthHist(featureData, maxSplitPoints)
            val maxRefined: Boolean = bins.length <= maxSplitPoints

            Some(SupportDict(bins = Some(bins), maxRefined = Some(maxRefined)))
          }
        }
      }
      case "SimpleCategorical" | "ClassCategorical" => {
        throw new RuntimeException("Not implemented yet")
      }
      case _ => throw new RuntimeException("Uknown feature type")
    }
  }

  /**
   * Find the best split of data along the feature column.
   * @param feature
   * @param data
   * @param auxData result of of preprocessData, which contains information to allow for the
   *                efficient finding of the optimal split
   */
  private def findBestSplit(feature: String, data: DataFrame, auxData: Option[SupportDict]):
      BestSplit = {
    val bins: Vector[NumericalValue] = auxData.get.bins.get
    val numSplits: Int = bins.length

    // Check that feature values on this node are not of a single value
    if (numSplits < 2) return (NumericalValue(0), None, None, None)

    // Split data into samples where feature is mssing and not missing. On samples with a missing
    // feature, half the weights. Eventually check for feature data type.
    val featureIsMissing: Series[Boolean] = data.getSchema.nameToField(feature).fieldType match {
      case "Numerical" => data[NumericalValue](feature).map(_.isnan)
      case "SimpleCategorical" | "ClassCategorical" => Series.fill[Boolean](data.length)(false)
    }
    val missingData: DataFrame = data(featureIsMissing)
      .map(row => row.update(weightColName, row[NumericalValue](weightColName) * 0.5))
    val numMissing = missingData[NumericalValue](weightColName).summarizeResponse
    val validData = data(featureIsMissing.:!)

    // Checkt that after removing samples with features missing, that the node contains enough
    // samples to warrant a further split
    if (validData[NumericalValue](weightColName).sum < minSamplesSplit)
      return (NumericalValue(0), None, None, None)

    // Break data into chunks separted by split points, also evalute block summaries which
    // will be used to effeciently evalute cost function imporvement on distinct considered
    // splits
    val splitData: Buffer[DataFrame] = Buffer(validData)
    val blockSummaries: Buffer[BlockSummary] = Buffer.empty
    val splitPoints: Buffer[NumericalValue] = Buffer.empty
    for (i <- 0 until numSplits) {
      val splitPoint = NumericalValue(Uniform(bins(i)(), bins(i + 1)()).sample)

      val (leftData, rightData) = auxData.get.catMap match {
        case Some(catMap) =>
          splitData(i).partition(row => catMap(row[CategoricalValue](feature)()) >= splitPoint)
        case None => splitData(i).partition(row => row[NumericalValue](feature) >= splitPoint)
      }

      splitPoints append splitPoint
      splitData(splitData.length -1) = leftData
      splitData append rightData
      blockSummaries append summarizeResponse(leftData)
    }
    blockSummaries append summarizeResponse(splitData.last)

    // Evaluate all cost functions for all considered splits
    val afterSplitCosts: Buffer[NumericalValue] =
    for (i <- 0 until numSplits) {
      val leftBlockSummary = blockSummaries.slice(0, i + 1).reduce(reduceBlockSummary)
      val rightBlockSummary = blockSummaries.slice(i + 1, blockSummaries.length).reduce(reduceBlockSummary)

      afterSplitCosts append if ()
        evalCostFromBlock(leftBlockSummary) + evalCostFromBlock(rightBlockSummary)
      else NumericalValue(Double.PositiveInfinity)
    }

    // Pick best split and evaluate cost improvement
  }


  protected def getMaxFeaturesPerSplit: Int = predictorColNames.get.length

  private def getDistinctColumnName(columnNames: List[String], newNameBase: String): String = {
    var newName = newNameBase
    var i = 0

    while (columnNames contains newName) {
      newName =  newNameBase + i.toString
      i += 1
    }

    newName
  }
}

trait RegressionTreeLike {
  protected case class BlockSummary(
      val sum0: NumericalValue, val sum1: NumericalValue, val sum2: NumericalValue)
  protected def summarizeResponse = {}
  protected def evalCostFromBlock = {}
  protected def EvalResponseOnCat = {}
  protected def reduceBlockSummary = {}
}

trait ClassificationTreeLike
trait BivariateClassificationTreeLike extends ClassificationTreeLike
trait MultivariateClassificationTreeLike extends ClassificationTreeLike
trait DecorrelatedTreeLike

class RegressionTree extends DecisionTree with RegressionTreeLike

class DecorrelatedRegressionTree extends DecisionTree with RegressionTreeLike with DecorrelatedTreeLike
