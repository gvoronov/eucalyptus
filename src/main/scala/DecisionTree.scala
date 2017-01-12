package eucalyptus.decisiontree

import scala.math
import scala.language.implicitConversions
import scala.util.Random
import scala.util.control.Breaks._
// import scala.collection.mutable then Buffer -> mutable.Buffer and MutableMap -> mutable.Map
import scala.collection.mutable.{Map => MutableMap}
// import scala.collection.mutable.{ArrayBuffer => Buffer}
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
  implicit def wrapToLeft[A, B](x: A) = Left[A, B](x)
  implicit def wrapToRight[A, B](x: B) = Right[A, B](x)
}

sealed trait WithBlockSummary {
  protected class BlockSummary(val sum0: NumericalValue)
}

abstract class DecisionTree(
    val maxSplitPoints: Int, val minSplitPoints: Int, val maxDepth: Int, val minSamplesSplit: Int,
    val minSamplesLeaf: Int) extends WithBlockSummary {
  // abstract defs and classes required by fit method, actually implmented in sub traits
  // protected class BlockSummary(val sum0: NumericalValue)
  protected def summarizeResponse(data: DataFrame): BlockSummary
  protected def reduceBlockSummary(
      leftBlockSummary: BlockSummary, rightBlockSummary: BlockSummary): BlockSummary
  protected def evalCostFromBlock(blockSummary: BlockSummary): NumericalValue
  protected def createLeaf(responses: Series[DataValue]): Leaf
  protected def getCatMap(
      feature: String, data: DataFrame): Map[DataValue, NumericalValue]

  // A few type aliases to reduce horrible long expressoins
  protected type FeatureSupportDict = Option[MutableMap[String, Option[SupportDict]]]
  protected type BestSplit = Tuple4[
    NumericalValue, Option[NumericalValue], Option[DataFrame],  Option[DataFrame]]

  /**
   * Container for results of preprocessing that can be easily passed around
   * @param bins
   * @param maxRefined
   * @param catMap
   */
  protected case class SupportDict(
      val bins: Option[Vector[NumericalValue]] = None, val maxRefined: Option[Boolean] = None,
      val catMap: Option[Map[DataValue, NumericalValue]] = None)

  // Self properties set by fit method
  protected var tree: Option[BiTree] = None
  protected var predictorColNames: Option[List[String]] = None
  protected var responseColName: Option[String] = None
  protected var weightColName: Option[String] = None
  protected var myMaxFeaturesPerSplit: Option[Int] = None

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

    myMaxFeaturesPerSplit = Some(getMaxFeaturesPerSplit)
    fitRecursive(parsedX, 0)
  }
  def predict(x: DataFrame): Series[DataValue]
  def predict(x: Row): DataValue
  // def predict[T](x: DataFrame): Series[T] = x.map[T](predict[T](_))
  // def predict[T](x: DataFrame): Series[T] = tree.get.predict[T](x)
  // def predict[T](x: Row): T = tree.get.predict[T](x)

  /**
   * Fit a decision tree by finding the best split and recusrively doing so on each partition
   * of resulting split
   * @param data
   * @param depth
   * @param support
   * @param parent
   * @param key
   */
  private def fitRecursive(
      data: DataFrame, depth: Int, support: FeatureSupportDict = None,
      parent: Option[Node] = None, key: Option[Boolean] = None): Unit = {
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
        for (tmpFeature <- Random.shuffle(predictorColNames.get)) {
          mySupport(tmpFeature) = preprocessData(tmpFeature, data, mySupport(tmpFeature))
          val (tmpCostImprovement, tmpSplit, tmpLeftData, tmpRightData) =
            findBestSplit(tmpFeature, data, mySupport(tmpFeature))

          // Update current best split
          if (tmpCostImprovement > costImprovement) {
            costImprovement = tmpCostImprovement
            feature = Some(tmpFeature)
            split = tmpSplit
            leftData = tmpLeftData
            rightData = tmpRightData
          }

          // Ensure that no more than maxFeaturesPerSplit considered
          if (tmpCostImprovement > 0){
            numFeaturesConsidered += 1
            if (numFeaturesConsidered >= myMaxFeaturesPerSplit.get)
              break
          }
        }
      }
    }

    // Create child node or leaf (figure out by context)
    val child: Node = if (costImprovement > 0) {
      mySupport(feature.get).get.catMap match {
        case Some(catMap) => new BiNode(feature.get, split.get, catMap)
        case None => new BiNode(feature.get, split.get)
      }
    } else
      createLeaf(data[DataValue](responseColName.get))

    // Mount child in tree
    if (parent.isEmpty)
      tree = Some(new BiTree(child))
    else
      parent.get.setChild(key.get, child)

    // Fit deeper branches if current child node is not a leaf
    if (! child.isLeaf) {
      fitRecursive(leftData.get, depth+1, Some(mySupport.clone), Some(child), Some(false))
      fitRecursive(rightData.get, depth+1, Some(mySupport.clone), Some(child), Some(true))
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
        val catMap = getCatMap(feature, data)

        val bins = if (catMap.size < maxSplitPoints)
          catMap.values.toVector.sorted
        else
          TreeUtil.equidepthHist(data[DataValue](feature).map(catMap(_)), maxSplitPoints)

        Some(SupportDict(bins = Some(bins), catMap = Some(catMap)))
      }
      case _ => throw new RuntimeException("Uknown feature type")
    }
  }

  /**
   * Find the best split of data along the feature column.
   * @param feature consider a split in this feature of the data
   * @param data the dataframe to consider splitting
   * @param auxData result of of preprocessData, which contains information to allow for the
   *                efficient finding of the optimal split
   */
  private def findBestSplit(feature: String, data: DataFrame, auxData: Option[SupportDict]):
      BestSplit = {
    val bins: Vector[NumericalValue] = auxData.get.bins.get
    val numSplits: Int = bins.length - 1

    // Check that feature values on this node are not of a single value
    if (numSplits < 1) return (NumericalValue(0), None, None, None)

    // Split data into samples where feature is mssing and not missing. On samples with a missing
    // feature, half the weights. Eventually check for feature data type.
    val featureIsMissing: Series[Boolean] = data.getSchema.nameToField(feature).fieldType match {
      case "Numerical" => data[NumericalValue](feature).map(_.isNaN)
      case "SimpleCategorical" | "ClassCategorical" => Series.fill[Boolean](data.length)(false)
    }
    val missingData: DataFrame = data(featureIsMissing)
      .map(row => row.update(weightColName.get, row[NumericalValue](weightColName.get) * 0.5))
    val numMissing = missingData[NumericalValue](weightColName.get).sum
    val validData = data(featureIsMissing.:!)

    // Checkt that after removing samples with features missing, that the node contains enough
    // samples to warrant a further split
    if (validData[NumericalValue](weightColName.get).sum < minSamplesSplit)
      return (NumericalValue(0), None, None, None)

    // Break data into chunks separted by split points, also evalute block summaries which
    // will be used to effeciently evalute cost function imporvement on distinct considered
    // splits
    val splitData: Buffer[DataFrame] = Buffer(validData)
    val blockSummaries: Buffer[BlockSummary] = Buffer.empty
    val splitPoints: Buffer[NumericalValue] = Buffer.empty
    for (i <- 0 until numSplits) {
      val splitPoint = NumericalValue(Uniform(bins(i)(), bins(i + 1)()).sample)

      val (leftData, rightData): Tuple2[DataFrame, DataFrame] = auxData.get.catMap match {
        case Some(catMap) =>
          splitData(i).partition(row => catMap(row[CategoricalValue](feature)) < splitPoint)
        case None => splitData(i).partition(row => row[NumericalValue](feature) < splitPoint)
      }

      splitPoints append splitPoint
      splitData(splitData.length -1) = leftData
      splitData append rightData
      blockSummaries append summarizeResponse(leftData)
    }
    blockSummaries append summarizeResponse(splitData.last)

    // Evaluate all cost functions for all considered splits
    val afterSplitCosts: Buffer[NumericalValue] = Buffer.empty
    for (i <- 0 until numSplits) {
      val leftBlockSummary = blockSummaries.slice(0, i + 1).reduce(reduceBlockSummary)
      val rightBlockSummary = blockSummaries
        .slice(i + 1, blockSummaries.length).reduce(reduceBlockSummary)
      afterSplitCosts append (
        if (
            leftBlockSummary.sum0 + numMissing >= minSamplesLeaf &&
            rightBlockSummary.sum0 + numMissing >= minSamplesLeaf)
          evalCostFromBlock(leftBlockSummary) + evalCostFromBlock(rightBlockSummary)
        else
          NumericalValue.posInf
      )
    }

    // Pick best split and evaluate cost improvement
    val minAfterSplitCost: NumericalValue = afterSplitCosts.min
    if (minAfterSplitCost < NumericalValue.posInf) {
      val minCostIndex: Int = Random.shuffle(
        afterSplitCosts.zipWithIndex.filter(_._1  == minAfterSplitCost).map(_._2)).last
      val beforeSplitCost: NumericalValue = evalCostFromBlock(blockSummaries.reduce(reduceBlockSummary))
      val costImprovement: NumericalValue = beforeSplitCost - afterSplitCosts(minCostIndex)
      val splitPoint: NumericalValue = splitPoints(minCostIndex)
      val leftData: DataFrame =
        missingData + splitData.slice(0, minCostIndex + 1).reduce((a, b) => a + b)
      val rightData: DataFrame =
        missingData + splitData.slice(minCostIndex + 1, splitData.length).reduce((a, b) => a + b)
      (costImprovement, Some(splitPoint), Some(leftData), Some(rightData))
    } else
      (NumericalValue(0), None, None, None)
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

/**
 * This trait contains methods required for the fitting adn prediction on regression trees.
 */
sealed trait RegressionTreeLike extends WithBlockSummary {
  protected var predictorColNames: Option[List[String]]
  protected var responseColName: Option[String]
  protected var weightColName: Option[String]
  protected var tree: Option[BiTree]

  protected case class RegressionBlockSummary(
      override val sum0: NumericalValue, val sum1: NumericalValue, val sum2: NumericalValue)
      extends BlockSummary(sum0)

  protected def summarizeResponse(data: DataFrame): BlockSummary = {
    val weights: Series[NumericalValue] = data[NumericalValue](weightColName.get)
    val responses: Series[NumericalValue] = data[NumericalValue](responseColName.get)

    RegressionBlockSummary(
      weights.sum, (weights :* responses).sum, (weights :* (responses :** 2)).sum)
      .asInstanceOf[BlockSummary]
  }
  protected def evalCostFromBlock(blockSummary: BlockSummary): NumericalValue = {
    // Try by adding asInstanceOf to every term and factor
    val regressionBlockSummary = blockSummary.asInstanceOf[RegressionBlockSummary]
    regressionBlockSummary.sum2 - ((regressionBlockSummary.sum1**2) / regressionBlockSummary.sum0)
  }
  protected def reduceBlockSummary(
      leftBlockSummary: BlockSummary, rightBlockSummary: BlockSummary): BlockSummary = {
    val leftRegressionBlockSummary = leftBlockSummary.asInstanceOf[RegressionBlockSummary]
    val rightRegressionBlockSummary = rightBlockSummary.asInstanceOf[RegressionBlockSummary]
    RegressionBlockSummary(
      leftRegressionBlockSummary.sum0 + rightRegressionBlockSummary.sum0,
      leftRegressionBlockSummary.sum1 + rightRegressionBlockSummary.sum1,
      leftRegressionBlockSummary.sum2 + rightRegressionBlockSummary.sum2)
      .asInstanceOf[BlockSummary]
  }
  protected def createLeaf(responses: Series[DataValue]): Leaf =
    new RegressionLeaf(responses.asInstanceOf[Series[NumericalValue]]).asInstanceOf[Leaf]

  /**
   * Partition data by category, evaluate
   * @param feature
   * @param responseColName
   * @param data
   */
  protected def getCatMap(feature: String, data: DataFrame):
      Map[DataValue, NumericalValue] = {
    data.groupBy(row => row[DataValue](feature))
      .map(kvpair => kvpair._1 -> kvpair._2[NumericalValue](responseColName.get).mean)
      .toList.sortBy(_._2).map(_._1).zipWithIndex
      .map(pair => pair._1 -> NumericalValue(pair._2)).toMap
  }
  def predict(x: DataFrame): Series[NumericalValue] = tree.get.predict[NumericalValue](x)
  def predict(x: Row): NumericalValue = tree.get.predict[NumericalValue](x)
}

sealed trait ClassificationTreeLike
sealed trait BivariateClassificationTreeLike extends ClassificationTreeLike
sealed trait MultivariateClassificationTreeLike extends ClassificationTreeLike

sealed trait DecorrelatedTreeLike {
  protected var predictorColNames: Option[List[String]]
  protected val maxFeaturesPerSplit: Any
  protected def getDecorrelatedMaxFeaturesPerSplit: Int = {
    val numFeatures = predictorColNames.get.length
    math.min(numFeatures,
      maxFeaturesPerSplit match {
        case tag: String => tag match {
          case "sqrt" => math.ceil(math.sqrt(numFeatures)).toInt
          case "log2" => math.ceil(math.log(numFeatures) / math.log(2)).toInt
          case _ => throw new RuntimeException("Invalid argument passed to maxFeaturesPerSplit")
        }
        case x: Double => math.ceil(x * numFeatures).toInt
        case n: Int => n
        case _ => throw new RuntimeException("maxFeaturesPerSplit of invalid type passed")
      }
    )
  }
}

class RegressionTree(
    maxSplitPoints: Int = 10, minSplitPoints: Int = 1, maxDepth: Int = 100,
    minSamplesSplit: Int = 2, minSamplesLeaf: Int = 1)
    extends DecisionTree(maxSplitPoints, minSplitPoints, maxDepth, minSamplesSplit, minSamplesLeaf)
    with RegressionTreeLike

/**
 * Implementation of a decorrealted decision tree. This object is distinct from a standard
 * decision tree in that as splits are evaluated, only a random subset of features are
 * considered at each node. The goal is to ameliorate some of the downside of the greedy
 * training approach and to sample a wider space of trees. The arguments are the same as the
 * inputs to a standard decision tree, except for the inclusion of a new argument,
 * num_features_per_split. Can pass a float (consider only a fraction of the features), an int
 * (consider that many features), or a string. The two possible string inputs are 'sqrt'
 * consider a number of features given by the square root of the number of features  or
 * 'log2' (consider a number of features given by the log base 2 of the number of features).
 * @param maxSplitPoints
 * @param minSplitPoints
 * @param maxDepth
 * @param minSamplesSplit
 * @param minSamplesLeaf
 * @param maxFeaturesPerSplit
 */
class DecorrelatedRegressionTree(
    maxSplitPoints: Int = 10, minSplitPoints: Int = 1, maxDepth: Int = 100,
    minSamplesSplit: Int = 2, minSamplesLeaf: Int = 1,
    protected val maxFeaturesPerSplit: Any = "sqrt")
    extends DecisionTree(maxSplitPoints, minSplitPoints, maxDepth, minSamplesSplit, minSamplesLeaf)
    with RegressionTreeLike with DecorrelatedTreeLike {
  override protected def getMaxFeaturesPerSplit = getDecorrelatedMaxFeaturesPerSplit
}
